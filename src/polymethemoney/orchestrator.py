from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from inspect import isawaitable
from pathlib import Path

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger
from redis.asyncio import Redis

from polymethemoney.adapters import PaperExchange, PolymarketClient
from polymethemoney.config import Settings
from polymethemoney.db import create_engine_and_session, init_db
from polymethemoney.domain import FeatureVector, MarketTick, TradingMode
from polymethemoney.services import (
    CollectorService,
    ExecutionEngine,
    FeatureEngine,
    Gatekeeper,
    ModelEngine,
    ReporterService,
    RiskEngine,
    SignalEngine,
    StructureAlphaService,
    TelegramBotService,
)
from polymethemoney.state import RuntimeState
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)


class NotificationHub:
    def __init__(self) -> None:
        self._service: TelegramBotService | None = None

    def bind(self, service: TelegramBotService) -> None:
        self._service = service

    async def send(self, text: str) -> None:
        if self._service is None:
            logger.info("notification (telegram not bound): %s", text)
            return
        await self._service.notify(text)


class TradingApp:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.engine = None
        self.redis: Redis | None = None
        self.scheduler: AsyncIOScheduler | None = None
        self.notifier: NotificationHub | None = None
        self.store: any = None
        self.model_engine = None
        self.runtime = None
        self._paper_perf_tune_inflight = False
        self._last_paper_perf_tune_at: datetime | None = None

    async def run(self) -> None:
        self.engine, session_factory = create_engine_and_session(self.settings.database_url)
        await init_db(self.engine)
        self.redis = Redis.from_url(self.settings.redis_url, decode_responses=False)
        store = Store(session_factory, self.settings.starting_capital_usd)

        configured_mode = (
            TradingMode(self.settings.trading_mode.lower())
            if self.settings.trading_mode.lower() in ("paper", "live")
            else TradingMode.PAPER
        )
        runtime_mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else configured_mode
        if self.settings.demo_paper_hardlock and configured_mode != TradingMode.PAPER:
            logger.warning("DEMO_PAPER_HARDLOCK enabled: forcing TRADING_MODE from %s to paper", configured_mode.value)
        runtime = RuntimeState(trading_mode=runtime_mode)
        tick_queue: asyncio.Queue[MarketTick] = asyncio.Queue(maxsize=10_000)
        feature_queue: asyncio.Queue[FeatureVector] = asyncio.Queue(maxsize=10_000)
        notifier = NotificationHub()

        gatekeeper = Gatekeeper(self.settings, redis_client=self.redis)
        await gatekeeper.load_historical_from_file(Path("data/historical_metrics.json"))

        polymarket = PolymarketClient(self.settings)
        paper_exchange = PaperExchange(settings=self.settings)
        model_engine = ModelEngine(settings=self.settings, store=store, training_file=Path("data/training.csv"))

        risk_engine = RiskEngine(
            settings=self.settings,
            store=store,
            runtime_state=runtime,
            gatekeeper=gatekeeper,
            alert_fn=notifier.send,
        )
        execution_engine = ExecutionEngine(
            settings=self.settings,
            store=store,
            runtime_state=runtime,
            risk_engine=risk_engine,
            gatekeeper=gatekeeper,
            paper_exchange=paper_exchange,
            polymarket_client=polymarket,
            notify_fn=notifier.send,
        )
        reporter = ReporterService(
            settings=self.settings,
            store=store,
            runtime_state=runtime,
            risk_engine=risk_engine,
            gatekeeper=gatekeeper,
            notify_fn=notifier.send,
        )
        telegram = TelegramBotService(
            settings=self.settings,
            runtime_state=runtime,
            store=store,
            execution_engine=execution_engine,
            risk_engine=risk_engine,
            gatekeeper=gatekeeper,
            reporter=reporter,
        )
        notifier.bind(telegram)
        self.notifier = notifier
        self.store = store
        self.model_engine = model_engine
        self.runtime = runtime

        collector = CollectorService(
            settings=self.settings,
            client=polymarket,
            store=store,
            tick_queue=tick_queue,
            runtime_state=runtime,
        )
        feature_engine = FeatureEngine(
            store=store,
            tick_queue=tick_queue,
            feature_queue=feature_queue,
            paper_exchange=paper_exchange,
        )
        signal_engine = SignalEngine(
            settings=self.settings,
            feature_queue=feature_queue,
            model_engine=model_engine,
            execution_engine=execution_engine,
            runtime_state=runtime,
        )
        structure_alpha = StructureAlphaService(
            settings=self.settings,
            store=store,
            runtime_state=runtime,
            polymarket_client=polymarket,
            execution_engine=execution_engine,
            gatekeeper=gatekeeper,
            notify_fn=notifier.send,
        )

        if self.settings.demo_paper_hardlock and self.settings.demo_reset_paper_on_startup:
            reset = await store.reset_paper_open_positions()
            logger.warning(
                "paper startup reset executed by config: positions=%s structure_legs=%s structure_bundles=%s",
                reset["positions_closed"],
                reset["structure_legs_closed"],
                reset["structure_bundles_closed"],
            )
        elif self.settings.demo_paper_hardlock:
            logger.info("paper startup reset skipped (DEMO_RESET_PAPER_ON_STARTUP=false)")

        await model_engine.retrain_incremental()
        self.scheduler = AsyncIOScheduler(timezone="UTC")
        self.scheduler.add_job(
            model_engine.retrain_incremental,
            CronTrigger(hour=self.settings.incremental_retrain_hour_utc, minute=0),
            id="incremental_retrain",
            replace_existing=True,
        )
        self.scheduler.add_job(
            self._paper_performance_guard,
            "interval",
            hours=max(1, self.settings.paper_perf_tune_interval_hours),
            id="paper_perf_tune_check",
            replace_existing=True,
            coalesce=True,
            max_instances=1,
        )
        self.scheduler.add_job(
            model_engine.retrain_full,
            CronTrigger(
                day_of_week=self.settings.full_retrain_day_of_week,
                hour=self.settings.full_retrain_hour_utc,
                minute=0,
            ),
            id="full_retrain",
            replace_existing=True,
        )
        if self.settings.trend_intraday_retrain_minutes > 0:
            self.scheduler.add_job(
                model_engine.retrain_incremental,
                "interval",
                minutes=self.settings.trend_intraday_retrain_minutes,
                kwargs={"target": "intraday"},
                id="trend_intraday_retrain",
                replace_existing=True,
                coalesce=True,
                max_instances=1,
            )
        self.scheduler.start()

        tasks = [
            asyncio.create_task(collector.run(), name="collector"),
            asyncio.create_task(feature_engine.run(), name="feature_engine"),
            asyncio.create_task(signal_engine.run(), name="signal_engine"),
            asyncio.create_task(risk_engine.run(), name="risk_engine"),
            asyncio.create_task(reporter.run_periodic(), name="reporter"),
            asyncio.create_task(structure_alpha.run(), name="structure_alpha"),
            asyncio.create_task(telegram.run(), name="telegram"),
        ]
        try:
            await asyncio.gather(*tasks)
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            if self.scheduler is not None:
                self.scheduler.shutdown(wait=False)
            if self.redis is not None:
                await self.redis.aclose()
            if self.engine is not None:
                await self.engine.dispose()

    async def _paper_performance_guard(self) -> None:
        if not self.settings.paper_perf_tune_enabled:
            return
        if self.runtime is None or self.model_engine is None or self.notifier is None or self.store is None:
            return
        if self.settings.paper_perf_tune_only_in_paper and self.runtime.trading_mode != TradingMode.PAPER:
            return
        if self._paper_perf_tune_inflight:
            return

        self._paper_perf_tune_inflight = True
        try:
            performance = await self._collect_recent_paper_performance(self.settings.paper_perf_tune_interval_hours)
            if performance is None:
                return

            pnl_usd = performance["pnl_usd"]
            trades = performance["trades"]
            target = self.settings.paper_perf_tune_target.lower()
            trigger = pnl_usd <= self.settings.paper_perf_tune_min_pnl_usd and trades >= self.settings.paper_perf_tune_min_trades

            if not trigger:
                return

            self._last_paper_perf_tune_at = datetime.now(timezone.utc)
            await self._notify(
                (
                    "[paper 성능튜닝]\n"
                    f"최근 {self.settings.paper_perf_tune_interval_hours}시간 누적 수익: {pnl_usd:+.4f} USD\n"
                    f"체결 건수: {trades}\n"
                    f"임계치 미달 조건: pnl <= {self.settings.paper_perf_tune_min_pnl_usd:.4f} 및 거래 >= {self.settings.paper_perf_tune_min_trades}\n"
                    f"{target} 모델 재학습을 즉시 실행합니다."
                )
            )

            kwargs: dict[str, object] = {}
            if target in {"settlement", "intraday"}:
                kwargs["target"] = target
            await self._trigger_model_retrain(**kwargs)
        except Exception as exc:
            logger.exception("paper performance tuning failed")
            await self._notify(f"[paper 성능튜닝] 실패: {exc}")
        finally:
            self._paper_perf_tune_inflight = False

    async def _trigger_model_retrain(self, **kwargs: object) -> None:
        if not self.model_engine:
            return

        retrain_fn = getattr(self.model_engine, "retrain_incremental", None)
        if retrain_fn is None:
            await self._notify("[paper 성능튜닝] 재학습 함수를 찾지 못해 종료합니다.")
            return

        try:
            result = retrain_fn(**kwargs) if kwargs else retrain_fn()
            if isawaitable(result):
                await result
            await self._notify("[paper 성능튜닝] 재학습 완료. 누적수익 미달 조건을 반영해 모델을 즉시 업데이트했습니다.")
            logger.info("paper performance tune triggered: target=%s kwargs=%s", self.settings.paper_perf_tune_target, kwargs)
        except Exception as exc:
            logger.exception("paper performance retrain failed")
            await self._notify(f"[paper 성능튜닝] 재학습 실패: {exc}")

    async def _collect_recent_paper_performance(self, window_hours: int) -> dict[str, float | int] | None:
        # 지원되는 store 메서드 우선순위로 성능 지표 조회(최신 구현체 기준)
        candidates = [
            ("get_recent_paper_performance", {"hours": window_hours}),
            ("get_paper_performance", {"hours": window_hours}),
            ("get_performance_summary", {"hours": window_hours}),
            ("get_report_window", {"window_hours": window_hours}),
        ]

        for name, kwargs in candidates:
            method = getattr(self.store, name, None)
            if method is None:
                continue
            try:
                result = method(**kwargs)
                if isawaitable(result):
                    result = await result
                perf = self._parse_performance_payload(result)
                if perf is not None:
                    return perf
            except Exception:
                logger.exception("paper performance collect failed: %s", name)
                continue

        logger.warning("paper performance source 미정의: 수동 튜닝 스킵")
        return None

    def _parse_performance_payload(self, payload: object) -> dict[str, float | int] | None:
        if payload is None:
            return None
        if isinstance(payload, tuple) and len(payload) >= 2:
            pnl = self._to_float(payload[0])
            trades = self._to_int(payload[1])
            if pnl is not None and trades is not None:
                return {"pnl_usd": pnl, "trades": trades}

        if isinstance(payload, dict):
            pnl = self._pick_first_float(
                payload,
                keys=("pnl_usd", "profit_usd", "net_pnl_usd", "cumulative_pnl_usd"),
                default=None,
            )
            trades = self._pick_first_int(payload, keys=("trades", "trade_count", "num_trades", "filled_trades"), default=0)
            if pnl is None:
                realized = self._pick_first_float(payload, keys=("realized_pnl_usd", "realized"), default=0.0)
                unrealized = self._pick_first_float(payload, keys=("unrealized_pnl_usd", "unrealized"), default=0.0)
                pnl = (realized or 0.0) + (unrealized or 0.0)
            if pnl is None or trades is None:
                return None
            return {"pnl_usd": pnl, "trades": trades}

        for attr in ("pnl_usd", "pnl", "profit_usd", "net_pnl_usd"):
            if hasattr(payload, attr):
                pnl = self._to_float(getattr(payload, attr))
                break
        else:
            pnl = None

        for attr in ("trades", "trade_count", "num_trades"):
            if hasattr(payload, attr):
                trades = self._to_int(getattr(payload, attr))
                break
        else:
            trades = 0

        if pnl is None or trades is None:
            return None
        return {"pnl_usd": pnl, "trades": trades}

    def _pick_first_float(self, payload: dict, keys: tuple[str, ...], default: float | None) -> float | None:
        for key in keys:
            val = payload.get(key)
            if val is not None:
                return self._to_float(val) if not isinstance(val, str) else self._to_float(val.strip())
        return default

    def _pick_first_int(self, payload: dict, keys: tuple[str, ...], default: int) -> int | None:
        for key in keys:
            val = payload.get(key)
            if val is not None:
                return self._to_int(val)
        return default

    def _to_float(self, value: object) -> float | None:
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _to_int(self, value: object) -> int | None:
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    async def _notify(self, text: str) -> None:
        if self.notifier is None:
            return
        await self.notifier.send(text)
