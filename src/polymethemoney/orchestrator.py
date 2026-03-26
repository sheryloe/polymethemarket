from __future__ import annotations

import asyncio
import logging
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
