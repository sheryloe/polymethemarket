from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from polymethemoney.config import Settings
from polymethemoney.domain import STRATEGY_MODEL_A, STRATEGY_MODEL_B
from polymethemoney.services.gatekeeper import Gatekeeper
from polymethemoney.services.risk_engine import RiskEngine
from polymethemoney.state import RuntimeState, TimedReasonEvent
from polymethemoney.storage import Store

NotifyFn = Callable[[str], Awaitable[None]]


class ReporterService:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        runtime_state: RuntimeState,
        risk_engine: RiskEngine,
        gatekeeper: Gatekeeper,
        notify_fn: NotifyFn,
    ) -> None:
        self.settings = settings
        self.store = store
        self.runtime_state = runtime_state
        self.risk_engine = risk_engine
        self.gatekeeper = gatekeeper
        self.notify_fn = notify_fn

    async def run_periodic(self) -> None:
        interval_sec = max(60, int(self.settings.report_interval_minutes) * 60)
        while True:
            await asyncio.sleep(interval_sec)
            await self.send_report()

    async def send_report(self) -> None:
        if self.settings.ab_test_enabled:
            text = await self.build_report30_text()
        else:
            text = await self.build_report60_text(include_tune=False)
        await self.notify_fn(text)

    async def build_periodic_text(self) -> str:
        if self.settings.ab_test_enabled:
            return await self.build_report30_text()

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        snapshot = await self.store.status_snapshot()
        await self.risk_engine.refresh_state()
        summary = await self.store.signal_window_summary(window_minutes=60)
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
        _, unrealized_total, pos_plus, pos_flat, pos_minus = await self._position_pnl_stats(open_positions)

        mode_label = self.runtime_state.trading_mode.value.upper()
        run_state = "중지" if self.runtime_state.paused else "운영"
        live_lock = "ON" if self.settings.demo_paper_hardlock else "OFF"
        pending = len(self.runtime_state.pending_approvals)
        reject_text = self._format_reject_top(summary.get("reject_top", []))
        fill_rate = summary["fill_rate"]

        return (
            "[10분 누적]\n"
            f"{now} | 모드 {mode_label} | 상태 {run_state} | 하드락 {live_lock}\n"
            f"오픈 포지션 {len(open_positions)}(+{pos_plus}/0:{pos_flat}/-{pos_minus}) | 승인대기 {pending}\n"
            f"일/주 손익 {snapshot['day_pnl']:+.2f}/{snapshot['week_pnl']:+.2f} USD | 미실현 {unrealized_total:+.2f} USD\n"
            f"최근 60분 신호 {summary['total_signals']} | 체결 {summary['filled_signals']}({fill_rate:.0%}) | 거절 {reject_text}"
        )

    async def build_status_text(self) -> str:
        if self.settings.ab_test_enabled:
            return await self._build_ab_status_text()

        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        snapshot = await self.store.status_snapshot()
        risk_state = await self.risk_engine.refresh_state()
        summary_60m = await self.store.signal_window_summary(window_minutes=60)
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
        _, unrealized_total, pos_plus, pos_flat, pos_minus = await self._position_pnl_stats(open_positions)

        mode_label = self.runtime_state.trading_mode.value.upper()
        run_state = "중지" if self.runtime_state.paused else "운영"
        live_lock = "ON" if self.settings.demo_paper_hardlock else "OFF"
        pending = len(self.runtime_state.pending_approvals)
        reject_text_60m = self._format_reject_top(summary_60m.get("reject_top", []))

        return (
            "[상태 요약]\n"
            f"{now} | 모드 {mode_label} | 상태 {run_state} | 하드락 {live_lock}\n"
            f"오픈 포지션 {len(open_positions)}(+{pos_plus}/0:{pos_flat}/-{pos_minus}) | 승인대기 {pending}\n"
            f"DD 일/주 {risk_state.daily_drawdown_pct:.2%}/{risk_state.weekly_drawdown_pct:.2%}\n"
            f"일/주 손익 {snapshot['day_pnl']:+.2f}/{snapshot['week_pnl']:+.2f} USD | 미실현 {unrealized_total:+.2f} USD\n"
            f"최근 60분 신호 {summary_60m['total_signals']} | 체결 {summary_60m['filled_signals']}({summary_60m['fill_rate']:.0%}) | 거절 {reject_text_60m}"
        )

    async def build_report30_text(self) -> str:
        if not self.settings.ab_test_enabled:
            return await self._build_single_report(window_minutes=30, title="[30분 누적]")
        return await self._build_ab_compare_text(window_minutes=30, title="[30분 A/B 비교]")

    async def build_report60_text(self, include_tune: bool = True) -> str:
        if self.settings.ab_test_enabled:
            return await self._build_ab_compare_text(window_minutes=60, title="[60분 A/B 비교]")

        tune_message = None
        if include_tune:
            tune_message = await self.maybe_apply_zero_fill_tuning()
        body = await self._build_single_report(window_minutes=60, title="[60분 누적]")
        if tune_message:
            return f"{body}\n\n{tune_message}"
        return body

    async def build_report6h_text(self) -> str:
        if self.settings.ab_test_enabled:
            return await self._build_ab_compare_text(window_minutes=360, title="[6시간 A/B 비교]")
        return await self._build_single_report(window_minutes=360, title="[6시간 누적]")

    async def maybe_apply_zero_fill_tuning(self) -> str | None:
        if self.settings.ab_test_enabled:
            return None
        now = datetime.now(timezone.utc)
        self.runtime_state.auto_tune_zero_fill_last_checked_at = now
        if not self.settings.auto_tune_zero_fill_enabled:
            return self._build_tune_notice("disabled", now, "자동 튜닝 비활성화")

        summary = await self.store.signal_window_summary(window_minutes=self.settings.auto_tune_window_minutes)
        total_signals = int(summary["total_signals"])
        filled = int(summary["filled_signals"])
        min_signals = max(1, int(self.settings.auto_tune_min_signals))
        if total_signals < min_signals:
            return self._build_tune_notice("insufficient_signals", now, f"신호 부족 ({total_signals}/{min_signals})")
        if filled > 0:
            return self._build_tune_notice("has_fills", now, f"체결 존재 ({filled}건)")
        if self.settings.auto_tune_apply_once and self.runtime_state.auto_tune_zero_fill_applied:
            return self._build_tune_notice("already_applied", now, "이미 1회 적용됨")

        base_min = float(self.settings.min_position_usd)
        target_min = max(0.1, float(self.settings.auto_tune_min_position_usd))
        if target_min >= base_min:
            return self._build_tune_notice(
                "target_not_lower",
                now,
                f"목표 최소금액이 기존보다 낮지 않음 (base {base_min:.2f} / target {target_min:.2f})",
            )

        self.runtime_state.min_position_usd_override = target_min
        self.runtime_state.auto_tune_zero_fill_applied = True
        self.runtime_state.auto_tune_zero_fill_applied_at = now
        return self._build_tune_notice(
            "applied",
            now,
            f"무체결 감지: 최소금액 조정 {base_min:.2f} -> {target_min:.2f} USD",
        )

    async def _build_ab_status_text(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        risk_state = await self.risk_engine.refresh_state()
        mode_label = self.runtime_state.trading_mode.value.upper()
        run_state = "중지" if self.runtime_state.paused else "운영"
        live_lock = "ON" if self.settings.demo_paper_hardlock else "OFF"
        metrics_a = await self._ab_strategy_metrics(STRATEGY_MODEL_A, window_minutes=60)
        metrics_b = await self._ab_strategy_metrics(STRATEGY_MODEL_B, window_minutes=60)

        return (
            "[상태 요약]\n"
            f"{now} | 모드 {mode_label} | 상태 {run_state} | 하드락 {live_lock}\n"
            f"DD 일/주 {risk_state.daily_drawdown_pct:.2%}/{risk_state.weekly_drawdown_pct:.2%}\n"
            f"모델 A 오픈 {metrics_a['open_positions']} | 누적 {metrics_a['total_pnl']:+.2f} USD ({metrics_a['return_rate']:+.2%}) | 승률 {metrics_a['win_rate']:.0%}\n"
            f"모델 B 오픈 {metrics_b['open_positions']} | 누적 {metrics_b['total_pnl']:+.2f} USD ({metrics_b['return_rate']:+.2%}) | 승률 {metrics_b['win_rate']:.0%}\n"
            f"최근 60분 우위 {self._leader_text(metrics_a['window_total_pnl'], metrics_b['window_total_pnl'])} | 누적 우위 {self._leader_text(metrics_a['total_pnl'], metrics_b['total_pnl'])}"
        )

    async def _build_ab_compare_text(self, window_minutes: int, title: str) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        metrics_a = await self._ab_strategy_metrics(STRATEGY_MODEL_A, window_minutes=window_minutes)
        metrics_b = await self._ab_strategy_metrics(STRATEGY_MODEL_B, window_minutes=window_minutes)
        return (
            f"{title}\n"
            f"{now}\n"
            f"모델 A | 신호 {metrics_a['signals']} | 체결 {metrics_a['fills']}({metrics_a['fill_rate']:.0%}) | 실현 {metrics_a['realized_pnl']:+.2f} | 미실현 {metrics_a['unrealized_pnl']:+.2f} | 승률 {metrics_a['win_rate']:.0%} | 오픈 {metrics_a['open_positions']}\n"
            f"모델 B | 신호 {metrics_b['signals']} | 체결 {metrics_b['fills']}({metrics_b['fill_rate']:.0%}) | 실현 {metrics_b['realized_pnl']:+.2f} | 미실현 {metrics_b['unrealized_pnl']:+.2f} | 승률 {metrics_b['win_rate']:.0%} | 오픈 {metrics_b['open_positions']}\n"
            f"최근 {window_minutes}분 우위 {self._leader_text(metrics_a['window_total_pnl'], metrics_b['window_total_pnl'])}\n"
            f"누적 우위 {self._leader_text(metrics_a['total_pnl'], metrics_b['total_pnl'])}"
        )

    async def _ab_strategy_metrics(self, strategy_id: str, window_minutes: int) -> dict[str, float | int]:
        summary = await self.store.signal_window_summary(window_minutes=window_minutes, strategy_id=strategy_id)
        snapshot = await self.store.status_snapshot(strategy_id=strategy_id)
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode, strategy_id=strategy_id)
        _, unrealized_total, _, _, _ = await self._position_pnl_stats(open_positions)
        capital = float(self.settings.model_portfolio_starting_capital_usd)
        total_pnl = float(snapshot["total_pnl"]) + unrealized_total
        return {
            "signals": int(summary["total_signals"]),
            "fills": int(summary["filled_signals"]),
            "fill_rate": float(summary["fill_rate"]),
            "realized_pnl": float(summary["realized_pnl"]),
            "unrealized_pnl": float(unrealized_total),
            "open_positions": int(snapshot["open_positions"]),
            "win_rate": float(summary["win_rate"]),
            "window_total_pnl": float(summary["realized_pnl"]) + float(unrealized_total),
            "total_pnl": total_pnl,
            "return_rate": total_pnl / max(1.0, capital),
        }

    async def _build_single_report(self, window_minutes: int, title: str) -> str:
        summary = await self.store.signal_window_summary(window_minutes=window_minutes)
        snapshot = await self.store.status_snapshot()
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
        _, unrealized_total, pos_plus, pos_flat, pos_minus = await self._position_pnl_stats(open_positions)
        reject_text = self._format_reject_top(summary.get("reject_top", []))
        no_guard_text = self._format_no_guard_top(window_minutes=window_minutes)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        label = f"{window_minutes}분"
        if title == "[6시간 누적]":
            label = "6시간"
        return (
            f"{title}\n"
            f"{now}\n"
            f"신호 {summary['total_signals']} | 체결 {summary['filled_signals']}({summary['fill_rate']:.0%}) | YES/NO {summary['yes_signals']}/{summary['no_signals']}\n"
            f"거절 {reject_text} | NO 가드 {no_guard_text}\n"
            f"실현 {summary['realized_pnl']:+.2f} | 미실현 {unrealized_total:+.2f} | 오픈 포지션 {snapshot['open_positions']}(+{pos_plus}/0:{pos_flat}/-{pos_minus})\n"
            f"{label} 승률 {float(summary['win_rate']):.0%}"
        )

    async def _position_pnl_stats(self, open_positions: list) -> tuple[list[str], float, int, int, int]:
        market_ids = [str(row.market_id) for row in open_positions]
        latest_prices = await self.store.latest_market_prices(market_ids)
        lines: list[str] = []
        unrealized_total = 0.0
        pos_plus = 0
        pos_flat = 0
        pos_minus = 0
        for row in open_positions:
            side = str(row.side).upper()
            market_id = str(row.market_id)
            yes_price = latest_prices.get(market_id, float(row.entry_price))
            mark_price = max(0.001, min(0.999, 1.0 - yes_price)) if side == "NO" else max(0.001, min(0.999, yes_price))
            shares = float(row.size_usd) / max(float(row.entry_price), 0.001)
            unrealized = (shares * mark_price) - float(row.size_usd)
            unrealized_total += unrealized
            if unrealized > 1e-6:
                pos_plus += 1
            elif unrealized < -1e-6:
                pos_minus += 1
            else:
                pos_flat += 1
            lines.append(
                f"{side} {float(row.size_usd):.2f} USD @ {float(row.entry_price):.4f} -> {mark_price:.4f} ({unrealized:+.2f})"
            )
        return lines, unrealized_total, pos_plus, pos_flat, pos_minus

    def _leader_text(self, pnl_a: float, pnl_b: float) -> str:
        if abs(pnl_a - pnl_b) < 1e-9:
            return f"동률 (A {pnl_a:+.2f} / B {pnl_b:+.2f} USD)"
        if pnl_a > pnl_b:
            return f"모델 A (A {pnl_a:+.2f} / B {pnl_b:+.2f} USD)"
        return f"모델 B (A {pnl_a:+.2f} / B {pnl_b:+.2f} USD)"

    def _build_tune_notice(self, outcome: str, now: datetime, detail: str) -> str | None:
        if self.runtime_state.auto_tune_zero_fill_last_outcome == outcome:
            return None
        self.runtime_state.auto_tune_zero_fill_last_outcome = outcome
        checked_at = now.strftime("%H:%M:%S UTC")
        return (
            "[자동 튜닝]\n"
            f"윈도우 {self.settings.auto_tune_window_minutes}분\n"
            f"결과 {outcome}\n"
            f"상세 {detail}\n"
            f"확인 {checked_at}"
        )

    def _format_reject_top(self, reject_top: object) -> str:
        if not isinstance(reject_top, list) or not reject_top:
            return "없음"
        return ", ".join(f"{str(status).replace('rejected:', '')} {count}건" for status, count in reject_top)

    def _format_no_guard_top(self, window_minutes: int) -> str:
        cutoff = datetime.now(timezone.utc) - timedelta(minutes=max(1, int(window_minutes)))
        reasons: list[str] = []
        for event in self.runtime_state.recent_no_guard_rejects:
            if isinstance(event, TimedReasonEvent):
                if event.timestamp >= cutoff:
                    reasons.append(event.reason)
                continue
            if isinstance(event, tuple) and len(event) == 2 and isinstance(event[0], datetime):
                if event[0] >= cutoff:
                    reasons.append(str(event[1]))
                continue
            if isinstance(event, str):
                reasons.append(event)
        top = Counter(reasons).most_common(3)
        return ", ".join(f"{reason} {count}건" for reason, count in top) if top else "없음"
