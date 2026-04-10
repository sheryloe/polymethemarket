from __future__ import annotations

import asyncio
from collections import Counter
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from polymethemoney.config import Settings
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
        await self.send_report()
        while True:
            await asyncio.sleep(interval_sec)
            await self.send_report()

    async def send_report(self) -> None:
        text = await self.build_report60_text(include_tune=False)
        await self.notify_fn(text)

    async def build_periodic_text(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        snapshot = await self.store.status_snapshot()
        await self.risk_engine.refresh_state()
        summary = await self.store.signal_window_summary(window_minutes=60)
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
        _, unrealized_total, pos_plus, pos_flat, pos_minus = await self._position_pnl_stats(open_positions)

        mode_label = self.runtime_state.trading_mode.value.upper()
        run_state = "\uc911\uc9c0" if self.runtime_state.paused else "\uc6b4\uc601"
        live_lock = "ON" if self.settings.demo_paper_hardlock else "OFF"
        pending = len(self.runtime_state.pending_approvals)
        reject_text = self._format_reject_top(summary.get("reject_top", []))
        fill_rate = summary["fill_rate"]

        return (
            "[10\ubd84 \ub204\uc801]\n"
            f"{now} | \ubaa8\ub4dc {mode_label} | \uc0c1\ud0dc {run_state} | \ud558\ub4dc\ub77d {live_lock}\n"
            f"\uc624\ud508 \ud3ec\uc9c0\uc158 {len(open_positions)}(+{pos_plus}/0:{pos_flat}/-{pos_minus}) | \uc2b9\uc778\ub300\uae30 {pending}\n"
            f"\uc77c/\uc8fc \uc190\uc775 {snapshot['day_pnl']:+.2f}/{snapshot['week_pnl']:+.2f} USD | \ubbf8\uc2e4\ud604 {unrealized_total:+.2f} USD\n"
            f"\ucd5c\uadfc 60\ubd84 \uc2e0\ud638 {summary['total_signals']} | \uccb4\uacb0 {summary['filled_signals']}({fill_rate:.0%}) | \uac70\uc808 {reject_text}"
        )

    async def build_status_text(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        snapshot = await self.store.status_snapshot()
        risk_state = await self.risk_engine.refresh_state()
        summary_60m = await self.store.signal_window_summary(window_minutes=60)
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
        _, unrealized_total, pos_plus, pos_flat, pos_minus = await self._position_pnl_stats(open_positions)

        mode_label = self.runtime_state.trading_mode.value.upper()
        run_state = "\uc911\uc9c0" if self.runtime_state.paused else "\uc6b4\uc601"
        live_lock = "ON" if self.settings.demo_paper_hardlock else "OFF"
        pending = len(self.runtime_state.pending_approvals)
        reject_text_60m = self._format_reject_top(summary_60m.get("reject_top", []))

        return (
            "[\uc0c1\ud0dc \uc694\uc57d]\n"
            f"{now} | \ubaa8\ub4dc {mode_label} | \uc0c1\ud0dc {run_state} | \ud558\ub4dc\ub77d {live_lock}\n"
            f"\uc624\ud508 \ud3ec\uc9c0\uc158 {len(open_positions)}(+{pos_plus}/0:{pos_flat}/-{pos_minus}) | \uc2b9\uc778\ub300\uae30 {pending}\n"
            f"DD \uc77c/\uc8fc {risk_state.daily_drawdown_pct:.2%}/{risk_state.weekly_drawdown_pct:.2%}\n"
            f"\uc77c/\uc8fc \uc190\uc775 {snapshot['day_pnl']:+.2f}/{snapshot['week_pnl']:+.2f} USD | \ubbf8\uc2e4\ud604 {unrealized_total:+.2f} USD\n"
            f"\ucd5c\uadfc 60\ubd84 \uc2e0\ud638 {summary_60m['total_signals']} | \uccb4\uacb0 {summary_60m['filled_signals']}({summary_60m['fill_rate']:.0%}) | \uac70\uc808 {reject_text_60m}"
        )

    async def build_report60_text(self, include_tune: bool = True) -> str:
        tune_message = None
        if include_tune:
            tune_message = await self.maybe_apply_zero_fill_tuning()
        summary = await self.store.signal_window_summary(window_minutes=60)
        snapshot = await self.store.status_snapshot()
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
        _, unrealized_total, pos_plus, pos_flat, pos_minus = await self._position_pnl_stats(open_positions)
        reject_text = self._format_reject_top(summary.get("reject_top", []))
        no_guard_text = self._format_no_guard_top(window_minutes=60)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        body = (
            "[60\ubd84 \ub204\uc801]\n"
            f"{now}\n"
            f"\uc2e0\ud638 {summary['total_signals']} | \uccb4\uacb0 {summary['filled_signals']}({summary['fill_rate']:.0%}) | YES/NO {summary['yes_signals']}/{summary['no_signals']}\n"
            f"\uac70\uc808 {reject_text} | NO \uac00\ub4dc {no_guard_text}\n"
            f"\uc2e4\ud604 {summary['realized_pnl']:+.2f} | \ubbf8\uc2e4\ud604 {unrealized_total:+.2f} | \uc624\ud508 \ud3ec\uc9c0\uc158 {snapshot['open_positions']}(+{pos_plus}/0:{pos_flat}/-{pos_minus})"
        )
        if tune_message:
            return f"{body}\n\n{tune_message}"
        return body

    async def build_report6h_text(self) -> str:
        summary = await self.store.signal_window_summary(window_minutes=360)
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
        _, unrealized_total, pos_plus, pos_flat, pos_minus = await self._position_pnl_stats(open_positions)
        reject_text = self._format_reject_top(summary.get("reject_top", []))
        no_guard_text = self._format_no_guard_top(window_minutes=360)
        six_hour_total_pnl = float(summary["realized_pnl"]) + float(unrealized_total)
        six_hour_return_rate = six_hour_total_pnl / float(self.settings.starting_capital_usd)
        side_policy = (self.settings.signal_side_policy or "BALANCED").strip().upper()
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        return (
            "[6\uc2dc\uac04 \ub204\uc801]\n"
            f"{now}\n"
            f"\uc815\ucc45 {side_policy} | 6\uc2dc\uac04 \uc218\uc775\ub960 {six_hour_return_rate:+.2%}\n"
            f"\uc2e0\ud638 {summary['total_signals']} | \uccb4\uacb0 {summary['filled_signals']}({summary['fill_rate']:.0%}) | YES/NO {summary['yes_signals']}/{summary['no_signals']}\n"
            f"\uac70\uc808 {reject_text} | NO \uac00\ub4dc {no_guard_text}\n"
            f"\uc2e4\ud604 {summary['realized_pnl']:+.2f} | \ubbf8\uc2e4\ud604 {unrealized_total:+.2f} | \uc624\ud508 \ud3ec\uc9c0\uc158 {len(open_positions)}(+{pos_plus}/0:{pos_flat}/-{pos_minus})"
        )

    async def maybe_apply_zero_fill_tuning(self) -> str | None:
        now = datetime.now(timezone.utc)
        self.runtime_state.auto_tune_zero_fill_last_checked_at = now
        if not self.settings.auto_tune_zero_fill_enabled:
            return self._build_tune_notice("disabled", now, "\uc790\ub3d9 \ud29c\ub2dd \ube44\ud65c\uc131\ud654")

        summary = await self.store.signal_window_summary(window_minutes=self.settings.auto_tune_window_minutes)
        total_signals = int(summary["total_signals"])
        filled = int(summary["filled_signals"])
        min_signals = max(1, int(self.settings.auto_tune_min_signals))
        if total_signals < min_signals:
            return self._build_tune_notice(
                "insufficient_signals",
                now,
                f"\uc2e0\ud638 \ubd80\uc871 ({total_signals}/{min_signals})",
            )
        if filled > 0:
            return self._build_tune_notice("has_fills", now, f"\uccb4\uacb0 \uc874\uc7ac ({filled}\uac74)")
        if self.settings.auto_tune_apply_once and self.runtime_state.auto_tune_zero_fill_applied:
            return self._build_tune_notice("already_applied", now, "\uc774\ubbf8 1\ud68c \uc801\uc6a9\ub428")

        base_min = float(self.settings.min_position_usd)
        target_min = max(0.1, float(self.settings.auto_tune_min_position_usd))
        if target_min >= base_min:
            return self._build_tune_notice(
                "target_not_lower",
                now,
                f"\ubaa9\ud45c \ucd5c\uc18c\uae08\uc561\uc774 \uae30\uc874\ubcf4\ub2e4 \ub0ae\uc9c0 \uc54a\uc74c (base {base_min:.2f} / target {target_min:.2f})",
            )

        self.runtime_state.min_position_usd_override = target_min
        self.runtime_state.auto_tune_zero_fill_applied = True
        self.runtime_state.auto_tune_zero_fill_applied_at = now
        return self._build_tune_notice(
            "applied",
            now,
            f"\ubb34\uccb4\uacb0 0\uac74 \uac10\uc9c0: \ucd5c\uc18c\uae08\uc561 \ub0ae\ucda4 {base_min:.2f} -> {target_min:.2f} USD",
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

    def _build_tune_notice(self, outcome: str, now: datetime, detail: str) -> str | None:
        if self.runtime_state.auto_tune_zero_fill_last_outcome == outcome:
            return None
        self.runtime_state.auto_tune_zero_fill_last_outcome = outcome
        checked_at = now.strftime("%H:%M:%S UTC")
        return (
            "[\uc790\ub3d9 \ud29c\ub2dd]\n"
            f"\uc708\ub3c4\uc6b0 {self.settings.auto_tune_window_minutes}\ubd84\n"
            f"\uacb0\uacfc {outcome}\n"
            f"\uc0c1\uc138 {detail}\n"
            f"\ud655\uc778 {checked_at}"
        )

    def _format_reject_top(self, reject_top: object) -> str:
        if not isinstance(reject_top, list) or not reject_top:
            return "\uc5c6\uc74c"
        return ", ".join(f"{str(status).replace('rejected:', '')} {count}\uac74" for status, count in reject_top)

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
        return ", ".join(f"{reason} {count}\uac74" for reason, count in top) if top else "\uc5c6\uc74c"


