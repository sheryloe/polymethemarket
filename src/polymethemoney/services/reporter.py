from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Awaitable, Callable

from polymethemoney.config import Settings
from polymethemoney.domain import EXECUTION_MODE_FUNDED_PAPER, STRATEGY_EXPIRY_ANCHOR, STRATEGY_TAPE_RIDER, get_strategy_spec, iter_active_strategy_specs
from polymethemoney.services.gatekeeper import Gatekeeper
from polymethemoney.services.risk_engine import RiskEngine
from polymethemoney.state import RuntimeState
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
        await self.notify_fn(await self.build_report60_text(include_tune=False))

    async def build_periodic_text(self) -> str:
        return await self.build_report60_text(include_tune=False)

    async def build_status_text(self) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        risk_state = await self.risk_engine.refresh_state()
        run_state = "중지" if self.runtime_state.paused else "운영"
        mode_label = self.runtime_state.trading_mode.value.upper()
        lines = [
            "[상태 요약]",
            f"{now} | 모드 {mode_label} | 상태 {run_state} | 하드락 {'ON' if self.settings.demo_paper_hardlock else 'OFF'}",
            f"DD 일/주 {risk_state.daily_drawdown_pct:.2%}/{risk_state.weekly_drawdown_pct:.2%}",
        ]
        for spec in iter_active_strategy_specs():
            metrics = await self._strategy_metrics(spec.strategy_id, window_minutes=60)
            lines.append(
                f"{spec.display_name} | {self._mode_label(spec.execution_mode)} | 실현 {metrics['realized_pnl']:+.2f} | 미실현 {metrics['unrealized_pnl']:+.2f} | 오픈 {metrics['open_positions']}"
            )
        return "\n".join(lines)

    async def build_report30_text(self) -> str:
        return await self._build_compare_report(window_minutes=30, title="[30분 전략 비교]")

    async def build_report60_text(self, include_tune: bool = True) -> str:
        return await self._build_compare_report(window_minutes=60, title="[60분 누적]")

    async def build_report6h_text(self) -> str:
        return await self._build_compare_report(window_minutes=360, title="[6시간 누적]")

    async def _build_compare_report(self, window_minutes: int, title: str) -> str:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        specs = list(iter_active_strategy_specs())
        lines = [title, now]
        for spec in specs:
            metrics = await self._strategy_metrics(spec.strategy_id, window_minutes=window_minutes)
            lines.append(
                f"{spec.display_name} | {self._mode_label(spec.execution_mode)} | 신호 {metrics['signals']} / 체결 {metrics['fills']}({metrics['fill_rate']:.0%}) / 승률 {metrics['win_rate']:.0%} / 실현 {metrics['realized_pnl']:+.2f} / 미실현 {metrics['unrealized_pnl']:+.2f} / 오픈 {metrics['open_positions']}"
            )
        if len(specs) >= 2:
            compare = await self.store.strategy_compare_summary(window_minutes, specs[0].strategy_id, specs[1].strategy_id)
            lines.append(
                f"비교 메모 | same-side overlap {float(compare['same_side_overlap']):.0%} | PnL corr {float(compare['pnl_correlation']):+.2f} | fallback_mark_close {int(compare['fallback_mark_close_count'])}"
            )
        return "\n".join(lines)

    async def _strategy_metrics(self, strategy_id: str, window_minutes: int) -> dict[str, float | int]:
        spec = get_strategy_spec(strategy_id)
        summary = await self.store.signal_window_summary(window_minutes=window_minutes, strategy_id=strategy_id)
        snapshot = await self.store.status_snapshot(strategy_id=strategy_id)
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode, strategy_id=strategy_id)
        unrealized = await self._unrealized_total(open_positions)
        capital_base = self.settings.funded_starting_capital_usd if spec.execution_mode == EXECUTION_MODE_FUNDED_PAPER else self.settings.shadow_starting_capital_usd
        total_pnl = float(snapshot['total_pnl']) + unrealized
        return {
            'signals': int(summary['total_signals']),
            'fills': int(summary['filled_signals']),
            'fill_rate': float(summary['fill_rate']),
            'win_rate': float(summary['win_rate']),
            'realized_pnl': float(summary['realized_pnl']),
            'unrealized_pnl': float(unrealized),
            'open_positions': int(snapshot['open_positions']),
            'total_pnl': total_pnl,
            'return_rate': total_pnl / max(1.0, float(capital_base)),
        }

    async def _unrealized_total(self, open_positions: list) -> float:
        if not open_positions:
            return 0.0
        latest_prices = await self.store.latest_market_prices([str(row.market_id) for row in open_positions])
        total = 0.0
        for row in open_positions:
            side = str(row.side).upper()
            market_id = str(row.market_id)
            yes_price = latest_prices.get(market_id, float(row.entry_price))
            mark_price = (1.0 - float(yes_price)) if side == 'NO' else float(yes_price)
            mtm = Store.estimate_position_unrealized(
                size_usd=float(row.size_usd),
                entry_price=float(row.entry_price),
                mark_price=max(0.001, min(0.999, mark_price)),
                entry_fee_usd=float(getattr(row, 'entry_fee_usd', 0.0) or 0.0),
                exit_fee_bps=float(self.settings.taker_fee_bps),
            )
            total += float(mtm['net_unrealized_pnl_usd'])
        return total

    @staticmethod
    def _mode_label(execution_mode: str) -> str:
        if execution_mode == EXECUTION_MODE_FUNDED_PAPER:
            return 'funded'
        return 'shadow'
