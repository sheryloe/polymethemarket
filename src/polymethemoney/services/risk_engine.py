from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from polymethemoney.config import Settings
from polymethemoney.domain import (
    Decision,
    DecisionType,
    EXECUTION_MODE_FUNDED_PAPER,
    EXECUTION_MODE_SHADOW_PAPER,
    RiskState,
    Signal,
    TradingMode,
    get_strategy_spec,
    iter_active_strategy_specs,
)
from polymethemoney.services.gatekeeper import Gatekeeper
from polymethemoney.state import RuntimeState
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)

AlertFn = Callable[[str], Awaitable[None]]


class RiskEngine:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        runtime_state: RuntimeState,
        gatekeeper: Gatekeeper,
        alert_fn: AlertFn,
    ) -> None:
        self.settings = settings
        self.store = store
        self.runtime_state = runtime_state
        self.gatekeeper = gatekeeper
        self.alert_fn = alert_fn
        self.state = RiskState()

    async def run(self) -> None:
        interval = max(1, int(self.settings.expiry_close_poll_seconds))
        while True:
            await self.refresh_state()
            await asyncio.sleep(interval)

    async def refresh_state(self) -> RiskState:
        now = datetime.now(timezone.utc)
        if self.settings.demo_paper_hardlock and self.runtime_state.trading_mode != TradingMode.PAPER:
            self.runtime_state.trading_mode = TradingMode.PAPER
            self.runtime_state.manual_live_approved = False

        if self.settings.contrarian_enabled or self.settings.ab_test_enabled:
            await self._enforce_expiry_rules(now)
        elif not self._is_demo_unlimited():
            await self._enforce_position_exit_rules(now)

        funded_strategy_ids = [spec.strategy_id for spec in iter_active_strategy_specs() if spec.execution_mode == EXECUTION_MODE_FUNDED_PAPER]
        day_pnl = 0.0
        week_pnl = 0.0
        for strategy_id in funded_strategy_ids:
            day_pnl += await self.store.realized_pnl_window(now - timedelta(days=1), strategy_id=strategy_id)
            week_pnl += await self.store.realized_pnl_window(now - timedelta(days=7), strategy_id=strategy_id)

        capital_base = max(1.0, float(self.settings.funded_starting_capital_usd))
        self.state.daily_drawdown_pct = max(0.0, -day_pnl / capital_base)
        self.state.weekly_drawdown_pct = max(0.0, -week_pnl / capital_base)
        self.state.trading_mode = self.runtime_state.trading_mode
        self.state.paused = self.runtime_state.paused

        for spec in iter_active_strategy_specs():
            equity = await self.estimated_equity(strategy_id=spec.strategy_id)
            await self.store.record_equity_curve(
                self.state,
                equity,
                strategy_id=spec.strategy_id,
                execution_mode=spec.execution_mode,
            )
        return self.state

    async def decide(self, signal: Signal, open_positions: int) -> Decision:
        await self.refresh_state()
        if self.runtime_state.paused:
            return Decision(kind=DecisionType.REJECT, reason="paused")

        spec = get_strategy_spec(signal.strategy_id)
        strategy_open = await self.store.open_position_exposure(
            self.runtime_state.trading_mode,
            strategy_id=signal.strategy_id,
        )
        if int(strategy_open["count"]) >= max(1, int(spec.max_positions)):
            return Decision(kind=DecisionType.REJECT, reason="strategy_max_positions")

        if spec.execution_mode == EXECUTION_MODE_SHADOW_PAPER:
            return Decision(
                kind=DecisionType.AUTO,
                reason="shadow_strategy",
                size_usd=float(self.settings.shadow_position_usd),
            )

        if signal.net_ev <= 0:
            return Decision(kind=DecisionType.REJECT, reason="negative_net_ev")
        if abs(float(signal.fair_prob) - float(signal.implied_prob)) < float(self.settings.funded_min_prediction_gap):
            return Decision(kind=DecisionType.REJECT, reason="prediction_gap_too_small")
        if float(signal.model_confidence) < float(self.settings.funded_min_model_confidence):
            return Decision(kind=DecisionType.REJECT, reason="low_confidence")
        if float(getattr(signal, "effective_edge", signal.edge)) <= 0:
            return Decision(kind=DecisionType.REJECT, reason="non_positive_edge")

        funded_total = await self.store.open_position_exposure(
            self.runtime_state.trading_mode,
            execution_mode=EXECUTION_MODE_FUNDED_PAPER,
        )
        if int(funded_total["count"]) >= int(self.settings.funded_max_open_positions):
            return Decision(kind=DecisionType.REJECT, reason="funded_max_open_positions")

        size_usd = float(self.settings.funded_position_usd)
        if float(funded_total["notional_usd"]) + size_usd > float(self.settings.funded_max_open_notional_usd) + 1e-9:
            return Decision(kind=DecisionType.REJECT, reason="funded_max_open_notional")

        funded_market_side = await self.store.open_position_exposure(
            self.runtime_state.trading_mode,
            execution_mode=EXECUTION_MODE_FUNDED_PAPER,
            market_id=signal.market_id,
            side=signal.side.value,
        )
        if float(funded_market_side["notional_usd"]) + size_usd > float(self.settings.funded_same_market_side_max_notional_usd) + 1e-9:
            return Decision(kind=DecisionType.REJECT, reason="funded_same_market_side_cap")

        slot_start = datetime.now(timezone.utc).replace(second=0, microsecond=0)
        slot_notional = await self.store.fill_notional_since(
            slot_start,
            trading_mode=self.runtime_state.trading_mode,
            execution_mode=EXECUTION_MODE_FUNDED_PAPER,
        )
        if slot_notional + size_usd > float(self.settings.funded_slot_new_notional_usd) + 1e-9:
            return Decision(kind=DecisionType.REJECT, reason="funded_slot_notional_cap")

        return Decision(kind=DecisionType.AUTO, reason="funded_strategy", size_usd=size_usd)

    async def estimated_equity(self, strategy_id: str | None = None) -> float:
        now = datetime.now(timezone.utc)
        if strategy_id is None:
            total_pnl = await self.store.realized_pnl_window(now - timedelta(days=3650))
            return float(self.settings.starting_capital_usd) + total_pnl
        spec = get_strategy_spec(strategy_id)
        total_pnl = await self.store.realized_pnl_window(now - timedelta(days=3650), strategy_id=strategy_id)
        capital_base = self._portfolio_capital_for_mode(spec.execution_mode)
        return capital_base + total_pnl

    def _portfolio_capital_for_mode(self, execution_mode: str) -> float:
        if execution_mode == EXECUTION_MODE_FUNDED_PAPER:
            return float(self.settings.funded_starting_capital_usd)
        if execution_mode == EXECUTION_MODE_SHADOW_PAPER:
            return float(self.settings.shadow_starting_capital_usd)
        return float(self.settings.starting_capital_usd)

    def _is_demo_unlimited(self) -> bool:
        return self.settings.demo_unlimited and self.runtime_state.trading_mode == TradingMode.PAPER

    async def _enforce_position_exit_rules(self, now: datetime) -> None:
        stop_loss_threshold = max(0.0, float(self.settings.position_stop_loss_pct))
        take_profit_threshold = max(0.0, float(self.settings.position_take_profit_pct))
        if stop_loss_threshold <= 0 and take_profit_threshold <= 0:
            return
        if self.runtime_state.trading_mode != TradingMode.PAPER:
            return

        open_positions = await self.store.get_open_positions(TradingMode.PAPER)
        if not open_positions:
            return

        latest_prices = await self.store.latest_market_prices([str(row.market_id) for row in open_positions])
        for row in open_positions:
            side = str(row.side).upper()
            market_id = str(row.market_id)
            yes_price = latest_prices.get(market_id)
            if yes_price is None:
                continue

            mark_price = (1.0 - float(yes_price)) if side == "NO" else float(yes_price)
            mark_price = max(0.001, min(0.999, mark_price))
            mtm = Store.estimate_position_unrealized(
                size_usd=float(row.size_usd),
                entry_price=float(row.entry_price),
                mark_price=mark_price,
                entry_fee_usd=float(getattr(row, "entry_fee_usd", 0.0) or 0.0),
                exit_fee_bps=float(self.settings.taker_fee_bps),
            )
            unrealized_pnl = float(mtm["net_unrealized_pnl_usd"])
            pnl_pct = unrealized_pnl / max(float(row.size_usd), 0.001)
            hit_stop_loss = stop_loss_threshold > 0 and (-pnl_pct) >= stop_loss_threshold
            hit_take_profit = take_profit_threshold > 0 and pnl_pct >= take_profit_threshold
            if not hit_stop_loss and not hit_take_profit:
                continue
            closed = await self.store.close_position_at_mark(
                position_id=int(row.id),
                mark_price=mark_price,
                mode=TradingMode.PAPER,
                exit_fee_bps=float(self.settings.taker_fee_bps),
                strategy_id=str(getattr(row, "strategy_id", "")) or None,
                close_reason="stop_loss" if hit_stop_loss else "take_profit",
            )
            if closed is None:
                continue
            realized = float(closed["realized_pnl_usd"])
            await self.store.add_fill(
                order_id=f"{'tp' if hit_take_profit else 'sl'}-{closed['position_id']}-{int(now.timestamp())}",
                market_id=str(closed["market_id"]),
                side=side,
                fill_price=float(closed["exit_price"]),
                size_usd=float(closed["size_usd"]),
                fee_usd=float(closed.get("exit_fee_usd", 0.0) or 0.0),
                pnl_usd=realized,
                trading_mode=TradingMode.PAPER,
                strategy_id=str(closed.get("strategy_id") or ""),
                execution_mode=str(closed.get("execution_mode") or "legacy"),
            )
            if realized != 0.0:
                await self.gatekeeper.register_paper_trade(realized, when=now)

    async def _enforce_expiry_rules(self, now: datetime) -> None:
        if self.runtime_state.trading_mode != TradingMode.PAPER:
            return

        open_positions = await self.store.get_open_positions(TradingMode.PAPER)
        if not open_positions:
            return

        market_ids = [str(row.market_id) for row in open_positions]
        expiries = await self.store.latest_market_expiries(market_ids)
        grace = timedelta(seconds=max(0, int(self.settings.contrarian_expiry_grace_seconds)))
        timeout = timedelta(seconds=max(5, int(self.settings.outcome_resolution_timeout_seconds)))
        expired_rows = [
            row
            for row in open_positions
            if expiries.get(str(row.market_id)) is not None and now >= expiries[str(row.market_id)] + grace
        ]
        if not expired_rows:
            return

        outcomes = await self.store.latest_market_outcomes([str(row.market_id) for row in expired_rows])
        fallback_rows = []
        for row in expired_rows:
            market_id = str(row.market_id)
            outcome = outcomes.get(market_id)
            outcome_yes = None if outcome is None else outcome.get("outcome_yes")
            if outcome_yes in {0.0, 1.0}:
                closed = await self.store.close_position_at_settlement(
                    position_id=int(row.id),
                    outcome_yes=float(outcome_yes),
                    mode=TradingMode.PAPER,
                    strategy_id=str(getattr(row, "strategy_id", "")) or None,
                )
                if closed is None:
                    continue
                realized = float(closed["realized_pnl_usd"])
                await self.store.add_fill(
                    order_id=f"expiry-{int(row.id)}-{int(now.timestamp())}",
                    market_id=market_id,
                    side=str(closed["side"]),
                    fill_price=float(closed["exit_price"]),
                    size_usd=float(closed["size_usd"]),
                    fee_usd=0.0,
                    pnl_usd=realized,
                    trading_mode=TradingMode.PAPER,
                    strategy_id=str(closed.get("strategy_id") or ""),
                    execution_mode=str(closed.get("execution_mode") or "legacy"),
                )
                if realized != 0.0:
                    await self.gatekeeper.register_paper_trade(realized, when=now)
                continue

            expiry = expiries.get(market_id)
            if expiry is not None and now >= expiry + timeout:
                fallback_rows.append(row)

        if not fallback_rows:
            return

        latest_prices = await self.store.latest_market_prices([str(row.market_id) for row in fallback_rows])
        for row in fallback_rows:
            market_id = str(row.market_id)
            yes_price = latest_prices.get(market_id)
            if yes_price is None:
                continue
            side = str(row.side).upper()
            mark_price = (1.0 - float(yes_price)) if side == "NO" else float(yes_price)
            mark_price = max(0.001, min(0.999, mark_price))
            closed = await self.store.close_position_at_mark(
                position_id=int(row.id),
                mark_price=mark_price,
                mode=TradingMode.PAPER,
                exit_fee_bps=float(self.settings.taker_fee_bps),
                strategy_id=str(getattr(row, "strategy_id", "")) or None,
                close_reason="fallback_mark_close",
                fallback_mark_close=True,
            )
            if closed is None:
                continue
            realized = float(closed["realized_pnl_usd"])
            await self.store.add_fill(
                order_id=f"fallback-{int(row.id)}-{int(now.timestamp())}",
                market_id=market_id,
                side=side,
                fill_price=float(closed["exit_price"]),
                size_usd=float(closed["size_usd"]),
                fee_usd=float(closed.get("exit_fee_usd", 0.0) or 0.0),
                pnl_usd=realized,
                trading_mode=TradingMode.PAPER,
                strategy_id=str(closed.get("strategy_id") or ""),
                execution_mode=str(closed.get("execution_mode") or "legacy"),
            )
            if realized != 0.0:
                await self.gatekeeper.register_paper_trade(realized, when=now)
