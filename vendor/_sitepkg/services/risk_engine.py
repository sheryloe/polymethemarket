from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from polymethemoney.config import Settings
from polymethemoney.domain import Decision, DecisionType, RiskState, Signal, TradingMode
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
        while True:
            await self.refresh_state()
            await asyncio.sleep(60)

    async def refresh_state(self) -> RiskState:
        now = datetime.now(timezone.utc)
        if self.settings.demo_paper_hardlock and self.runtime_state.trading_mode != TradingMode.PAPER:
            self.runtime_state.trading_mode = TradingMode.PAPER
            self.runtime_state.manual_live_approved = False
        demo_unlimited = self._is_demo_unlimited()
        if not demo_unlimited:
            await self._enforce_position_exit_rules(now)
        day_pnl = await self.store.realized_pnl_window(now - timedelta(days=1))
        week_pnl = await self.store.realized_pnl_window(now - timedelta(days=7))
        daily_drawdown = max(0.0, -day_pnl / self.settings.starting_capital_usd)
        weekly_drawdown = max(0.0, -week_pnl / self.settings.starting_capital_usd)
        daily_limit_enabled = self.settings.daily_loss_limit_pct > 0
        weekly_limit_enabled = self.settings.weekly_loss_limit_pct > 0
        self.state.daily_drawdown_pct = daily_drawdown
        self.state.weekly_drawdown_pct = weekly_drawdown
        self.state.trading_mode = self.runtime_state.trading_mode
        self.state.kill_switch = self.runtime_state.kill_switch
        self.state.paused = self.runtime_state.paused
        if (
            not demo_unlimited
            and (
                (daily_limit_enabled and daily_drawdown >= self.settings.daily_loss_limit_pct)
                or (weekly_limit_enabled and weekly_drawdown >= self.settings.weekly_loss_limit_pct)
            )
        ):
            await self.activate_kill_switch(
                f"\uc190\uc2e4 \ud55c\ub3c4 \ub3c4\ub2ec (\uc77c\uac04 {daily_drawdown:.2%}, \uc8fc\uac04 {weekly_drawdown:.2%})"
            )
        if demo_unlimited and self.runtime_state.kill_switch:
            self.runtime_state.kill_switch = False
            self.state.kill_switch = False
            self.runtime_state.paused = False
            self.state.paused = False
        await self.store.record_equity_curve(self.state, await self.estimated_equity())
        return self.state

    async def _enforce_position_exit_rules(self, now: datetime) -> None:
        stop_loss_threshold = max(0.0, float(self.settings.position_stop_loss_pct))
        take_profit_threshold = max(0.0, float(self.settings.position_take_profit_pct))
        if stop_loss_threshold <= 0 and take_profit_threshold <= 0:
            return
        if self.runtime_state.trading_mode != TradingMode.PAPER:
            return
        required_methods = (
            "get_open_positions",
            "latest_market_prices",
            "close_position_at_mark",
            "add_fill",
        )
        if any(not hasattr(self.store, method) for method in required_methods):
            return
        open_positions = await self.store.get_open_positions(TradingMode.PAPER)
        if not open_positions:
            return
        market_ids = [str(row.market_id) for row in open_positions]
        latest_prices = await self.store.latest_market_prices(market_ids)
        for row in open_positions:
            market_id = str(row.market_id)
            side = str(row.side).upper()
            yes_price = latest_prices.get(market_id)
            if yes_price is None:
                continue
            mark_price = (1.0 - yes_price) if side == "NO" else yes_price
            mark_price = max(0.001, min(0.999, float(mark_price)))
            shares = float(row.size_usd) / max(float(row.entry_price), 0.001)
            unrealized_pnl = (shares * mark_price) - float(row.size_usd)
            size_usd = max(float(row.size_usd), 0.001)
            pnl_pct = unrealized_pnl / size_usd
            hit_stop_loss = stop_loss_threshold > 0 and (-pnl_pct) >= stop_loss_threshold
            hit_take_profit = take_profit_threshold > 0 and pnl_pct >= take_profit_threshold
            if not hit_stop_loss and not hit_take_profit:
                continue
            closed = await self.store.close_position_at_mark(
                position_id=int(row.id),
                mark_price=mark_price,
                mode=TradingMode.PAPER,
            )
            if closed is None:
                continue
            rule = "tp" if hit_take_profit else "sl"
            order_id = f"{rule}-{closed['position_id']}-{int(now.timestamp())}"
            realized = float(closed["realized_pnl_usd"])
            await self.store.add_fill(
                order_id=order_id,
                market_id=str(closed["market_id"]),
                side=side,
                fill_price=float(closed["exit_price"]),
                size_usd=float(closed["size_usd"]),
                fee_usd=0.0,
                pnl_usd=realized,
                trading_mode=TradingMode.PAPER,
            )
            await self.gatekeeper.register_paper_trade(realized, when=now)
            reason = "TP" if hit_take_profit else "SL"
            await self.alert_fn(
                f"[RISK]\n{reason} exit triggered\n"
                f"position {closed['position_id']} | market {closed['market_id']}\n"
                f"entry {closed['entry_price']:.4f} -> exit {closed['exit_price']:.4f}\n"
                f"pnl {realized:+.2f} USD ({pnl_pct:.1%})"
            )

    async def estimated_equity(self) -> float:
        now = datetime.now(timezone.utc)
        total_pnl = await self.store.realized_pnl_window(now - timedelta(days=3650))
        return self.settings.starting_capital_usd + total_pnl

    async def activate_kill_switch(self, reason: str) -> None:
        if self.runtime_state.kill_switch:
            return
        self.runtime_state.kill_switch = True
        self.runtime_state.paused = True
        self.state.kill_switch = True
        self.state.paused = True
        await self.gatekeeper.register_violation()
        await self.store.add_risk_event("kill_switch", reason)
        await self.alert_fn(f"[\ub9ac\uc2a4\ud06c \uacbd\ubcf4]\n\ud0ac\uc2a4\uc704\uce58 \ubc1c\ub3d9\n\uc0ac\uc720: {reason}")
        logger.warning("Kill switch activated: %s", reason)

    async def decide(self, signal: Signal, open_positions: int) -> Decision:
        await self.refresh_state()
        if self._is_demo_unlimited():
            size_usd = max(self.settings.min_position_usd, self.settings.max_position_usd)
            return Decision(
                kind=DecisionType.AUTO,
                reason="demo_unlimited",
                size_usd=size_usd,
            )
        if self.runtime_state.kill_switch:
            return Decision(kind=DecisionType.REJECT, reason="kill_switch")
        if self.runtime_state.paused:
            return Decision(kind=DecisionType.REJECT, reason="paused")
        if open_positions >= self.settings.max_positions:
            return Decision(kind=DecisionType.REJECT, reason="max_positions_reached")
        if signal.net_ev <= 0:
            return Decision(kind=DecisionType.REJECT, reason="negative_net_ev")
        size_usd = self._position_size_usd(signal)
        if size_usd <= 0:
            return Decision(kind=DecisionType.REJECT, reason="size_below_min")
        if signal.score >= self.settings.auto_threshold:
            return Decision(
                kind=DecisionType.AUTO,
                reason="auto_threshold",
                size_usd=size_usd,
            )
        if signal.score >= self.settings.semi_threshold:
            return Decision(
                kind=DecisionType.SEMI,
                reason="semi_threshold",
                size_usd=size_usd,
            )
        return Decision(kind=DecisionType.REJECT, reason="score_below_threshold")

    def _position_size_usd(self, signal: Signal) -> float:
        if self.settings.fixed_position_usd > 0:
            fixed = float(self.settings.fixed_position_usd)
            return max(0.0, min(self.settings.max_position_usd, fixed))

        min_position_usd = (
            self.runtime_state.min_position_usd_override
            if self.runtime_state.min_position_usd_override is not None
            else self.settings.min_position_usd
        )
        implied = max(0.001, min(0.999, signal.implied_prob))
        fair = max(0.001, min(0.999, signal.fair_prob))
        odds = (1.0 - implied) / implied
        full_kelly = ((odds * fair) - (1.0 - fair)) / max(odds, 1e-6)
        scaled_kelly = max(0.0, full_kelly) * self.settings.kelly_fraction
        confidence_scale = 0.35 + 0.65 * max(0.0, min(1.0, signal.model_confidence))
        bankroll_fraction = scaled_kelly * confidence_scale
        size = self.settings.starting_capital_usd * bankroll_fraction
        if size < min_position_usd:
            return 0.0
        return max(min_position_usd, min(self.settings.max_position_usd, size))

    def _is_demo_unlimited(self) -> bool:
        return self.settings.demo_unlimited and self.runtime_state.trading_mode == TradingMode.PAPER
