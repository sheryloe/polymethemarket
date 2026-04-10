from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from polymethemoney.config import Settings
from polymethemoney.domain import AB_STRATEGY_IDS, Decision, DecisionType, RiskState, Signal, TradingMode
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

        if self.settings.contrarian_enabled or self.settings.ab_test_enabled:
            await self._enforce_expiry_rules(now)
        elif not self._is_demo_unlimited():
            await self._enforce_position_exit_rules(now)

        day_pnl = await self.store.realized_pnl_window(now - timedelta(days=1))
        week_pnl = await self.store.realized_pnl_window(now - timedelta(days=7))
        capital_base = self._portfolio_capital() * (len(AB_STRATEGY_IDS) if self.settings.ab_test_enabled else 1)

        self.state.daily_drawdown_pct = max(0.0, -day_pnl / max(1.0, capital_base))
        self.state.weekly_drawdown_pct = max(0.0, -week_pnl / max(1.0, capital_base))
        self.state.trading_mode = self.runtime_state.trading_mode
        self.state.paused = self.runtime_state.paused

        if self.settings.ab_test_enabled:
            for strategy_id in AB_STRATEGY_IDS:
                await self.store.record_equity_curve(
                    self.state,
                    await self.estimated_equity(strategy_id=strategy_id),
                    strategy_id=strategy_id,
                )
        else:
            await self.store.record_equity_curve(self.state, await self.estimated_equity())
        return self.state

    async def decide(self, signal: Signal, open_positions: int) -> Decision:
        await self.refresh_state()

        if self.runtime_state.paused:
            return Decision(kind=DecisionType.REJECT, reason="paused")

        if self.settings.ab_test_enabled:
            max_positions = max(1, int(self.settings.model_max_positions))
            if open_positions >= max_positions:
                return Decision(kind=DecisionType.REJECT, reason="max_positions_reached")
            size_usd = max(0.0, min(float(self.settings.max_position_usd), float(self.settings.model_position_usd)))
            if size_usd <= 0:
                return Decision(kind=DecisionType.REJECT, reason="size_below_min")
            return Decision(kind=DecisionType.AUTO, reason=signal.strategy_id, size_usd=size_usd)

        if self.settings.contrarian_enabled:
            max_positions = max(1, int(self.settings.contrarian_max_positions))
            if open_positions >= max_positions:
                return Decision(kind=DecisionType.REJECT, reason="max_positions_reached")
            size_usd = max(
                0.0,
                min(float(self.settings.max_position_usd), float(self.settings.contrarian_position_usd)),
            )
            if size_usd <= 0:
                return Decision(kind=DecisionType.REJECT, reason="size_below_min")
            return Decision(kind=DecisionType.AUTO, reason="contrarian", size_usd=size_usd)

        if self._is_demo_unlimited():
            size_usd = max(self.settings.min_position_usd, self.settings.max_position_usd)
            return Decision(kind=DecisionType.AUTO, reason="demo_unlimited", size_usd=size_usd)

        if open_positions >= self.settings.max_positions:
            return Decision(kind=DecisionType.REJECT, reason="max_positions_reached")
        if signal.net_ev <= 0:
            return Decision(kind=DecisionType.REJECT, reason="negative_net_ev")

        size_usd = self._position_size_usd(signal)
        if size_usd <= 0:
            return Decision(kind=DecisionType.REJECT, reason="size_below_min")
        if signal.score >= self.settings.auto_threshold:
            return Decision(kind=DecisionType.AUTO, reason="auto_threshold", size_usd=size_usd)
        if signal.score >= self.settings.semi_threshold:
            return Decision(kind=DecisionType.SEMI, reason="semi_threshold", size_usd=size_usd)
        return Decision(kind=DecisionType.REJECT, reason="score_below_threshold")

    async def estimated_equity(self, strategy_id: str | None = None) -> float:
        now = datetime.now(timezone.utc)
        total_pnl = await self.store.realized_pnl_window(now - timedelta(days=3650), strategy_id=strategy_id)
        return self._portfolio_capital() + total_pnl

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
        confidence_scale = 0.35 + (0.65 * max(0.0, min(1.0, signal.model_confidence)))
        bankroll_fraction = scaled_kelly * confidence_scale
        size = self.settings.starting_capital_usd * bankroll_fraction
        if size < min_position_usd:
            return 0.0
        return max(min_position_usd, min(self.settings.max_position_usd, size))

    def _portfolio_capital(self) -> float:
        if self.settings.ab_test_enabled:
            return float(self.settings.model_portfolio_starting_capital_usd)
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
            shares = float(row.size_usd) / max(float(row.entry_price), 0.001)
            unrealized_pnl = (shares * mark_price) - float(row.size_usd)
            pnl_pct = unrealized_pnl / max(float(row.size_usd), 0.001)

            hit_stop_loss = stop_loss_threshold > 0 and (-pnl_pct) >= stop_loss_threshold
            hit_take_profit = take_profit_threshold > 0 and pnl_pct >= take_profit_threshold
            if not hit_stop_loss and not hit_take_profit:
                continue

            closed = await self.store.close_position_at_mark(
                position_id=int(row.id),
                mark_price=mark_price,
                mode=TradingMode.PAPER,
                strategy_id=str(getattr(row, "strategy_id", "")) or None,
            )
            if closed is None:
                continue

            realized = float(closed["realized_pnl_usd"])
            order_id = f"{'tp' if hit_take_profit else 'sl'}-{closed['position_id']}-{int(now.timestamp())}"
            await self.store.add_fill(
                order_id=order_id,
                market_id=str(closed["market_id"]),
                side=side,
                fill_price=float(closed["exit_price"]),
                size_usd=float(closed["size_usd"]),
                fee_usd=0.0,
                pnl_usd=realized,
                trading_mode=TradingMode.PAPER,
                strategy_id=str(closed.get("strategy_id") or ""),
            )
            if realized != 0.0:
                await self.gatekeeper.register_paper_trade(realized, when=now)
            await self.alert_fn(
                "[리스크 청산]\n"
                f"시장 {closed['market_id']} | 방향 {side}\n"
                f"진입 {closed['entry_price']:.4f} -> 청산 {closed['exit_price']:.4f}\n"
                f"실현 손익 {realized:+.2f} USD"
            )

    async def _enforce_expiry_rules(self, now: datetime) -> None:
        if self.runtime_state.trading_mode != TradingMode.PAPER:
            return

        open_positions = await self.store.get_open_positions(TradingMode.PAPER)
        if not open_positions:
            return

        expiries = await self.store.latest_market_expiries([str(row.market_id) for row in open_positions])
        grace = timedelta(seconds=max(0, int(self.settings.contrarian_expiry_grace_seconds)))
        expired_rows = [
            row
            for row in open_positions
            if expiries.get(str(row.market_id)) is not None and now >= expiries[str(row.market_id)] + grace
        ]
        if not expired_rows:
            return

        latest_prices = await self.store.latest_market_prices([str(row.market_id) for row in expired_rows])
        closed_count = 0
        realized_total = 0.0
        strategy_counts: dict[str, int] = {}

        for row in expired_rows:
            market_id = str(row.market_id)
            side = str(row.side).upper()
            yes_price = latest_prices.get(market_id)
            if yes_price is None:
                continue

            mark_price = (1.0 - float(yes_price)) if side == "NO" else float(yes_price)
            mark_price = max(0.001, min(0.999, mark_price))
            strategy_id = str(getattr(row, "strategy_id", "")) or None
            closed = await self.store.close_position_at_mark(
                position_id=int(row.id),
                mark_price=mark_price,
                mode=TradingMode.PAPER,
                strategy_id=strategy_id,
            )
            if closed is None:
                continue

            realized = float(closed["realized_pnl_usd"])
            await self.store.add_fill(
                order_id=f"expiry-{int(row.id)}-{int(now.timestamp())}",
                market_id=market_id,
                side=side,
                fill_price=float(closed["exit_price"]),
                size_usd=float(closed["size_usd"]),
                fee_usd=0.0,
                pnl_usd=realized,
                trading_mode=TradingMode.PAPER,
                strategy_id=str(closed.get("strategy_id") or ""),
            )
            if realized != 0.0:
                await self.gatekeeper.register_paper_trade(realized, when=now)

            closed_count += 1
            realized_total += realized
            label = str(closed.get("strategy_id") or "legacy")
            strategy_counts[label] = strategy_counts.get(label, 0) + 1

        if closed_count <= 0:
            return

        if self.settings.ab_test_enabled:
            parts = []
            for strategy_id in AB_STRATEGY_IDS:
                equity = await self.estimated_equity(strategy_id=strategy_id)
                base = max(1.0, self._portfolio_capital())
                ret = (equity - base) / base
                count = strategy_counts.get(strategy_id, 0)
                label = "모델 A" if strategy_id == AB_STRATEGY_IDS[0] else "모델 B"
                parts.append(f"{label} {count}건 | 누적 {equity - base:+.2f} USD ({ret:+.2%})")
            await self.alert_fn(
                "[만기 청산]\n"
                f"총 {closed_count}건 청산 | 실현 {realized_total:+.2f} USD\n"
                + "\n".join(parts)
            )
            return

        equity = await self.estimated_equity()
        base = max(1.0, self._portfolio_capital())
        ret = (equity - base) / base
        await self.alert_fn(
            "[만기 청산]\n"
            f"총 {closed_count}건 청산 | 실현 {realized_total:+.2f} USD\n"
            f"누적 손익 {equity - base:+.2f} USD | 누적 수익률 {ret:+.2%}"
        )
