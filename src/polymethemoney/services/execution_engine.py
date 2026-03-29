from __future__ import annotations

import logging
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

from polymethemoney.adapters.paper_exchange import PaperExchange
from polymethemoney.adapters.polymarket_client import PolymarketClient
from polymethemoney.config import Settings
from polymethemoney.domain import (
    DecisionType,
    ExecutionBundleIntent,
    ExecutionBundleResult,
    FillResult,
    OrderIntent,
    Side,
    Signal,
    TradingMode,
)
from polymethemoney.services.gatekeeper import Gatekeeper
from polymethemoney.services.risk_engine import RiskEngine
from polymethemoney.state import RuntimeState
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)

NotifyFn = Callable[[str], Awaitable[None]]


class ExecutionEngine:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        runtime_state: RuntimeState,
        risk_engine: RiskEngine,
        gatekeeper: Gatekeeper,
        paper_exchange: PaperExchange,
        polymarket_client: PolymarketClient,
        notify_fn: NotifyFn,
    ) -> None:
        self.settings = settings
        self.store = store
        self.runtime_state = runtime_state
        self.risk_engine = risk_engine
        self.gatekeeper = gatekeeper
        self.paper_exchange = paper_exchange
        self.polymarket_client = polymarket_client
        self.notify_fn = notify_fn

    async def on_signal(self, signal: Signal) -> None:
        await self.store.add_signal(signal)
        self.runtime_state.recent_signals.append(signal)
        open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
        max_per_side = max(1, self.settings.max_open_per_market_side)
        if self._open_same_market_side_count(open_positions, signal) >= max_per_side:
            rotated = await self._rotate_market_side_if_needed(signal, open_positions, max_per_side)
            if rotated:
                open_positions = await self.store.get_open_positions(self.runtime_state.trading_mode)
            if self._open_same_market_side_count(open_positions, signal) >= max_per_side:
                await self.store.update_signal_status(signal.signal_id, "rejected:market_side_max_open")
                return
        structure_open = await self.store.structure_open_legs_count()
        decision = await self.risk_engine.decide(signal, len(open_positions) + structure_open)
        if decision.kind == DecisionType.REJECT:
            await self.store.update_signal_status(signal.signal_id, f"rejected:{decision.reason}")
            return
        intent = OrderIntent(
            signal_id=signal.signal_id,
            market_id=signal.market_id,
            side=signal.side,
            price=self._entry_limit_price(signal),
            size_usd=decision.size_usd,
            mode=decision.kind,
        )
        if decision.kind == DecisionType.SEMI:
            self.runtime_state.pending_approvals[signal.signal_id] = intent
            await self.store.update_signal_status(signal.signal_id, "pending_approval")
            await self.notify_fn(
                "[\uc2b9\uc778 \ub300\uae30 \uc2e0\ud638]\n"
                f"\uc2dc\uadf8\ub110 ID {signal.signal_id}\n"
                f"\uc2dc\uc7a5 {signal.market_id} | \ubc29\ud5a5 {signal.side.value}\n"
                f"\uc810\uc218 {signal.score} | \uc5e3\uc9c0 {signal.edge:.4f}\n"
                f"/approve {signal.signal_id} \ub610\ub294 /reject {signal.signal_id}"
            )
            return
        await self._execute_intent(intent)

    async def approve_signal(self, signal_id: str) -> str:
        intent = self.runtime_state.pending_approvals.pop(signal_id, None)
        if intent is None:
            return f"[\uc2b9\uc778 \uc2e4\ud328]\n\ud574\ub2f9 \uc2dc\uadf8\ub110\uc744 \ucc3e\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4: {signal_id}"
        await self._execute_intent(intent)
        return f"[\uc2b9\uc778 \uc644\ub8cc]\n\uc2dc\uadf8\ub110 {signal_id} \uc8fc\ubb38 \uc2e4\ud589\uc744 \uc2dc\ub3c4\ud588\uc2b5\ub2c8\ub2e4."

    async def reject_signal(self, signal_id: str) -> str:
        intent = self.runtime_state.pending_approvals.pop(signal_id, None)
        if intent is None:
            return f"[\uac70\uc808 \uc2e4\ud328]\n\ud574\ub2f9 \uc2dc\uadf8\ub110\uc744 \ucc3e\uc9c0 \ubabb\ud588\uc2b5\ub2c8\ub2e4: {signal_id}"
        await self.store.update_signal_status(signal_id, "rejected:manual")
        return f"[\uac70\uc808 \uc644\ub8cc]\n\uc2dc\uadf8\ub110 {signal_id}\ub97c \uc218\ub3d9 \uac70\uc808\ud588\uc2b5\ub2c8\ub2e4."

    async def close_all_positions(self) -> int:
        mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else self.runtime_state.trading_mode
        closed = await self.store.close_all_open_positions(mode)
        await self.notify_fn(
            "[\uc804\uccb4 \uccad\uc0b0]\n"
            f"\ubaa8\ub4dc {mode.value.upper()} \uc5d0\uc11c {closed}\uac1c \ud3ec\uc9c0\uc158 \uccad\uc0b0 \ucc98\ub9ac"
        )
        return closed

    async def reject_all_pending(self) -> int:
        signal_ids = list(self.runtime_state.pending_approvals.keys())
        self.runtime_state.pending_approvals.clear()
        for signal_id in signal_ids:
            await self.store.update_signal_status(signal_id, "rejected:manual_batch")
        return len(signal_ids)

    async def force_close_position(self, position_id: int) -> dict:
        return await self.force_close_positions([position_id], reason="manual_close_one")

    async def force_close_market(self, market_id: str) -> dict:
        mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else self.runtime_state.trading_mode
        rows = await self.store.get_open_positions(mode)
        target_ids = [int(row.id) for row in rows if str(row.market_id) == str(market_id)]
        return await self.force_close_positions(target_ids, reason=f"manual_close_market:{market_id}")

    async def force_close_side(self, side: str, limit: int | None = None) -> dict:
        mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else self.runtime_state.trading_mode
        side_upper = side.upper()
        rows = await self.store.get_open_positions(mode)
        target_ids = [int(row.id) for row in rows if str(row.side).upper() == side_upper]
        if limit is not None and limit > 0:
            target_ids = target_ids[:limit]
        return await self.force_close_positions(target_ids, reason=f"manual_close_side:{side_upper}")

    async def force_close_positions(self, position_ids: list[int], reason: str) -> dict:
        mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else self.runtime_state.trading_mode
        unique_ids = list(dict.fromkeys(int(pid) for pid in position_ids if int(pid) > 0))
        if not unique_ids:
            return {
                "mode": mode.value,
                "requested": 0,
                "closed": 0,
                "realized_pnl_usd": 0.0,
                "missing": 0,
                "no_price": 0,
                "error": None,
            }

        if mode != TradingMode.PAPER:
            # Live flatten path is intentionally blocked until exchange-close path is implemented.
            return {
                "mode": mode.value,
                "requested": len(unique_ids),
                "closed": 0,
                "realized_pnl_usd": 0.0,
                "missing": len(unique_ids),
                "no_price": 0,
                "error": "live_manual_close_not_supported",
            }

        rows = await self.store.get_open_positions(mode)
        by_id = {int(row.id): row for row in rows}
        target_rows = [by_id[pid] for pid in unique_ids if pid in by_id]
        missing = len(unique_ids) - len(target_rows)
        if not target_rows:
            return {
                "mode": mode.value,
                "requested": len(unique_ids),
                "closed": 0,
                "realized_pnl_usd": 0.0,
                "missing": missing,
                "no_price": 0,
                "error": None,
            }

        market_ids = [str(row.market_id) for row in target_rows]
        latest_prices = await self.store.latest_market_prices(market_ids)

        closed_count = 0
        no_price = 0
        realized_total = 0.0
        now = datetime.now(timezone.utc)
        for row in target_rows:
            market_id = str(row.market_id)
            side = str(row.side).upper()
            yes_price = latest_prices.get(market_id)
            if yes_price is None:
                no_price += 1
                continue
            mark_price = (1.0 - float(yes_price)) if side == "NO" else float(yes_price)
            mark_price = max(0.001, min(0.999, mark_price))
            closed = await self.store.close_position_at_mark(
                position_id=int(row.id),
                mark_price=mark_price,
                mode=mode,
            )
            if closed is None:
                continue
            realized = float(closed["realized_pnl_usd"])
            order_id = f"manual-{int(row.id)}-{int(now.timestamp())}"
            await self.store.add_fill(
                order_id=order_id,
                market_id=market_id,
                side=side,
                fill_price=float(closed["exit_price"]),
                size_usd=float(closed["size_usd"]),
                fee_usd=0.0,
                pnl_usd=realized,
                trading_mode=mode,
            )
            if mode == TradingMode.PAPER and realized != 0.0:
                await self.gatekeeper.register_paper_trade(realized, when=now)
            realized_total += realized
            closed_count += 1

        await self.notify_fn(
            "[수동 청산]\n"
            f"사유 {reason}\n"
            f"요청 {len(unique_ids)} | 청산 {closed_count} | 미존재 {missing} | 가격없음 {no_price}\n"
            f"실현손익 {realized_total:+.2f} USD"
        )
        return {
            "mode": mode.value,
            "requested": len(unique_ids),
            "closed": closed_count,
            "realized_pnl_usd": realized_total,
            "missing": missing,
            "no_price": no_price,
            "error": None,
        }

    async def emergency_stop(self, flatten: bool = True) -> dict:
        self.runtime_state.paused = True
        self.runtime_state.kill_switch = True
        rejected = await self.reject_all_pending()
        flatten_result = {
            "requested": 0,
            "closed": 0,
            "realized_pnl_usd": 0.0,
            "missing": 0,
            "no_price": 0,
            "error": None,
        }
        if flatten:
            mode = TradingMode.PAPER if self.settings.demo_paper_hardlock else self.runtime_state.trading_mode
            rows = await self.store.get_open_positions(mode)
            flatten_result = await self.force_close_positions(
                [int(row.id) for row in rows],
                reason="emergency_stop",
            )
        return {
            "paused": self.runtime_state.paused,
            "kill_switch": self.runtime_state.kill_switch,
            "rejected_pending": rejected,
            "flatten": flatten_result,
        }

    async def execute_bundle(self, intent: ExecutionBundleIntent) -> ExecutionBundleResult:
        detected_at = datetime.now(timezone.utc)
        await self.store.upsert_structure_bundle(
            {
                "opportunity_id": intent.opportunity_id,
                "event_id": intent.event_id,
                "kind": intent.kind,
                "status": "detected",
                "legs_count": len(intent.legs),
                "target_payout_usd": intent.target_payout_usd,
                "gross_edge_usd": intent.gross_edge_usd,
                "net_edge_usd": intent.net_edge_usd,
                "detected_at": detected_at,
                "opened_at": None,
                "closed_at": None,
                "rolled_back": False,
                "error": None,
            }
        )

        mode = self.runtime_state.trading_mode
        if mode != TradingMode.PAPER:
            await self.store.upsert_structure_bundle(
                {
                    "opportunity_id": intent.opportunity_id,
                    "event_id": intent.event_id,
                    "kind": intent.kind,
                    "status": "rejected",
                    "legs_count": len(intent.legs),
                    "target_payout_usd": intent.target_payout_usd,
                    "gross_edge_usd": intent.gross_edge_usd,
                    "net_edge_usd": intent.net_edge_usd,
                    "detected_at": detected_at,
                    "opened_at": None,
                    "closed_at": detected_at,
                    "rolled_back": False,
                    "error": "live_bundle_not_supported_yet",
                }
            )
            return ExecutionBundleResult(
                opportunity_id=intent.opportunity_id,
                status="rejected",
                filled_legs=0,
                rolled_back=False,
                realized_pnl_usd=0.0,
                opened_legs=0,
                error="live_bundle_not_supported_yet",
            )

        filled_rows: list[dict] = []
        try:
            for leg in intent.legs:
                if leg.size_usd <= 0 or leg.size_usd > (self.settings.max_position_usd + 1e-9):
                    raise ValueError("leg_size_out_of_range")
                if leg.size_shares <= 0 or leg.vwap_price <= 0:
                    raise ValueError("invalid_leg_price_or_size")
                slippage = leg.vwap_price * (self.settings.arb_slippage_buffer_bps / 10000.0)
                fill_price = max(0.001, min(0.999, leg.vwap_price + slippage))
                if fill_price >= 0.999:
                    raise ValueError("execution_price_out_of_range")
                filled_rows.append(
                    {
                        "opportunity_id": intent.opportunity_id,
                        "event_id": intent.event_id,
                        "kind": intent.kind,
                        "market_id": leg.market_id,
                        "token_id": leg.token_id,
                        "outcome_idx": leg.outcome_idx,
                        "outcome_name": leg.outcome_name,
                        "entry_price": fill_price,
                        "size_shares": leg.size_shares,
                        "size_usd": leg.size_shares * fill_price,
                        "status": "open",
                        "opened_at": datetime.now(timezone.utc),
                        "closed_at": None,
                        "exit_price": None,
                        "realized_pnl_usd": 0.0,
                    }
                )
        except Exception as exc:
            realized = 0.0
            rolled_back = False
            if filled_rows:
                await self.store.mark_structure_bundle_open(intent.opportunity_id)
                await self.store.add_structure_positions(filled_rows)
                exit_prices = {
                    str(row["token_id"]): max(
                        0.001,
                        min(
                            0.999,
                            float(row["entry_price"])
                            * (1.0 - ((self.settings.arb_fee_buffer_bps + self.settings.arb_slippage_buffer_bps) / 10000.0)),
                        ),
                    )
                    for row in filled_rows
                }
                realized = await self.store.close_structure_bundle(
                    opportunity_id=intent.opportunity_id,
                    exit_prices=exit_prices,
                    status="rolled_back",
                    rolled_back=True,
                    error=str(exc),
                )
                rolled_back = True
                if realized != 0:
                    await self.gatekeeper.register_paper_trade(realized, when=datetime.now(timezone.utc))
            else:
                await self.store.upsert_structure_bundle(
                    {
                        "opportunity_id": intent.opportunity_id,
                        "event_id": intent.event_id,
                        "kind": intent.kind,
                        "status": "failed",
                        "legs_count": len(intent.legs),
                        "target_payout_usd": intent.target_payout_usd,
                        "gross_edge_usd": intent.gross_edge_usd,
                        "net_edge_usd": intent.net_edge_usd,
                        "detected_at": detected_at,
                        "opened_at": None,
                        "closed_at": datetime.now(timezone.utc),
                        "rolled_back": False,
                        "error": str(exc),
                    }
                )
            return ExecutionBundleResult(
                opportunity_id=intent.opportunity_id,
                status="rolled_back" if rolled_back else "failed",
                filled_legs=len(filled_rows),
                rolled_back=rolled_back,
                realized_pnl_usd=realized,
                opened_legs=0,
                error=str(exc),
            )

        await self.store.mark_structure_bundle_open(intent.opportunity_id)
        opened = await self.store.add_structure_positions(filled_rows)
        return ExecutionBundleResult(
            opportunity_id=intent.opportunity_id,
            status="filled",
            filled_legs=opened,
            rolled_back=False,
            realized_pnl_usd=0.0,
            opened_legs=opened,
            error=None,
        )

    async def _execute_intent(self, intent: OrderIntent) -> None:
        mode = self.runtime_state.trading_mode
        if self.settings.demo_paper_hardlock and mode != TradingMode.PAPER:
            self.runtime_state.trading_mode = TradingMode.PAPER
            self.runtime_state.manual_live_approved = False
            await self.store.update_signal_status(intent.signal_id, "rejected:demo_paper_hardlock")
            await self.notify_fn("[데모 모드] 실거래 경로가 차단되어 PAPER로 고정됩니다.")
            return
        if mode == TradingMode.LIVE:
            gate_ok = await self.gatekeeper.is_live_gate_passed()
            if not gate_ok:
                await self.store.update_signal_status(intent.signal_id, "rejected:live_gate_fail")
                await self.notify_fn("[\ub9ac\uc2a4\ud06c]\n\uc2e4\uac70\ub798 \uac8c\uc774\ud2b8 \ubbf8\ud1b5\uacfc\ub85c PAPER \ubaa8\ub4dc \uc720\uc9c0")
                return
            if not self.runtime_state.manual_live_approved:
                await self.store.update_signal_status(intent.signal_id, "rejected:manual_live_not_approved")
                await self.notify_fn("[\ub9ac\uc2a4\ud06c]\n/go_live \uc218\ub3d9 \uc2b9\uc778 \ud6c4\uc5d0\ub9cc \uc2e4\uac70\ub798 \uc9c4\uc785 \uac00\ub2a5")
                return
        result = await self._execute_with_requote(intent, mode)
        if result is None:
            await self.store.update_signal_status(intent.signal_id, "rejected:execution_failed")
            return
        if result.status != "filled":
            await self.store.update_signal_status(intent.signal_id, f"not_filled:{result.status}")
            await self.notify_fn(
                "[\uc8fc\ubb38 \ubbf8\uccb4\uacb0]\n"
                f"\uc2dc\uadf8\ub110 {intent.signal_id}\n"
                f"\uc2dc\uc7a5 {intent.market_id}\n"
                f"\uc0c1\ud0dc {result.status}"
            )
            return
        await self.store.update_signal_status(intent.signal_id, "filled")
        await self._apply_fill_to_positions(intent, result)
        await self.risk_engine.refresh_state()

    async def _execute_with_requote(self, intent: OrderIntent, mode: TradingMode) -> FillResult | None:
        current_intent = intent
        for attempt in range(self.settings.limit_requote_retries + 1):
            try:
                if mode == TradingMode.PAPER:
                    fill = await self.paper_exchange.place_limit_order(current_intent)
                else:
                    fill = await self.polymarket_client.place_limit_order(current_intent, mode)
                await self.store.add_order(fill.order_id, current_intent, mode, fill.status)
            except Exception as exc:
                logger.exception("Order execution failed: %s", exc)
                if attempt == self.settings.limit_requote_retries:
                    return None
                current_intent = self._requote(current_intent)
                continue
            if fill.status == "filled":
                return fill
            if attempt == self.settings.limit_requote_retries:
                return fill
            current_intent = self._requote(current_intent)
        return None

    def _requote(self, intent: OrderIntent) -> OrderIntent:
        step = self.settings.limit_requote_step
        if intent.side == Side.YES:
            price = min(0.999, intent.price + step)
        else:
            price = min(0.999, intent.price + step)
        return replace(intent, price=price)

    async def _apply_fill_to_positions(self, intent: OrderIntent, fill: FillResult) -> None:
        mode = fill.mode
        opposite = Side.NO.value if intent.side == Side.YES else Side.YES.value
        closed = await self.store.close_positions_for_market(
            market_id=intent.market_id,
            opposite_side=opposite,
            exit_price=fill.fill_price,
            mode=mode,
        )
        realized = sum(pnl for _, pnl in closed)
        if realized != 0 and mode == TradingMode.PAPER:
            await self.gatekeeper.register_paper_trade(realized, when=datetime.now(timezone.utc))
        await self.store.add_fill(
            order_id=fill.order_id,
            market_id=fill.market_id,
            side=fill.side.value,
            fill_price=fill.fill_price,
            size_usd=fill.size_usd,
            fee_usd=fill.fee_usd,
            pnl_usd=realized,
            trading_mode=mode,
        )
        await self.store.open_position(
            market_id=intent.market_id,
            side=intent.side.value,
            entry_price=fill.fill_price,
            size_usd=intent.size_usd,
            mode=mode,
        )

    def _entry_limit_price(self, signal: Signal) -> float:
        if signal.side == Side.YES:
            return max(0.001, min(0.999, signal.implied_prob + 0.002))
        no_implied = 1.0 - signal.implied_prob
        return max(0.001, min(0.999, no_implied + 0.002))

    @staticmethod
    def _open_same_market_side_count(open_positions: list, signal: Signal) -> int:
        side = signal.side.value.upper()
        market_id = str(signal.market_id)
        return sum(
            1
            for row in open_positions
            if str(getattr(row, "market_id", "")) == market_id and str(getattr(row, "side", "")).upper() == side
        )

    async def _rotate_market_side_if_needed(
        self,
        signal: Signal,
        open_positions: list,
        max_per_side: int,
    ) -> bool:
        if not self.settings.market_side_rotation_enabled:
            return False
        mode = self.runtime_state.trading_mode
        if mode != TradingMode.PAPER:
            return False
        market_id = str(signal.market_id)
        side = signal.side.value.upper()
        candidates = [
            row
            for row in open_positions
            if str(getattr(row, "market_id", "")) == market_id and str(getattr(row, "side", "")).upper() == side
        ]
        if len(candidates) < max_per_side:
            return False

        now = datetime.now(timezone.utc)
        min_age = timedelta(minutes=max(0, int(self.settings.market_side_rotation_min_age_minutes)))
        eligible = [
            row
            for row in candidates
            if getattr(row, "opened_at", None) is None or (now - row.opened_at) >= min_age
        ]
        if not eligible:
            await self.store.update_signal_status(signal.signal_id, "rejected:market_side_max_open_recent")
            return False

        eligible.sort(key=lambda row: getattr(row, "opened_at", now) or now)
        required_close = (len(candidates) - max_per_side) + 1
        required_close = max(1, required_close)
        to_close = eligible[:required_close]

        latest_prices = await self.store.latest_market_prices([market_id])
        yes_price = latest_prices.get(market_id)
        if yes_price is None:
            return False

        closed_count = 0
        realized_total = 0.0
        for row in to_close:
            mark_price = (1.0 - float(yes_price)) if side == "NO" else float(yes_price)
            mark_price = max(0.001, min(0.999, mark_price))
            closed = await self.store.close_position_at_mark(
                position_id=int(row.id),
                mark_price=mark_price,
                mode=mode,
            )
            if closed is None:
                continue
            realized = float(closed["realized_pnl_usd"])
            order_id = f"rotate-{int(row.id)}-{int(now.timestamp())}"
            await self.store.add_fill(
                order_id=order_id,
                market_id=market_id,
                side=side,
                fill_price=float(closed["exit_price"]),
                size_usd=float(closed["size_usd"]),
                fee_usd=0.0,
                pnl_usd=realized,
                trading_mode=mode,
            )
            if realized != 0.0:
                await self.gatekeeper.register_paper_trade(realized, when=now)
            realized_total += realized
            closed_count += 1

        if closed_count > 0:
            await self.risk_engine.refresh_state()
            await self.notify_fn(
                "[포지션 로테이션]\n"
                f"시장 {market_id} | 방향 {side}\n"
                f"청산 {closed_count} | 실현손익 {realized_total:+.2f} USD"
            )
            return True
        return False
