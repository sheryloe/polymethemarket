from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable
from uuid import uuid4

from polymethemoney.adapters.polymarket_client import PolymarketClient
from polymethemoney.config import Settings
from polymethemoney.domain import ArbOpportunity, BasketLeg, ExecutionBundleIntent, TradingMode
from polymethemoney.services.execution_engine import ExecutionEngine
from polymethemoney.services.gatekeeper import Gatekeeper
from polymethemoney.state import RuntimeState
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)

NotifyFn = Callable[[str], Awaitable[None]]


class StructureAlphaService:
    def __init__(
        self,
        settings: Settings,
        store: Store,
        runtime_state: RuntimeState,
        polymarket_client: PolymarketClient,
        execution_engine: ExecutionEngine,
        gatekeeper: Gatekeeper,
        notify_fn: NotifyFn,
    ) -> None:
        self.settings = settings
        self.store = store
        self.runtime_state = runtime_state
        self.polymarket_client = polymarket_client
        self.execution_engine = execution_engine
        self.gatekeeper = gatekeeper
        self.notify_fn = notify_fn

    async def run(self) -> None:
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("structure_alpha tick error: %s", exc)
            await asyncio.sleep(max(5, self.settings.arb_scan_interval_seconds))

    async def _tick(self) -> None:
        if not self.settings.arb_enabled:
            return
        await self._try_early_exit()
        if self.runtime_state.kill_switch or self.runtime_state.paused:
            await self._refresh_runtime_metrics()
            return
        opportunities = await self._discover_opportunities()
        self.runtime_state.structure_alpha.detected += len(opportunities)
        if opportunities:
            await self._try_enter(opportunities[0])
        await self._refresh_runtime_metrics()

    async def _discover_opportunities(self) -> list[ArbOpportunity]:
        markets = await self.polymarket_client.fetch_markets(limit=self.settings.arb_market_limit)
        opportunities: list[ArbOpportunity] = []
        for market in markets:
            token_ids = self._parse_json_list(market.get("clobTokenIds"))
            outcomes = self._parse_json_list(market.get("outcomes"))
            if len(token_ids) < 2:
                continue
            if len(token_ids) > self.settings.arb_max_legs:
                continue
            kind = "pair" if len(token_ids) == 2 else "basket"
            event_id = str(market.get("conditionId") or market.get("eventSlug") or market.get("id") or "")
            if not event_id:
                continue
            books = await self.polymarket_client.fetch_order_books(token_ids)
            opportunity = await self._evaluate_group(
                event_id=event_id,
                kind=kind,
                token_ids=token_ids,
                outcomes=outcomes,
                market_id=str(market.get("id") or ""),
                books=books,
            )
            if opportunity is not None:
                opportunities.append(opportunity)
        opportunities.sort(key=lambda x: x.net_edge_usd, reverse=True)
        return opportunities

    async def _evaluate_group(
        self,
        event_id: str,
        kind: str,
        token_ids: list[str],
        outcomes: list[str],
        market_id: str,
        books: dict[str, dict],
    ) -> ArbOpportunity | None:
        open_notional = await self.store.structure_open_notional()
        budget = max(0.0, self.settings.starting_capital_usd * self.settings.arb_capital_share - open_notional)
        if budget <= 0:
            return None

        first_prices: list[float] = []
        for token_id in token_ids:
            asks = books.get(token_id, {}).get("asks", [])
            if not asks:
                return None
            first_prices.append(float(asks[0].get("price", 0.0)))
        if any(price <= 0 for price in first_prices):
            return None

        q_cap_by_leg = [self.settings.max_position_usd / max(price, 1e-6) for price in first_prices]
        q_guess = min(min(q_cap_by_leg), budget / max(sum(first_prices), 1e-6), self.settings.max_position_usd)
        if q_guess <= 0:
            return None

        legs: list[BasketLeg] = []
        vwap_prices: list[float] = []
        for idx, token_id in enumerate(token_ids):
            ask_levels = books.get(token_id, {}).get("asks", [])
            ask_vwap = self._vwap(ask_levels, q_guess)
            if ask_vwap is None:
                return None
            vwap_prices.append(ask_vwap)
            name = outcomes[idx] if idx < len(outcomes) else f"outcome-{idx}"
            legs.append(
                BasketLeg(
                    market_id=market_id,
                    token_id=token_id,
                    outcome_idx=idx,
                    outcome_name=name,
                    vwap_price=ask_vwap,
                    size_shares=q_guess,
                    size_usd=q_guess * ask_vwap,
                )
            )

        q_adjusted = min(
            q_guess,
            min(self.settings.max_position_usd / max(price, 1e-6) for price in vwap_prices),
            budget / max(sum(vwap_prices), 1e-6),
        )
        if q_adjusted <= 0:
            return None
        if abs(q_adjusted - q_guess) > 1e-8:
            legs = [
                BasketLeg(
                    market_id=leg.market_id,
                    token_id=leg.token_id,
                    outcome_idx=leg.outcome_idx,
                    outcome_name=leg.outcome_name,
                    vwap_price=leg.vwap_price,
                    size_shares=q_adjusted,
                    size_usd=q_adjusted * leg.vwap_price,
                )
                for leg in legs
            ]

        gross_cost = sum(leg.size_usd for leg in legs)
        payout = float(q_adjusted)
        cost_buffer = gross_cost * ((self.settings.arb_fee_buffer_bps + self.settings.arb_slippage_buffer_bps) / 10000.0)
        gross_edge = payout - gross_cost
        net_edge = payout - gross_cost - cost_buffer
        if gross_cost <= 0:
            return None
        net_edge_pct = net_edge / gross_cost
        if net_edge_pct < self.settings.arb_min_net_edge_pct:
            return None

        return ArbOpportunity(
            opportunity_id=f"arb-{uuid4().hex[:12]}",
            event_id=event_id,
            kind=kind,
            legs=legs,
            gross_edge_usd=gross_edge,
            net_edge_usd=net_edge,
            net_edge_pct=net_edge_pct,
            ttl_seconds=max(3, self.settings.arb_exec_window_ms // 1000),
            detected_at=datetime.now(timezone.utc),
        )

    async def _try_enter(self, opportunity: ArbOpportunity) -> None:
        demo_unlimited = self.settings.demo_unlimited and self.runtime_state.trading_mode == TradingMode.PAPER
        if not demo_unlimited:
            open_count = len(await self.store.get_open_positions(self.runtime_state.trading_mode))
            open_count += await self.store.structure_open_legs_count()
            if open_count + len(opportunity.legs) > self.settings.max_positions:
                return

        self.runtime_state.structure_alpha.exec_attempts += 1
        intent = ExecutionBundleIntent(
            opportunity_id=opportunity.opportunity_id,
            event_id=opportunity.event_id,
            kind=opportunity.kind,
            legs=opportunity.legs,
            atomic_deadline_ms=self.settings.arb_exec_window_ms,
            target_payout_usd=opportunity.legs[0].size_shares,
            gross_edge_usd=opportunity.gross_edge_usd,
            net_edge_usd=opportunity.net_edge_usd,
        )
        result = await self.execution_engine.execute_bundle(intent)
        if result.status == "filled":
            self.runtime_state.structure_alpha.entered += 1
            self.runtime_state.structure_alpha.exec_success += 1
            await self.notify_fn(
                "[구조알파 진입]\n"
                f"유형 {opportunity.kind} | 이벤트 {opportunity.event_id}\n"
                f"레그 {len(opportunity.legs)}개 | 순엣지 {opportunity.net_edge_usd:+.2f} USD ({opportunity.net_edge_pct:.2%})"
            )

    async def _try_early_exit(self) -> None:
        bundles = await self.store.open_structure_bundles()
        for bundle in bundles:
            legs = await self.store.open_structure_positions(bundle.opportunity_id)
            if not legs:
                continue
            token_ids = [str(leg.token_id) for leg in legs]
            books = await self.polymarket_client.fetch_order_books(token_ids)
            exit_prices: dict[str, float] = {}
            proceeds = 0.0
            cost = 0.0
            for leg in legs:
                bid_levels = books.get(str(leg.token_id), {}).get("bids", [])
                bid_vwap = self._vwap(bid_levels, float(leg.size_shares))
                if bid_vwap is None:
                    break
                exit_prices[str(leg.token_id)] = bid_vwap
                proceeds += float(leg.size_shares) * bid_vwap
                cost += float(leg.size_usd)
            if len(exit_prices) != len(legs):
                continue
            current_profit = proceeds - cost
            target_profit = float(bundle.net_edge_usd) * self.settings.arb_early_exit_edge_capture
            if current_profit < target_profit or current_profit <= 0:
                continue
            realized = await self.store.close_structure_bundle(
                opportunity_id=bundle.opportunity_id,
                exit_prices=exit_prices,
                status="closed",
                rolled_back=False,
                error=None,
            )
            if self.runtime_state.trading_mode == TradingMode.PAPER and realized != 0:
                await self.gatekeeper.register_paper_trade(realized, when=datetime.now(timezone.utc))
            await self.notify_fn(
                "[구조알파 조기청산]\n"
                f"유형 {bundle.kind} | 이벤트 {bundle.event_id}\n"
                f"실현손익 {realized:+.2f} USD"
            )

    async def _refresh_runtime_metrics(self) -> None:
        summary = await self.store.structure_status_snapshot()
        self.runtime_state.structure_alpha.open_pairs = int(summary["open_pairs"])
        self.runtime_state.structure_alpha.open_baskets = int(summary["open_baskets"])
        self.runtime_state.structure_alpha.realized_pnl_usd = float(summary["week_realized"])
        open_positions = await self.store.open_structure_positions()
        unrealized = 0.0
        if open_positions:
            token_ids = [str(row.token_id) for row in open_positions]
            books = await self.polymarket_client.fetch_order_books(token_ids)
            for row in open_positions:
                bid_levels = books.get(str(row.token_id), {}).get("bids", [])
                bid_vwap = self._vwap(bid_levels, float(row.size_shares))
                if bid_vwap is None:
                    continue
                unrealized += (float(row.size_shares) * bid_vwap) - float(row.size_usd)
        self.runtime_state.structure_alpha.unrealized_pnl_usd = unrealized

    @staticmethod
    def _parse_json_list(value: object) -> list[str]:
        if isinstance(value, list):
            return [str(v) for v in value if str(v).strip()]
        if isinstance(value, str) and value.strip():
            try:
                parsed = json.loads(value)
            except json.JSONDecodeError:
                return []
            if isinstance(parsed, list):
                return [str(v) for v in parsed if str(v).strip()]
        return []

    @staticmethod
    def _vwap(levels: list[dict], target_size: float) -> float | None:
        remaining = max(0.0, float(target_size))
        if remaining <= 0:
            return None
        notional = 0.0
        for level in levels:
            try:
                price = float(level.get("price", 0.0))
                size = float(level.get("size", 0.0))
            except (TypeError, ValueError):
                continue
            if price <= 0 or size <= 0:
                continue
            take = min(remaining, size)
            notional += take * price
            remaining -= take
            if remaining <= 1e-9:
                break
        if remaining > 1e-9:
            return None
        return notional / max(float(target_size), 1e-9)
