from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any

from polymethemoney.adapters.polymarket_client import PolymarketClient
from polymethemoney.config import Settings
from polymethemoney.domain import MarketTick
from polymethemoney.state import RuntimeState
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)


class CollectorService:
    def __init__(
        self,
        settings: Settings,
        client: PolymarketClient,
        store: Store,
        tick_queue: asyncio.Queue[MarketTick],
        runtime_state: RuntimeState,
    ) -> None:
        self.settings = settings
        self.client = client
        self.store = store
        self.tick_queue = tick_queue
        self.runtime_state = runtime_state
        self.market_ids: list[str] = []
        self.market_stats: dict[str, tuple[float, float]] = {}

    async def bootstrap(self) -> None:
        markets = await self.client.fetch_markets(limit=500)
        ticks: list[MarketTick] = []
        for market in markets:
            tick = self._tick_from_market_snapshot(market)
            if tick is None:
                continue
            if not self._passes_liquidity_filter(tick):
                continue
            ticks.append(tick)
        ticks.sort(key=lambda x: (x.open_interest, x.volume_1h), reverse=True)
        selected = ticks[:300]
        self.market_ids = [t.market_id for t in selected]
        for tick in selected:
            self.market_stats[tick.market_id] = (tick.volume_1h, tick.open_interest)
        for tick in selected:
            await self.store.add_market_tick(tick)
            await self.tick_queue.put(tick)
        logger.info("Bootstrap selected markets=%s", len(self.market_ids))

    async def run(self) -> None:
        await self.bootstrap()
        await asyncio.gather(self._stream_loop(), self._reconciliation_loop())

    async def _stream_loop(self) -> None:
        while True:
            try:
                async for tick in self.client.stream_market_ticks(self.market_ids):
                    self.runtime_state.last_data_at = datetime.now(timezone.utc)
                    tick = self._hydrate_tick(tick)
                    if tick.market_id not in self.market_stats:
                        continue
                    await self.store.add_market_tick(tick)
                    await self.tick_queue.put(tick)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Collector ws stream error: %s", exc)
                await asyncio.sleep(3)

    async def _reconciliation_loop(self) -> None:
        while True:
            await asyncio.sleep(300)
            try:
                markets = await self.client.fetch_markets(limit=150)
                for market in markets:
                    tick = self._tick_from_market_snapshot(market)
                    if tick is None or not self._passes_liquidity_filter(tick):
                        continue
                    self.market_stats[tick.market_id] = (tick.volume_1h, tick.open_interest)
                    await self.store.add_market_tick(tick)
                    await self.tick_queue.put(tick)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Collector reconciliation error: %s", exc)

    def _passes_liquidity_filter(self, tick: MarketTick) -> bool:
        return (
            tick.open_interest >= self.settings.min_open_interest_usd
            and (tick.ask - tick.bid) <= self.settings.max_spread
            and tick.volume_1h >= self.settings.min_hourly_volume_usd
        )

    @staticmethod
    def _to_float(market: dict[str, Any], keys: list[str], fallback: float = 0.0) -> float:
        for key in keys:
            if key in market:
                try:
                    return float(market[key])
                except (TypeError, ValueError):
                    continue
        return fallback

    def _tick_from_market_snapshot(self, market: dict[str, Any]) -> MarketTick | None:
        market_id = str(
            market.get("id")
            or market.get("market")
            or market.get("market_id")
            or market.get("conditionId")
            or ""
        )
        if not market_id:
            return None
        bid = self._to_float(market, ["bestBid", "bid", "best_bid"])
        ask = self._to_float(market, ["bestAsk", "ask", "best_ask"])
        if bid <= 0 or ask <= 0:
            return None
        last_price = self._to_float(market, ["lastTradePrice", "last_price", "price"], fallback=(bid + ask) / 2.0)
        volume_24h = self._to_float(market, ["volume24hr", "volume24h"], fallback=0.0)
        volume_total = self._to_float(market, ["volume", "volumeNum"], fallback=0.0)
        volume_1h = volume_24h
        if volume_1h <= 0 and volume_total > 0:
            volume_1h = volume_total / 24.0
        open_interest = self._to_float(
            market,
            ["openInterest", "open_interest", "liquidity", "liquidityNum"],
            fallback=0.0,
        )
        if open_interest <= 0 and volume_total > 0:
            # Gamma payload often omits true openInterest; use total traded volume as conservative proxy.
            open_interest = volume_total
        expiry_ts = None
        end_date = market.get("endDate") or market.get("end_date")
        if isinstance(end_date, str):
            try:
                if end_date.endswith("Z"):
                    end_date = end_date[:-1] + "+00:00"
                expiry_ts = datetime.fromisoformat(end_date).astimezone(timezone.utc)
            except ValueError:
                expiry_ts = None
        return MarketTick(
            market_id=market_id,
            bid=bid,
            ask=ask,
            last_price=max(0.001, min(0.999, last_price)),
            volume_1h=volume_1h,
            open_interest=open_interest,
            expiry_ts=expiry_ts,
            raw=market,
        )

    def _hydrate_tick(self, tick: MarketTick) -> MarketTick:
        cached = self.market_stats.get(tick.market_id)
        if cached is None:
            return tick
        cached_volume_1h, cached_open_interest = cached
        volume_1h = tick.volume_1h if tick.volume_1h > 0 else cached_volume_1h
        open_interest = tick.open_interest if tick.open_interest > 0 else cached_open_interest
        return MarketTick(
            market_id=tick.market_id,
            bid=tick.bid,
            ask=tick.ask,
            last_price=tick.last_price,
            volume_1h=volume_1h,
            open_interest=open_interest,
            expiry_ts=tick.expiry_ts,
            timestamp=tick.timestamp,
            raw=tick.raw,
        )
