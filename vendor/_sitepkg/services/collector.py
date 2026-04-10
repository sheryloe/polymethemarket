from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

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
        self._market_ids_version: int = 0

    async def bootstrap(self) -> None:
        markets = await self.client.fetch_markets(limit=500)
        selected = await self._select_markets(markets)
        self._update_market_ids(selected)
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
                if not self.market_ids:
                    await asyncio.sleep(2)
                    continue
                current_version = self._market_ids_version
                current_ids = list(self.market_ids)
                async for tick in self.client.stream_market_ticks(current_ids):
                    if self._market_ids_version != current_version:
                        break
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
            await asyncio.sleep(60 if self._contrarian_enabled() else 300)
            try:
                markets = await self.client.fetch_markets(limit=150)
                selected = await self._select_markets(markets)
                self._update_market_ids(selected)
                for tick in selected:
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

    def _contrarian_enabled(self) -> bool:
        return bool(self.settings.contrarian_enabled or self.settings.ab_test_enabled)

    async def _select_markets(self, markets: list[dict[str, Any]]) -> list[MarketTick]:
        if self._contrarian_enabled():
            return await self._select_contrarian_markets(markets)
        ticks: list[MarketTick] = []
        for market in markets:
            tick = self._tick_from_market_snapshot(market)
            if tick is None:
                continue
            if not self._passes_liquidity_filter(tick):
                continue
            ticks.append(tick)
        ticks.sort(key=lambda x: (x.open_interest, x.volume_1h), reverse=True)
        return ticks[:300]

    async def _select_contrarian_markets(self, markets: list[dict[str, Any]]) -> list[MarketTick]:
        now = datetime.now(timezone.utc)
        event_tick = await self._fetch_contrarian_event_tick(now)
        if event_tick is not None:
            return [event_tick]
        return self._select_contrarian_markets_from_list(markets, now)

    def _select_contrarian_markets_from_list(
        self,
        markets: list[dict[str, Any]],
        now: datetime,
    ) -> list[MarketTick]:
        candidates: list[tuple[datetime, MarketTick]] = []
        for market in markets:
            if not self._matches_contrarian_market(market):
                continue
            if not self._is_active_contrarian_window(market, now):
                continue
            end_ts = self._parse_end_date(market)
            if end_ts is None:
                continue
            tick = self._tick_from_market_snapshot(market)
            if tick is None:
                continue
            candidates.append((end_ts, tick))
        if not candidates:
            return []
        candidates.sort(key=lambda item: item[0])
        return [candidates[0][1]]

    async def _fetch_contrarian_event_tick(self, now: datetime) -> MarketTick | None:
        slug = self._current_contrarian_slug(now)
        if not slug:
            return None
        event = await self._fetch_event_by_slug(slug)
        if not event:
            return None
        markets = event.get("markets")
        if not isinstance(markets, list):
            return None
        for market in markets:
            if not isinstance(market, dict):
                continue
            if market.get("closed") is True:
                continue
            if market.get("active") is False:
                continue
            if market.get("acceptingOrders") is False:
                continue
            tick = self._tick_from_market_snapshot(market)
            if tick is not None:
                return tick
        return None

    async def _fetch_event_by_slug(self, slug: str) -> dict[str, Any] | None:
        url = self._events_url()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url, params={"slug": slug})
            response.raise_for_status()
            payload = response.json()
        if isinstance(payload, list):
            return payload[0] if payload and isinstance(payload[0], dict) else None
        if isinstance(payload, dict):
            events = payload.get("events")
            if isinstance(events, list) and events:
                return events[0] if isinstance(events[0], dict) else None
        return None

    def _events_url(self) -> str:
        base = str(self.settings.polymarket_gamma_url or "").strip()
        if not base:
            return "https://gamma-api.polymarket.com/events"
        if "/markets" in base:
            return base.rsplit("/markets", 1)[0] + "/events"
        return base.rstrip("/") + "/events"

    def _current_contrarian_slug(self, now: datetime) -> str | None:
        prefix = str(self.settings.contrarian_market_slug_prefix or "").strip()
        if not prefix:
            return None
        epoch = int(now.timestamp())
        window_start = epoch - (epoch % 300)
        return f"{prefix}{window_start}"

    def _matches_contrarian_market(self, market: dict[str, Any]) -> bool:
        prefix = str(self.settings.contrarian_market_slug_prefix or "").strip().lower()
        slug = str(market.get("eventSlug") or market.get("event_slug") or market.get("slug") or "").lower()
        if prefix and slug.startswith(prefix):
            return True
        question = str(market.get("question") or market.get("title") or market.get("marketTitle") or "").lower()
        return "bitcoin up or down - 5 minutes" in question

    def _is_active_contrarian_window(self, market: dict[str, Any], now: datetime) -> bool:
        end_ts = self._parse_end_date(market)
        if end_ts is None:
            return False
        if end_ts < now:
            return False
        return end_ts <= now + timedelta(minutes=5)

    @staticmethod
    def _parse_end_date(market: dict[str, Any]) -> datetime | None:
        end_date = market.get("endDate") or market.get("end_date")
        if isinstance(end_date, str):
            try:
                if end_date.endswith("Z"):
                    end_date = end_date[:-1] + "+00:00"
                return datetime.fromisoformat(end_date).astimezone(timezone.utc)
            except ValueError:
                return None
        return None

    def _update_market_ids(self, ticks: list[MarketTick]) -> None:
        new_ids = [t.market_id for t in ticks]
        if new_ids != self.market_ids:
            self.market_ids = new_ids
            self._market_ids_version += 1
        self.market_stats = {t.market_id: (t.volume_1h, t.open_interest) for t in ticks}

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
