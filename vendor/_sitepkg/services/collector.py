from __future__ import annotations

import asyncio
import json
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
        self.market_expiries: dict[str, datetime | None] = {}
        self._recent_expiries: dict[str, datetime] = {}
        self._market_ids_version: int = 0
        self._last_settlement_sync_at: datetime | None = None

    async def bootstrap(self) -> None:
        markets = await self.client.fetch_markets(limit=500)
        selected = await self._select_markets(markets)
        self._update_market_ids(selected)
        for tick in selected:
            await self.store.add_market_tick(tick)
            await self.tick_queue.put(tick)
        await self.sync_settlement_labels()
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
            await asyncio.sleep(self._reconciliation_sleep_seconds())
            try:
                markets = await self.client.fetch_markets(limit=150)
                selected = await self._select_markets(markets)
                self._update_market_ids(selected)
                for tick in selected:
                    await self.store.add_market_tick(tick)
                    await self.tick_queue.put(tick)
                await self.sync_settlement_labels(force=self._should_force_settlement_sync())
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                logger.exception("Collector reconciliation error: %s", exc)

    async def sync_settlement_labels(self, force: bool = False) -> dict[str, int]:
        now = datetime.now(timezone.utc)
        if not force and self._last_settlement_sync_at is not None:
            if now - self._last_settlement_sync_at < timedelta(minutes=10):
                return {"markets": 0, "outcomes": 0, "scanned": 0}

        lookback_days = max(1, min(int(self.settings.training_lookback_days), 7))
        since = now - timedelta(days=lookback_days)
        page_size = 500
        max_pages = 12
        scanned = 0
        market_rows: list[dict[str, Any]] = []
        outcome_rows: list[dict[str, Any]] = []
        seen_market_ids: set[str] = set()

        for page in range(max_pages):
            markets = await self.client.fetch_markets(
                limit=page_size,
                active=None,
                closed=True,
                offset=page * page_size,
                order="updatedAt",
                ascending=False,
            )
            if not markets:
                break
            scanned += len(markets)
            for market in markets:
                if not self._matches_contrarian_market(market):
                    continue
                if not self._is_resolved_market(market):
                    continue
                market_id = self._market_id_from_snapshot(market)
                if not market_id or market_id in seen_market_ids:
                    continue
                resolved_at = self._resolved_at_for_market(market)
                end_ts = self._parse_end_date(market)
                reference_ts = resolved_at or end_ts
                if reference_ts is None or reference_ts < since:
                    continue
                outcome_yes = self._infer_outcome_yes(market)
                if outcome_yes is None:
                    continue
                market_rows.append(self._market_metadata_row(market, now))
                outcome_rows.append(self._outcome_row(market, outcome_yes, resolved_at or end_ts or now, now))
                seen_market_ids.add(market_id)
            if len(markets) < page_size:
                break

        market_count = await self.store.upsert_market_metadata(market_rows)
        outcome_count = await self.store.upsert_outcomes(outcome_rows)
        self._last_settlement_sync_at = now
        if market_count or outcome_count:
            logger.info(
                "Settlement sync completed: markets=%s outcomes=%s scanned=%s lookback_days=%s",
                market_count,
                outcome_count,
                scanned,
                lookback_days,
            )
        return {"markets": market_count, "outcomes": outcome_count, "scanned": scanned}

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
        return CollectorService._parse_datetime_value(market.get("endDate") or market.get("end_date"))

    @staticmethod
    def _parse_datetime_value(value: Any) -> datetime | None:
        if isinstance(value, datetime):
            return value.astimezone(timezone.utc) if value.tzinfo else value.replace(tzinfo=timezone.utc)
        if isinstance(value, str):
            raw = value.strip()
            if not raw:
                return None
            try:
                if raw.endswith("Z"):
                    raw = raw[:-1] + "+00:00"
                return datetime.fromisoformat(raw).astimezone(timezone.utc)
            except ValueError:
                return None
        return None

    @staticmethod
    def _parse_json_list(raw: Any) -> list[Any]:
        if isinstance(raw, list):
            return raw
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            try:
                parsed = json.loads(text)
            except json.JSONDecodeError:
                return []
            return parsed if isinstance(parsed, list) else []
        return []

    @staticmethod
    def _market_id_from_snapshot(market: dict[str, Any]) -> str:
        return str(
            market.get("id")
            or market.get("market")
            or market.get("market_id")
            or market.get("conditionId")
            or ""
        ).strip()

    def _resolved_at_for_market(self, market: dict[str, Any]) -> datetime | None:
        return (
            self._parse_datetime_value(market.get("umaEndDate"))
            or self._parse_datetime_value(market.get("closedTime"))
            or self._parse_datetime_value(market.get("closed_time"))
            or self._parse_end_date(market)
        )

    def _is_resolved_market(self, market: dict[str, Any]) -> bool:
        if market.get("closed") is not True:
            return False
        resolution_status = str(market.get("umaResolutionStatus") or "").strip().lower()
        if resolution_status in {"resolved", "settled", "finalized"}:
            return True
        return bool(market.get("automaticallyResolved") or market.get("closedTime") or market.get("umaEndDate"))

    def _infer_outcome_yes(self, market: dict[str, Any]) -> float | None:
        outcomes = [str(v).strip().lower() for v in self._parse_json_list(market.get("outcomes"))]
        raw_prices = self._parse_json_list(market.get("outcomePrices"))
        prices: list[float] = []
        for value in raw_prices:
            try:
                prices.append(float(value))
            except (TypeError, ValueError):
                prices.append(0.0)

        yes_prob: float | None = None
        if len(outcomes) >= 2 and len(prices) >= 2:
            for idx, label in enumerate(outcomes):
                if idx >= len(prices):
                    continue
                if label in {"yes", "up"}:
                    yes_prob = prices[idx]
                    break
                if label in {"no", "down"}:
                    yes_prob = 1.0 - prices[idx]
                    break
        elif prices:
            yes_prob = prices[0]

        if yes_prob is None:
            yes_prob = self._to_float(market, ["lastTradePrice", "bestBid", "bestAsk"], fallback=None)
        if yes_prob is None:
            return None
        if yes_prob >= 0.99:
            return 1.0
        if yes_prob <= 0.01:
            return 0.0
        return None

    def _market_metadata_row(self, market: dict[str, Any], now: datetime) -> dict[str, Any]:
        market_id = self._market_id_from_snapshot(market)
        token_ids = [str(v) for v in self._parse_json_list(market.get("clobTokenIds")) if str(v).strip()]
        token_id = str(
            market.get("tokenId")
            or market.get("token_id")
            or (token_ids[0] if token_ids else "")
        ).strip()
        events = market.get("events")
        event_slug = str(market.get("eventSlug") or market.get("event_slug") or market.get("slug") or "").strip()
        if isinstance(events, list) and events and isinstance(events[0], dict):
            event_slug = str(events[0].get("slug") or event_slug).strip()
        return {
            "market_id": market_id,
            "condition_id": str(market.get("conditionId") or market.get("condition_id") or market_id),
            "token_id": token_id,
            "token_ids_json": token_ids,
            "outcomes_json": [str(v) for v in self._parse_json_list(market.get("outcomes"))],
            "question": str(market.get("question") or market.get("title") or "").strip(),
            "category": str(market.get("category") or "").strip() or None,
            "event_slug": event_slug or None,
            "open_interest": self._to_float(market, ["openInterest", "open_interest", "liquidity"], fallback=0.0),
            "volume_24h": self._to_float(market, ["volume24hr", "volume24h", "volume"], fallback=0.0),
            "best_bid": self._to_float(market, ["bestBid", "bid", "best_bid"], fallback=0.0),
            "best_ask": self._to_float(market, ["bestAsk", "ask", "best_ask"], fallback=0.0),
            "status": "closed" if market.get("closed") is True else "active",
            "end_ts": self._parse_end_date(market),
            "source": "gamma",
            "ingested_at": now,
            "idempotency_key": f"market:{market_id}",
            "raw_json": market,
        }

    def _outcome_row(
        self,
        market: dict[str, Any],
        outcome_yes: float,
        resolved_at: datetime,
        now: datetime,
    ) -> dict[str, Any]:
        market_id = self._market_id_from_snapshot(market)
        return {
            "market_id": market_id,
            "outcome_yes": float(outcome_yes),
            "resolved_at": resolved_at,
            "status": "resolved",
            "source": "gamma",
            "ingested_at": now,
            "idempotency_key": f"outcome:{market_id}",
            "raw_json": market,
        }

    def _update_market_ids(self, ticks: list[MarketTick]) -> None:
        now = datetime.now(timezone.utc)
        timeout = max(5, int(self.settings.outcome_resolution_timeout_seconds))
        previous_expiries = dict(self.market_expiries)
        new_ids = [t.market_id for t in ticks]
        if new_ids != self.market_ids:
            removed_ids = [market_id for market_id in self.market_ids if market_id not in new_ids]
            for market_id in removed_ids:
                expiry = previous_expiries.get(market_id)
                if isinstance(expiry, datetime):
                    self._recent_expiries[market_id] = expiry
            self._recent_expiries = {
                market_id: expiry
                for market_id, expiry in self._recent_expiries.items()
                if isinstance(expiry, datetime) and now <= expiry + timedelta(seconds=timeout)
            }
            self.market_ids = new_ids
            self._market_ids_version += 1
            logger.info("Contrarian market rollover ids=%s recent_expiries=%s", self.market_ids, len(self._recent_expiries))
        self.market_stats = {t.market_id: (t.volume_1h, t.open_interest) for t in ticks}
        self.market_expiries = {t.market_id: t.expiry_ts for t in ticks}

    def _reconciliation_sleep_seconds(self) -> int:
        if not self._contrarian_enabled():
            return 300
        if not self.market_ids:
            return max(1, int(self.settings.strategy_reconcile_idle_seconds))
        now = datetime.now(timezone.utc)
        expiries = [expiry for expiry in self.market_expiries.values() if isinstance(expiry, datetime)]
        if not expiries:
            return max(1, int(self.settings.strategy_reconcile_idle_seconds))
        nearest = min(expiries)
        if nearest <= now + timedelta(seconds=90):
            return max(1, int(self.settings.strategy_reconcile_near_expiry_seconds))
        return max(1, int(self.settings.strategy_reconcile_normal_seconds))

    def _should_force_settlement_sync(self) -> bool:
        if not self._contrarian_enabled():
            return False
        now = datetime.now(timezone.utc)
        timeout = max(5, int(self.settings.outcome_resolution_timeout_seconds))
        for expiry in self._iter_expiries_for_settlement_sync(now):
            if not isinstance(expiry, datetime):
                continue
            if expiry <= now <= expiry + timedelta(seconds=timeout):
                return True
        return False

    def _iter_expiries_for_settlement_sync(self, now: datetime) -> list[datetime]:
        timeout = max(5, int(self.settings.outcome_resolution_timeout_seconds))
        self._recent_expiries = {
            market_id: expiry
            for market_id, expiry in self._recent_expiries.items()
            if isinstance(expiry, datetime) and now <= expiry + timedelta(seconds=timeout)
        }
        expiries = [expiry for expiry in self.market_expiries.values() if isinstance(expiry, datetime)]
        expiries.extend(self._recent_expiries.values())
        return expiries

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
        market_id = self._market_id_from_snapshot(market)
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
