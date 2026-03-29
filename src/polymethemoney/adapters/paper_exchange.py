from __future__ import annotations

import json
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any
from uuid import uuid4

import httpx

from polymethemoney.config import Settings
from polymethemoney.domain import FillResult, MarketTick, OrderIntent, TradingMode


@dataclass(slots=True)
class MarketQuote:
    bid: float
    ask: float
    last_price: float
    source: str
    symbol: str
    updated_at: datetime


class PaperExchange:
    def __init__(self, settings: Settings | None = None, taker_fee_bps: float | None = None) -> None:
        resolved_settings = settings or Settings()
        self.settings = resolved_settings
        self.taker_fee_bps = float(
            taker_fee_bps if taker_fee_bps is not None else self.settings.taker_fee_bps,
        )

        self._quotes: dict[str, MarketQuote] = {}
        self._market_volatility_hint: dict[str, float] = defaultdict(float)
        self._symbol_quotes: dict[str, MarketQuote] = {}
        self._symbol_windows: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=128))
        self._symbol_market_map: dict[str, str] = self._parse_market_symbol_map()
        self._venues = self._parse_venues()
        self._tick_symbols = self._parse_ticker_symbols()

    def _parse_venues(self) -> list[str]:
        venues = [
            v.strip().lower()
            for v in self.settings.paper_exchange_venues.split(",")
            if v and v.strip()
        ]
        return [v for v in venues if v in {"binance", "bybit"}]

    def _parse_ticker_symbols(self) -> list[str]:
        symbols = [s.strip().upper() for s in self.settings.paper_ticker_symbols.split(",") if s.strip()]
        return [s for s in symbols if s.endswith("USDT")]

    def _parse_market_symbol_map(self) -> dict[str, str]:
        raw = self.settings.paper_market_symbol_map.strip()
        if not raw:
            return {}
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        if not isinstance(payload, dict):
            return {}
        parsed: dict[str, str] = {}
        for key, value in payload.items():
            if not isinstance(key, str) or not isinstance(value, str):
                continue
            parsed[str(key)] = value.strip().upper()
        return parsed

    async def on_tick(self, tick: MarketTick) -> None:
        self._quotes[tick.market_id] = MarketQuote(
            bid=tick.bid,
            ask=tick.ask,
            last_price=tick.last_price,
            source="polymarket",
            symbol="polymarket",
            updated_at=tick.timestamp,
        )

    async def place_limit_order(self, intent: OrderIntent) -> FillResult:
        market_id = str(intent.market_id)
        symbol, quote = await self._resolve_price_quote(market_id)
        if quote is None:
            return FillResult(
                order_id=str(uuid4()),
                market_id=market_id,
                side=intent.side,
                fill_price=intent.price,
                size_usd=intent.size_usd,
                status="rejected_no_quote",
                fee_usd=0.0,
                mode=TradingMode.PAPER,
                created_at=datetime.now(timezone.utc),
            )

        if quote.ask <= 0 or quote.bid <= 0:
            return FillResult(
                order_id=str(uuid4()),
                market_id=market_id,
                side=intent.side,
                fill_price=intent.price,
                size_usd=intent.size_usd,
                status="rejected_bad_quote",
                fee_usd=0.0,
                mode=TradingMode.PAPER,
                created_at=datetime.now(timezone.utc),
            )

        spread = quote.ask - quote.bid
        if spread > self.settings.max_spread:
            return FillResult(
                order_id=str(uuid4()),
                market_id=market_id,
                side=intent.side,
                fill_price=intent.price,
                size_usd=intent.size_usd,
                status="rejected_wide_spread",
                fee_usd=0.0,
                mode=TradingMode.PAPER,
                created_at=datetime.now(timezone.utc),
            )

        yes_bid = quote.bid
        yes_ask = quote.ask
        no_bid = max(0.001, 1.0 - yes_ask)
        no_ask = max(0.001, 1.0 - yes_bid)

        post_only = bool(self.settings.paper_post_only)
        fill_price = intent.price
        status = "open"
        if intent.side.value == "YES":
            if post_only:
                if intent.price >= yes_ask:
                    return FillResult(
                        order_id=str(uuid4()),
                        market_id=market_id,
                        side=intent.side,
                        fill_price=intent.price,
                        size_usd=intent.size_usd,
                        status="rejected_post_only",
                        fee_usd=0.0,
                        mode=TradingMode.PAPER,
                        created_at=datetime.now(timezone.utc),
                    )
                if intent.price <= yes_bid:
                    fill_price = yes_bid
                    status = "filled"
                else:
                    fill_price = intent.price
                    status = "unfilled"
            else:
                if intent.price >= yes_ask:
                    fill_price = yes_ask
                    status = "filled"
        else:
            if post_only:
                if intent.price >= no_ask:
                    return FillResult(
                        order_id=str(uuid4()),
                        market_id=market_id,
                        side=intent.side,
                        fill_price=intent.price,
                        size_usd=intent.size_usd,
                        status="rejected_post_only",
                        fee_usd=0.0,
                        mode=TradingMode.PAPER,
                        created_at=datetime.now(timezone.utc),
                    )
                if intent.price <= no_bid:
                    fill_price = no_bid
                    status = "filled"
                else:
                    fill_price = intent.price
                    status = "unfilled"
            else:
                if intent.price >= no_ask:
                    fill_price = no_ask
                    status = "filled"
                else:
                    fill_price = intent.price
        if status != "filled":
            return FillResult(
                order_id=str(uuid4()),
                market_id=market_id,
                side=intent.side,
                fill_price=fill_price,
                size_usd=intent.size_usd,
                status="unfilled",
                fee_usd=0.0,
                mode=TradingMode.PAPER,
                created_at=datetime.now(timezone.utc),
            )

        fee_usd = intent.size_usd * (self.taker_fee_bps / 10000.0)
        return FillResult(
            order_id=str(uuid4()),
            market_id=market_id,
            side=intent.side,
            fill_price=max(0.001, min(0.999, fill_price)),
            size_usd=intent.size_usd,
            status="filled",
            fee_usd=fee_usd,
            mode=TradingMode.PAPER,
            created_at=datetime.now(timezone.utc),
        )

    async def _resolve_price_quote(self, market_id: str) -> tuple[str | None, MarketQuote | None]:
        tick_quote = self._quotes.get(market_id)
        if tick_quote is not None:
            return "polymarket", MarketQuote(
                bid=tick_quote.bid,
                ask=tick_quote.ask,
                last_price=tick_quote.last_price,
                source=tick_quote.source,
                symbol="polymarket",
                updated_at=tick_quote.updated_at,
            )

        symbol = self._symbol_market_map.get(market_id)
        if not symbol:
            return None, None

        venue_quote = await self._get_venue_quote(symbol)
        if venue_quote is not None:
            return symbol, venue_quote

        return None, None

    async def _get_venue_quote(self, symbol: str) -> MarketQuote | None:
        cached = self._symbol_quotes.get(symbol)
        ttl_seconds = int(self.settings.paper_ticker_ttl_seconds)
        if cached is not None:
            age = datetime.now(timezone.utc) - cached.updated_at
            if age <= timedelta(seconds=max(ttl_seconds, 1)):
                return cached

        last_error = None
        for venue in self._venues:
            if venue == "binance":
                value = await self._fetch_binance_quote(symbol)
            elif venue == "bybit":
                value = await self._fetch_bybit_quote(symbol)
            else:
                value = None
            if value is None:
                last_error = f"venue_empty:{venue}"
                continue
            yes_prob, source = value
            price = self._probability_to_orderbook(yes_prob, symbol)
            self._symbol_quotes[symbol] = price
            return price

        if last_error:
            return None
        return None

    async def _fetch_binance_quote(self, symbol: str) -> tuple[float, str] | None:
        try:
            async with httpx.AsyncClient(timeout=self.settings.paper_ticker_timeout_seconds) as client:
                response = await client.get(
                    self.settings.paper_binance_bookticker_url,
                    params={"symbol": symbol},
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError:
            return None
        if not isinstance(payload, dict):
            return None
        price = self._to_float(payload.get("lastPrice"), fallback=None)
        if price is None:
            price = self._to_float(payload.get("weightedAvgPrice"), fallback=None)
        if price is None:
            return None
        return price, "binance"

    async def _fetch_bybit_quote(self, symbol: str) -> tuple[float, str] | None:
        try:
            async with httpx.AsyncClient(timeout=self.settings.paper_ticker_timeout_seconds) as client:
                response = await client.get(
                    self.settings.paper_bybit_tickers_url,
                    params={"category": "spot", "symbol": symbol},
                )
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPError:
            return None
        result = payload.get("result") if isinstance(payload, dict) else None
        rows = result.get("list") if isinstance(result, dict) else None
        if not isinstance(rows, list) or not rows:
            return None
        first = rows[0]
        if not isinstance(first, dict):
            return None
        price = self._to_float(first.get("lastPrice"), fallback=None)
        if price is None:
            return None
        return price, "bybit"

    def _probability_to_orderbook(self, price: float, symbol: str) -> MarketQuote:
        window = self._symbol_windows[symbol]
        window.append(price)
        if len(window) >= 2:
            sorted_prices = sorted(window)
            floor_idx = max(0, int(len(sorted_prices) * 0.1))
            ceil_idx = max(floor_idx + 1, int(len(sorted_prices) * 0.9))
            lo = float(sorted_prices[floor_idx])
            hi = float(sorted_prices[ceil_idx - 1])
        else:
            # First tick fallback: use current price range as baseline.
            lo = price * 0.95
            hi = price * 1.05
        if hi <= lo:
            yes_prob = 0.5
        else:
            yes_prob = (price - lo) / (hi - lo)
        yes_prob = max(0.001, min(0.999, yes_prob))

        spread = self.settings.paper_ticker_spread_bps / 10000.0
        half_spread = max(0.0, spread) / 2.0
        yes_bid = max(0.001, min(0.999, yes_prob - half_spread))
        yes_ask = max(0.001, min(0.999, yes_prob + half_spread))
        return MarketQuote(
            bid=yes_bid,
            ask=yes_ask,
            last_price=yes_prob,
            source="venue",
            symbol=symbol,
            updated_at=datetime.now(timezone.utc),
        )

    @staticmethod
    def _to_float(value: Any, fallback: float | None = 0.0) -> float | None:
        if value is None:
            return fallback
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return fallback
        return parsed

    def _stable_symbol_for_market(self, market_id: str) -> str | None:
        return None
