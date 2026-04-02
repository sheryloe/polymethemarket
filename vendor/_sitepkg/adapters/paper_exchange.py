from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from uuid import uuid4

from polymethemoney.domain import FillResult, MarketTick, OrderIntent, TradingMode


@dataclass(slots=True)
class MarketQuote:
    bid: float
    ask: float
    last_price: float
    updated_at: datetime


class PaperExchange:
    def __init__(self, taker_fee_bps: float) -> None:
        self.taker_fee_bps = taker_fee_bps
        self._quotes: dict[str, MarketQuote] = {}
        self._market_volatility_hint: dict[str, float] = defaultdict(float)

    def on_tick(self, tick: MarketTick) -> None:
        self._quotes[tick.market_id] = MarketQuote(
            bid=tick.bid,
            ask=tick.ask,
            last_price=tick.last_price,
            updated_at=tick.timestamp,
        )

    async def place_limit_order(self, intent: OrderIntent) -> FillResult:
        quote = self._quotes.get(intent.market_id)
        if quote is None:
            return FillResult(
                order_id=str(uuid4()),
                market_id=intent.market_id,
                side=intent.side,
                fill_price=intent.price,
                size_usd=intent.size_usd,
                status="rejected_no_quote",
                fee_usd=0.0,
                mode=TradingMode.PAPER,
            )
        order_id = str(uuid4())
        fill_price = intent.price
        status = "open"
        if intent.side.value == "YES":
            if intent.price >= quote.ask:
                fill_price = quote.ask
                status = "filled"
        else:
            no_bid = max(0.001, 1.0 - quote.ask)
            if intent.price >= no_bid:
                fill_price = no_bid
                status = "filled"
        if status != "filled":
            return FillResult(
                order_id=order_id,
                market_id=intent.market_id,
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
            order_id=order_id,
            market_id=intent.market_id,
            side=intent.side,
            fill_price=max(0.001, min(0.999, fill_price)),
            size_usd=intent.size_usd,
            status="filled",
            fee_usd=fee_usd,
            mode=TradingMode.PAPER,
            created_at=datetime.now(timezone.utc),
        )

