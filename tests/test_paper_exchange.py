from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import AsyncMock

import pytest

from polymethemoney.adapters.paper_exchange import MarketQuote, PaperExchange
from polymethemoney.config import Settings
from polymethemoney.domain import DecisionType, OrderIntent, STRATEGY_LEGACY, Side


@pytest.mark.asyncio
async def test_paper_exchange_rejects_when_no_venues_and_no_fallback() -> None:
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@x:5432/x",
        REDIS_URL="redis://localhost:6379/0",
        PAPER_EXCHANGE_VENUES="",
        PAPER_TICKER_SYMBOLS="",
        PAPER_FALLBACK_TO_POLYMARKET=False,
        PAPER_POST_ONLY=False,
    )
    exchange = PaperExchange(settings=settings, taker_fee_bps=10.0)
    result = await exchange.place_limit_order(
        OrderIntent(
            strategy_id=STRATEGY_LEGACY,
            signal_id="s1",
            market_id="m-empty",
            side=Side.YES,
            price=0.5,
            size_usd=5.0,
            mode=DecisionType.AUTO,
        ),
    )
    assert result.status == "rejected_no_quote"


@pytest.mark.asyncio
async def test_paper_exchange_fills_yes_with_stubbed_symbol_quote() -> None:
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@x:5432/x",
        REDIS_URL="redis://localhost:6379/0",
        PAPER_EXCHANGE_VENUES="binance",
        PAPER_TICKER_SYMBOLS="BTCUSDT",
        PAPER_MARKET_SYMBOL_MAP='{"m1":"BTCUSDT"}',
        PAPER_POST_ONLY=False,
    )
    exchange = PaperExchange(settings=settings, taker_fee_bps=10.0)
    quote = MarketQuote(
        bid=0.480,
        ask=0.490,
        last_price=0.49,
        source="stub",
        symbol="BTCUSDT",
        updated_at=datetime.now(timezone.utc),
    )
    exchange._get_venue_quote = AsyncMock(return_value=quote)  # type: ignore[method-assign]

    result = await exchange.place_limit_order(
        OrderIntent(
            strategy_id=STRATEGY_LEGACY,
            signal_id="s2",
            market_id="m1",
            side=Side.YES,
            price=0.50,
            size_usd=5.0,
            mode=DecisionType.AUTO,
        ),
    )
    assert result.status == "filled"
    assert result.fill_price == 0.490
    assert result.fee_usd > 0


@pytest.mark.asyncio
async def test_paper_exchange_uses_stable_hash_symbol_for_unmapped_market() -> None:
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@x:5432/x",
        REDIS_URL="redis://localhost:6379/0",
        PAPER_EXCHANGE_VENUES="binance",
        PAPER_TICKER_SYMBOLS="BTCUSDT,ETHUSDT,SOLUSDT",
        PAPER_MARKET_SYMBOL_MAP="{}",
        PAPER_POST_ONLY=False,
    )
    exchange = PaperExchange(settings=settings, taker_fee_bps=10.0)

    called: list[str] = []

    async def fake_get_quote(symbol: str) -> MarketQuote:
        called.append(symbol)
        return MarketQuote(
            bid=0.330,
            ask=0.340,
            last_price=0.34,
            source="stub",
            symbol=symbol,
            updated_at=datetime.now(timezone.utc),
        )

    exchange._get_venue_quote = fake_get_quote  # type: ignore[method-assign]

    market_id = "market-foo"
    await exchange.place_limit_order(
        OrderIntent(
            strategy_id=STRATEGY_LEGACY,
            signal_id="s3",
            market_id=market_id,
            side=Side.NO,
            price=0.70,
            size_usd=5.0,
            mode=DecisionType.AUTO,
        ),
    )
    assert called == [exchange._stable_symbol_for_market(market_id)]
