from __future__ import annotations

from datetime import datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polymethemoney.config import Settings
from polymethemoney.domain import TradingMode
from polymethemoney.orchestrator import TradingApp


def _test_settings(**overrides: object) -> Settings:
    base = {
        "DATABASE_URL": "postgresql+asyncpg://x:x@x:5432/x",
        "REDIS_URL": "redis://localhost:6379/0",
        "CONTRARIAN_ENABLED": False,
        "AB_TEST_ENABLED": False,
        "PAPER_PERF_TUNE_ENABLED": True,
        "PAPER_PERF_TUNE_ONLY_IN_PAPER": True,
        "PAPER_PERF_TUNE_MIN_TRADES": 5,
        "SIGNAL_SIDE_POLICY": "BALANCED",
    }
    base.update(overrides)
    return Settings(**base)


def _build_ready_app(settings: Settings) -> TradingApp:
    app = TradingApp(settings)
    app.runtime = SimpleNamespace(trading_mode=TradingMode.PAPER)
    app.model_engine = SimpleNamespace(retrain_incremental=AsyncMock())
    app.notifier = SimpleNamespace(send=AsyncMock())
    app.store = SimpleNamespace()
    return app


@pytest.mark.asyncio
async def test_paper_perf_guard_triggers_policy_toggle_and_retrain_on_negative_return() -> None:
    settings = _test_settings(SIGNAL_SIDE_POLICY="BALANCED")
    app = _build_ready_app(settings)
    app._collect_recent_paper_performance = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "pnl_usd": -120.0,
            "trades": 7,
            "realized_pnl": -100.0,
            "unrealized_pnl": -20.0,
            "return_rate": -0.03,
        }
    )

    await app._paper_performance_guard(window_hours=6)

    assert settings.signal_side_policy == "YES_PRIORITY"
    app.model_engine.retrain_incremental.assert_awaited_once_with(target="both")
    assert app._last_paper_perf_tune_at is not None

    sent_messages = [call.args[0] for call in app.notifier.send.await_args_list]
    assert any("전략 변경 BASELINE -> AGGRESSIVE_ENTRY" in msg for msg in sent_messages)


@pytest.mark.asyncio
async def test_paper_perf_guard_skips_when_return_non_negative() -> None:
    settings = _test_settings(SIGNAL_SIDE_POLICY="YES_PRIORITY")
    app = _build_ready_app(settings)
    app._collect_recent_paper_performance = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "pnl_usd": 0.0,
            "trades": 9,
            "realized_pnl": 0.0,
            "unrealized_pnl": 0.0,
            "return_rate": 0.0,
        }
    )

    await app._paper_performance_guard(window_hours=6)

    assert settings.signal_side_policy == "YES_PRIORITY"
    app.model_engine.retrain_incremental.assert_not_awaited()
    assert app._last_paper_perf_tune_at is None


@pytest.mark.asyncio
async def test_paper_perf_guard_skips_when_trade_count_below_minimum() -> None:
    settings = _test_settings(PAPER_PERF_TUNE_MIN_TRADES=5, SIGNAL_SIDE_POLICY="BALANCED")
    app = _build_ready_app(settings)
    app._collect_recent_paper_performance = AsyncMock(  # type: ignore[method-assign]
        return_value={
            "pnl_usd": -50.0,
            "trades": 2,
            "realized_pnl": -45.0,
            "unrealized_pnl": -5.0,
            "return_rate": -0.0125,
        }
    )

    await app._paper_performance_guard(window_hours=6)

    assert settings.signal_side_policy == "BALANCED"
    app.model_engine.retrain_incremental.assert_not_awaited()
    assert app._last_paper_perf_tune_at is None


class _StoreForPerfCalc:
    def __init__(self) -> None:
        self.requested_mode = None
        self.requested_market_ids: list[str] | None = None
        self.positions = [
            SimpleNamespace(market_id="m1", side="YES", entry_price=0.5, size_usd=100.0),
            SimpleNamespace(market_id="m2", side="NO", entry_price=0.6, size_usd=50.0),
        ]

    async def realized_pnl_window(self, since: datetime) -> float:
        assert isinstance(since, datetime)
        return -120.0

    async def paper_trade_outcomes(self, since: datetime) -> list[float]:
        assert isinstance(since, datetime)
        return [-1.0, -2.0, 3.0, -4.0, 5.0, -6.0]

    async def get_open_positions(self, mode: TradingMode) -> list[SimpleNamespace]:
        self.requested_mode = mode
        return list(self.positions)

    async def latest_market_prices(self, market_ids: list[str]) -> dict[str, float]:
        self.requested_market_ids = list(market_ids)
        return {"m1": 0.4}


@pytest.mark.asyncio
async def test_collect_recent_paper_performance_uses_realized_plus_unrealized_return_rate() -> None:
    settings = _test_settings(STARTING_CAPITAL_USD=4000)
    app = TradingApp(settings)
    fake_store = _StoreForPerfCalc()
    app.store = fake_store

    result = await app._collect_recent_paper_performance(window_hours=6)

    assert result is not None
    assert fake_store.requested_mode == TradingMode.PAPER
    assert fake_store.requested_market_ids == ["m1", "m2"]
    assert result["trades"] == 6
    assert result["realized_pnl"] == pytest.approx(-120.0)
    assert result["unrealized_pnl"] == pytest.approx(-36.78, rel=1e-6)
    assert result["pnl_usd"] == pytest.approx(-156.78, rel=1e-6)
    assert result["return_rate"] == pytest.approx(-156.78 / 4000.0, rel=1e-6)
