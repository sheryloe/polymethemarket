from __future__ import annotations

import asyncio
import importlib.util
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

from polymethemoney.config import Settings
from polymethemoney.domain import MarketTick, TradingMode
from polymethemoney.state import RuntimeState

ROOT = Path(__file__).resolve().parents[1]


def _load_collector_service_class():
    module_name = "polymethemoney.services.collector_rollover_runtime_test"
    if module_name in sys.modules:
        return sys.modules[module_name].CollectorService
    stub_name = "polymethemoney.adapters.polymarket_client"
    if stub_name not in sys.modules:
        stub = type(sys)(stub_name)
        stub.PolymarketClient = type("PolymarketClient", (), {})
        sys.modules[stub_name] = stub
    services_module = sys.modules.get("polymethemoney.services")
    if services_module is None:
        services_module = types.ModuleType("polymethemoney.services")
        sys.modules["polymethemoney.services"] = services_module
    if not hasattr(services_module, "__path__"):
        services_module.__path__ = []
    file_path = ROOT / "vendor" / "_sitepkg" / "services" / "collector.py"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load collector module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CollectorService


def test_collector_keeps_recent_expiry_during_rollover_for_forced_outcome_sync() -> None:
    collector_cls = _load_collector_service_class()
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@x:5432/x",
        REDIS_URL="redis://localhost:6379/0",
        CONTRARIAN_ENABLED=True,
        OUTCOME_RESOLUTION_TIMEOUT_SECONDS=180,
    )
    runtime = RuntimeState(trading_mode=TradingMode.PAPER)
    collector = collector_cls(
        settings=settings,
        client=types.SimpleNamespace(),
        store=types.SimpleNamespace(),
        tick_queue=asyncio.Queue(),
        runtime_state=runtime,
    )

    now = datetime.now(timezone.utc)
    expiring_tick = MarketTick(
        market_id="old-market",
        bid=0.49,
        ask=0.51,
        last_price=0.50,
        volume_1h=1000.0,
        open_interest=2000.0,
        expiry_ts=now - timedelta(seconds=30),
        raw={},
    )
    next_tick = MarketTick(
        market_id="new-market",
        bid=0.49,
        ask=0.51,
        last_price=0.50,
        volume_1h=1000.0,
        open_interest=2000.0,
        expiry_ts=now + timedelta(minutes=4),
        raw={},
    )

    collector._update_market_ids([expiring_tick])
    collector._update_market_ids([next_tick])

    assert collector._should_force_settlement_sync() is True
