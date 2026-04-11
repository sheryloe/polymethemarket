from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from polymethemoney.config import Settings
from polymethemoney.domain import (
    DecisionType,
    EXECUTION_MODE_FUNDED_PAPER,
    EXECUTION_MODE_SHADOW_PAPER,
    DualModelScore,
    FeatureVector,
    STRATEGY_EXPIRY_ANCHOR,
    STRATEGY_TAPE_RIDER,
    Side,
    Signal,
    get_strategy_spec,
    iter_active_strategy_specs,
)
from polymethemoney.state import RuntimeState
from polymethemoney.storage import Store
from polymethemoney.domain import TradingMode

ROOT = Path(__file__).resolve().parents[1]


def _load_risk_engine_class():
    module_name = "polymethemoney.services.risk_engine_runtime_test"
    if module_name in sys.modules:
        return sys.modules[module_name].RiskEngine
    services_module = sys.modules.get("polymethemoney.services")
    if services_module is None:
        services_module = types.ModuleType("polymethemoney.services")
        sys.modules["polymethemoney.services"] = services_module
    if not hasattr(services_module, "__path__"):
        services_module.__path__ = []
    gatekeeper_name = "polymethemoney.services.gatekeeper"
    if gatekeeper_name not in sys.modules:
        gatekeeper_module = types.ModuleType(gatekeeper_name)
        gatekeeper_module.Gatekeeper = type("Gatekeeper", (), {})
        sys.modules[gatekeeper_name] = gatekeeper_module
    file_path = ROOT / "src" / "polymethemoney" / "services" / "risk_engine.py"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load risk engine module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.RiskEngine


@pytest.mark.asyncio
async def test_strategy_registry_exposes_one_funded_one_shadow() -> None:
    specs = list(iter_active_strategy_specs())

    assert [spec.strategy_id for spec in specs] == [STRATEGY_EXPIRY_ANCHOR, STRATEGY_TAPE_RIDER]
    assert get_strategy_spec(STRATEGY_EXPIRY_ANCHOR).execution_mode == EXECUTION_MODE_FUNDED_PAPER
    assert get_strategy_spec(STRATEGY_TAPE_RIDER).execution_mode == EXECUTION_MODE_SHADOW_PAPER


@pytest.mark.asyncio
async def test_close_position_at_settlement_uses_terminal_price_without_exit_fee() -> None:
    win = Store.estimate_position_settlement(
        size_usd=100.0,
        entry_price=0.4,
        settled_price=1.0,
        entry_fee_usd=2.0,
    )
    lose = Store.estimate_position_settlement(
        size_usd=100.0,
        entry_price=0.5,
        settled_price=0.0,
        entry_fee_usd=2.0,
    )

    assert win["exit_fee_usd"] == 0.0
    assert win["settled_price"] == 1.0
    assert win["net_realized_pnl_usd"] == pytest.approx(148.0, rel=1e-6)
    assert lose["exit_fee_usd"] == 0.0
    assert lose["settled_price"] == 0.0
    assert lose["net_realized_pnl_usd"] == pytest.approx(-102.0, rel=1e-6)


def test_new_strategy_predictors_return_probabilities_in_range() -> None:
    module_name = "polymethemoney.services.model_engine_runtime_test"
    if module_name in sys.modules:
        model_module = sys.modules[module_name]
    else:
        file_path = ROOT / "vendor" / "_sitepkg" / "services" / "model_engine.py"
        spec = importlib.util.spec_from_file_location(module_name, str(file_path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot load model engine module from {file_path}")
        model_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = model_module
        spec.loader.exec_module(model_module)
    model_engine = model_module.ModelEngine(
        settings=Settings(DATABASE_URL="postgresql+asyncpg://x:x@x:5432/x", REDIS_URL="redis://localhost:6379/0"),
        store=SimpleNamespace(),
    )
    model_engine.predict_dual_score = lambda fv: DualModelScore(  # type: ignore[method-assign]
        settlement_prob=0.58,
        intraday_prob=0.54,
        blended_prob=0.56,
        confidence=0.63,
    )
    fv = FeatureVector(
        market_id="m1",
        implied_prob=0.52,
        spread=0.01,
        spread_pct=0.02,
        volume_1h=120000.0,
        open_interest=240000.0,
        volume_oi_ratio=0.5,
        time_to_expiry_hours=0.04,
        orderbook_imbalance=0.2,
        momentum_20=0.01,
        zscore_20=0.4,
        volatility_20=0.02,
        volatility_30=0.025,
    )

    expiry = model_engine.predict_strategy(STRATEGY_EXPIRY_ANCHOR, fv)
    tape = model_engine.predict_strategy(STRATEGY_TAPE_RIDER, fv)

    assert 0.001 <= expiry.fair_probability <= 0.999
    assert 0.05 <= expiry.confidence <= 0.99
    assert 0.001 <= tape.fair_probability <= 0.999
    assert 0.05 <= tape.confidence <= 0.99


@pytest.mark.asyncio
async def test_risk_engine_rejects_negative_net_ev_for_funded_and_allows_shadow() -> None:
    risk_engine_cls = _load_risk_engine_class()
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@x:5432/x",
        REDIS_URL="redis://localhost:6379/0",
        CONTRARIAN_ENABLED=True,
        FUNDED_POSITION_USD=500,
        SHADOW_POSITION_USD=500,
    )

    class _FakeStore:
        async def open_position_exposure(self, *args, **kwargs):
            return {"count": 0, "notional_usd": 0.0}

        async def fill_notional_since(self, *args, **kwargs):
            return 0.0

    risk_engine = risk_engine_cls(
        settings=settings,
        store=_FakeStore(),
        runtime_state=RuntimeState(trading_mode=TradingMode.PAPER),
        gatekeeper=SimpleNamespace(),
        alert_fn=lambda text: None,
    )
    async def _refresh_state():
        return risk_engine.state

    risk_engine.refresh_state = _refresh_state  # type: ignore[method-assign]

    funded_signal = Signal(
        strategy_id=STRATEGY_EXPIRY_ANCHOR,
        execution_mode=EXECUTION_MODE_FUNDED_PAPER,
        market_id="m-funded",
        side=Side.YES,
        fair_prob=0.52,
        implied_prob=0.53,
        edge=-0.01,
        net_ev=-0.02,
        score=10,
        ttl_seconds=60,
        liquidity_score=1.0,
        model_edge_score=1.0,
        regime_score=0.0,
        model_confidence=0.9,
    )
    shadow_signal = Signal(
        strategy_id=STRATEGY_TAPE_RIDER,
        execution_mode=EXECUTION_MODE_SHADOW_PAPER,
        market_id="m-shadow",
        side=Side.YES,
        fair_prob=0.52,
        implied_prob=0.53,
        edge=-0.01,
        net_ev=-0.02,
        score=10,
        ttl_seconds=60,
        liquidity_score=1.0,
        model_edge_score=1.0,
        regime_score=0.0,
        model_confidence=0.2,
    )

    funded_decision = await risk_engine.decide(funded_signal, 0)
    shadow_decision = await risk_engine.decide(shadow_signal, 0)

    assert funded_decision.kind == DecisionType.REJECT
    assert funded_decision.reason == "negative_net_ev"
    assert shadow_decision.kind == DecisionType.AUTO
    assert shadow_decision.size_usd == pytest.approx(500.0, rel=1e-6)


@pytest.mark.asyncio
async def test_risk_engine_allows_positive_funded_signal_within_caps() -> None:
    risk_engine_cls = _load_risk_engine_class()
    settings = Settings(
        DATABASE_URL="postgresql+asyncpg://x:x@x:5432/x",
        REDIS_URL="redis://localhost:6379/0",
        CONTRARIAN_ENABLED=True,
        FUNDED_POSITION_USD=500,
        FUNDED_MAX_OPEN_POSITIONS=3,
        FUNDED_MAX_OPEN_NOTIONAL_USD=1500,
        FUNDED_SAME_MARKET_SIDE_MAX_NOTIONAL_USD=1000,
        FUNDED_SLOT_NEW_NOTIONAL_USD=500,
        FUNDED_MIN_PREDICTION_GAP=0.03,
        FUNDED_MIN_MODEL_CONFIDENCE=0.55,
    )

    class _FakeStore:
        async def open_position_exposure(self, *args, **kwargs):
            return {"count": 0, "notional_usd": 0.0}

        async def fill_notional_since(self, *args, **kwargs):
            return 0.0

    risk_engine = risk_engine_cls(
        settings=settings,
        store=_FakeStore(),
        runtime_state=RuntimeState(trading_mode=TradingMode.PAPER),
        gatekeeper=SimpleNamespace(),
        alert_fn=lambda text: None,
    )

    async def _refresh_state():
        return risk_engine.state

    risk_engine.refresh_state = _refresh_state  # type: ignore[method-assign]

    funded_signal = Signal(
        strategy_id=STRATEGY_EXPIRY_ANCHOR,
        execution_mode=EXECUTION_MODE_FUNDED_PAPER,
        market_id="m-funded",
        side=Side.YES,
        fair_prob=0.60,
        implied_prob=0.55,
        edge=0.05,
        net_ev=0.04,
        score=80,
        ttl_seconds=60,
        liquidity_score=1.0,
        model_edge_score=20.0,
        regime_score=0.0,
        model_confidence=0.72,
        effective_edge=0.04,
    )

    funded_decision = await risk_engine.decide(funded_signal, 0)

    assert funded_decision.kind == DecisionType.AUTO
    assert funded_decision.reason == "funded_strategy"
    assert funded_decision.size_usd == pytest.approx(500.0, rel=1e-6)
