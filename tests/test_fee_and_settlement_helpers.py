from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

from polymethemoney.storage import Store


ROOT = Path(__file__).resolve().parents[1]


def _load_collector_class():
    module_name = "polymethemoney.services.collector_runtime_test"
    if module_name in sys.modules:
        return sys.modules[module_name].CollectorService
    stub_name = "polymethemoney.adapters.polymarket_client"
    if stub_name not in sys.modules:
        stub = type(sys)(stub_name)
        stub.PolymarketClient = type("PolymarketClient", (), {})
        sys.modules[stub_name] = stub
    file_path = ROOT / "vendor" / "_sitepkg" / "services" / "collector.py"
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load collector module from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.CollectorService


def test_estimate_position_unrealized_includes_entry_and_exit_fee() -> None:
    result = Store.estimate_position_unrealized(
        size_usd=1000.0,
        entry_price=0.5,
        mark_price=0.6,
        entry_fee_usd=10.0,
        exit_fee_bps=100.0,
    )

    assert result["shares"] == 2000.0
    assert result["exit_value"] == 1200.0
    assert result["gross_unrealized_pnl_usd"] == 200.0
    assert result["exit_fee_usd"] == 12.0
    assert result["net_unrealized_pnl_usd"] == 178.0


def test_infer_outcome_yes_from_resolved_updown_market() -> None:
    collector_cls = _load_collector_class()
    collector = collector_cls(
        settings=SimpleNamespace(training_lookback_days=7),
        client=None,
        store=None,
        tick_queue=None,
        runtime_state=None,
    )

    up_market = {
        "id": "1927201",
        "closed": True,
        "umaResolutionStatus": "resolved",
        "outcomes": "[\"Up\", \"Down\"]",
        "outcomePrices": "[\"1\", \"0\"]",
    }
    down_market = {
        "id": "1927202",
        "closed": True,
        "umaResolutionStatus": "resolved",
        "outcomes": "[\"Up\", \"Down\"]",
        "outcomePrices": "[\"0\", \"1\"]",
    }
    open_market = {
        "id": "1927203",
        "closed": False,
        "umaResolutionStatus": "",
        "outcomes": "[\"Up\", \"Down\"]",
        "outcomePrices": "[\"0.62\", \"0.38\"]",
    }

    assert collector._infer_outcome_yes(up_market) == 1.0
    assert collector._infer_outcome_yes(down_market) == 0.0
    assert collector._infer_outcome_yes(open_market) is None
