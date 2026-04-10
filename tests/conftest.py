from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_module(module_name: str, file_path: Path) -> None:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load module spec: {module_name} <- {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)


ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = ROOT / "src" / "polymethemoney"
VENDOR_ROOT = ROOT / "vendor" / "_sitepkg"

_load_module("polymethemoney.models", VENDOR_ROOT / "models.py")
_load_module("polymethemoney.domain", VENDOR_ROOT / "domain.py")
_load_module("polymethemoney.db", VENDOR_ROOT / "db.py")
_load_module("polymethemoney.state", VENDOR_ROOT / "state.py")
_load_module("polymethemoney.storage", VENDOR_ROOT / "storage.py")
_load_module("polymethemoney.config", SRC_ROOT / "config.py")
_load_module("polymethemoney.adapters.paper_exchange", SRC_ROOT / "adapters" / "paper_exchange.py")

# Provide a minimal adapters module to avoid heavy deps (websockets) during tests.
_paper_exchange = sys.modules["polymethemoney.adapters.paper_exchange"]
_adapters = types.ModuleType("polymethemoney.adapters")
_adapters.PaperExchange = _paper_exchange.PaperExchange

class _PolymarketClient:  # pragma: no cover - stub for import wiring
    def __init__(self, *args, **kwargs) -> None:
        pass

_adapters.PolymarketClient = _PolymarketClient
sys.modules["polymethemoney.adapters"] = _adapters

_services = types.ModuleType("polymethemoney.services")
for _name in (
    "CollectorService",
    "ContrarianEngine",
    "ExecutionEngine",
    "FeatureEngine",
    "Gatekeeper",
    "ModelEngine",
    "ReporterService",
    "RiskEngine",
    "SignalEngine",
    "StructureAlphaService",
    "TelegramBotService",
):
    setattr(_services, _name, type(_name, (), {}))
sys.modules["polymethemoney.services"] = _services

_load_module("polymethemoney.orchestrator", SRC_ROOT / "orchestrator.py")
