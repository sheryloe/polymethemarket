from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder
from importlib.util import spec_from_file_location
from pathlib import Path

_PYC_DIR = Path(__file__).with_name("__pycache__")


class _ServicePycFinder(MetaPathFinder):
    def find_spec(self, fullname: str, path, target=None):
        if not fullname.startswith(__name__ + "."):
            return None
        mod = fullname.rsplit(".", 1)[-1]
        py = Path(__file__).with_name(f"{mod}.py")
        if py.exists():
            return None
        pyc = _PYC_DIR / f"{mod}.cpython-312.pyc"
        if not pyc.exists():
            return None
        return spec_from_file_location(fullname, pyc)


if not any(isinstance(finder, _ServicePycFinder) for finder in sys.meta_path):
    sys.meta_path.insert(0, _ServicePycFinder())

from .collector import CollectorService  # type: ignore  # noqa: E402
from .execution_engine import ExecutionEngine  # type: ignore  # noqa: E402
from .feature_engine import FeatureEngine  # type: ignore  # noqa: E402
from .gatekeeper import Gatekeeper  # type: ignore  # noqa: E402
from .model_engine import ModelEngine  # type: ignore  # noqa: E402
from .reporter import ReporterService  # type: ignore  # noqa: E402
from .risk_engine import RiskEngine  # type: ignore  # noqa: E402
from .signal_engine import SignalEngine  # type: ignore  # noqa: E402
from .structure_alpha_engine import StructureAlphaService  # type: ignore  # noqa: E402
from .telegram_bot import TelegramBotService  # noqa: E402

__all__ = [
    "CollectorService",
    "ExecutionEngine",
    "FeatureEngine",
    "Gatekeeper",
    "ModelEngine",
    "ReporterService",
    "RiskEngine",
    "SignalEngine",
    "StructureAlphaService",
    "TelegramBotService",
]
