from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Deque

from polymethemoney.domain import OrderIntent, Signal, TradingMode, utcnow


@dataclass(slots=True)
class StructureAlphaRuntime:
    detected: int = 0
    entered: int = 0
    exec_attempts: int = 0
    exec_success: int = 0
    open_pairs: int = 0
    open_baskets: int = 0
    realized_pnl_usd: float = 0.0
    unrealized_pnl_usd: float = 0.0


@dataclass(slots=True)
class TimedReasonEvent:
    timestamp: datetime
    reason: str


@dataclass(slots=True)
class RuntimeState:
    trading_mode: TradingMode = TradingMode.PAPER
    paused: bool = False
    kill_switch: bool = False
    manual_live_approved: bool = False
    pending_approvals: dict[str, OrderIntent] = field(default_factory=dict)
    recent_signals: Deque[Signal] = field(default_factory=lambda: deque(maxlen=30))
    recent_signal_candidate_sides: Deque[str] = field(default_factory=lambda: deque(maxlen=300))
    recent_signal_candidate_rejects: Deque[str] = field(default_factory=lambda: deque(maxlen=300))
    recent_approved_signal_sides: Deque[str] = field(default_factory=lambda: deque(maxlen=300))
    recent_no_guard_rejects: Deque[TimedReasonEvent] = field(default_factory=lambda: deque(maxlen=1000))
    min_position_usd_override: float | None = None
    auto_tune_zero_fill_applied: bool = False
    auto_tune_zero_fill_applied_at: datetime | None = None
    auto_tune_zero_fill_last_outcome: str | None = None
    auto_tune_zero_fill_last_checked_at: datetime | None = None
    tuning_overrides: dict[str, float] = field(default_factory=dict)
    tuning_originals: dict[str, float] = field(default_factory=dict)
    tuning_last_applied_at: datetime | None = None
    structure_alpha: StructureAlphaRuntime = field(default_factory=StructureAlphaRuntime)
    started_at: datetime = field(default_factory=utcnow)
    last_data_at: datetime | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
