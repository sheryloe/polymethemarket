from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


STRATEGY_LEGACY = "legacy"
STRATEGY_MODEL_A = "model_a_contrarian"
STRATEGY_MODEL_B = "model_b_ensemble"
AB_STRATEGY_IDS = (STRATEGY_MODEL_A, STRATEGY_MODEL_B)
EXECUTION_MODE_LEGACY = "legacy"
EXECUTION_MODE_FUNDED_PAPER = "funded_paper"
EXECUTION_MODE_SHADOW_PAPER = "shadow_paper"
STRATEGY_EXPIRY_ANCHOR = "expiry_anchor_active"
STRATEGY_TAPE_RIDER = "tape_rider_shadow"
STRATEGY_PULSE_FADE = "pulse_fade_reserved"
STRATEGY_PIN_MAGNET = "pin_magnet_reserved"


@dataclass(frozen=True, slots=True)
class StrategySpec:
    strategy_id: str
    display_name: str
    execution_mode: str
    predictor_key: str
    direction_mode: str
    position_usd: float
    max_positions: int
    enabled: bool = True
    report_order: int = 0


STRATEGY_REGISTRY: dict[str, StrategySpec] = {
    STRATEGY_EXPIRY_ANCHOR: StrategySpec(
        strategy_id=STRATEGY_EXPIRY_ANCHOR,
        display_name="ExpiryAnchor",
        execution_mode=EXECUTION_MODE_FUNDED_PAPER,
        predictor_key="expiry_anchor",
        direction_mode="direct",
        position_usd=500.0,
        max_positions=3,
        report_order=1,
    ),
    STRATEGY_TAPE_RIDER: StrategySpec(
        strategy_id=STRATEGY_TAPE_RIDER,
        display_name="TapeRider",
        execution_mode=EXECUTION_MODE_SHADOW_PAPER,
        predictor_key="tape_rider",
        direction_mode="direct",
        position_usd=500.0,
        max_positions=3,
        report_order=2,
    ),
    STRATEGY_PULSE_FADE: StrategySpec(
        strategy_id=STRATEGY_PULSE_FADE,
        display_name="PulseFade",
        execution_mode=EXECUTION_MODE_SHADOW_PAPER,
        predictor_key="pulse_fade",
        direction_mode="direct",
        position_usd=500.0,
        max_positions=3,
        enabled=False,
        report_order=3,
    ),
    STRATEGY_PIN_MAGNET: StrategySpec(
        strategy_id=STRATEGY_PIN_MAGNET,
        display_name="PinMagnet",
        execution_mode=EXECUTION_MODE_SHADOW_PAPER,
        predictor_key="pin_magnet",
        direction_mode="direct",
        position_usd=500.0,
        max_positions=3,
        enabled=False,
        report_order=4,
    ),
}
ACTIVE_STRATEGY_IDS = (STRATEGY_EXPIRY_ANCHOR, STRATEGY_TAPE_RIDER)


def get_strategy_spec(strategy_id: str) -> StrategySpec:
    return STRATEGY_REGISTRY.get(
        strategy_id,
        StrategySpec(
            strategy_id=strategy_id,
            display_name=strategy_id,
            execution_mode=EXECUTION_MODE_LEGACY,
            predictor_key="legacy",
            direction_mode="direct",
            position_usd=0.0,
            max_positions=0,
            enabled=False,
        ),
    )


def iter_active_strategy_specs() -> tuple[StrategySpec, ...]:
    return tuple(
        sorted(
            (spec for spec in STRATEGY_REGISTRY.values() if spec.enabled and spec.strategy_id in ACTIVE_STRATEGY_IDS),
            key=lambda spec: spec.report_order,
        )
    )


class Side(str, Enum):
    YES = "YES"
    NO = "NO"


class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class DecisionType(str, Enum):
    AUTO = "auto"
    SEMI = "semi"
    REJECT = "reject"


@dataclass(slots=True)
class MarketTick:
    market_id: str
    bid: float
    ask: float
    last_price: float
    volume_1h: float
    open_interest: float
    expiry_ts: datetime | None
    timestamp: datetime = field(default_factory=utcnow)
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class FeatureVector:
    market_id: str
    implied_prob: float
    spread: float
    spread_pct: float
    volume_1h: float
    open_interest: float
    volume_oi_ratio: float
    time_to_expiry_hours: float
    orderbook_imbalance: float
    momentum_20: float
    zscore_20: float
    volatility_20: float
    volatility_30: float
    timestamp: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class Signal:
    strategy_id: str
    market_id: str
    side: Side
    fair_prob: float
    implied_prob: float
    edge: float
    net_ev: float
    score: int
    ttl_seconds: int
    liquidity_score: float
    model_edge_score: float
    regime_score: float
    execution_mode: str = EXECUTION_MODE_LEGACY
    model_confidence: float = 0.0
    effective_edge: float = 0.0
    created_at: datetime = field(default_factory=utcnow)
    signal_id: str = field(default_factory=lambda: str(uuid4()))


@dataclass(slots=True)
class OrderIntent:
    strategy_id: str
    signal_id: str
    market_id: str
    side: Side
    price: float
    size_usd: float
    mode: DecisionType
    execution_mode: str = EXECUTION_MODE_LEGACY
    created_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class RiskState:
    daily_drawdown_pct: float = 0.0
    weekly_drawdown_pct: float = 0.0
    trading_mode: TradingMode = TradingMode.PAPER
    paused: bool = False


@dataclass(slots=True)
class Decision:
    kind: DecisionType
    reason: str
    size_usd: float = 0.0


@dataclass(slots=True)
class FillResult:
    strategy_id: str
    order_id: str
    market_id: str
    side: Side
    fill_price: float
    size_usd: float
    status: str
    fee_usd: float
    mode: TradingMode
    execution_mode: str = EXECUTION_MODE_LEGACY
    created_at: datetime = field(default_factory=utcnow)
    pnl_usd: float = 0.0


@dataclass(slots=True)
class BasketLeg:
    market_id: str
    token_id: str
    outcome_idx: int
    outcome_name: str
    vwap_price: float
    size_shares: float
    size_usd: float


@dataclass(slots=True)
class ArbOpportunity:
    opportunity_id: str
    event_id: str
    kind: str
    legs: list[BasketLeg]
    gross_edge_usd: float
    net_edge_usd: float
    net_edge_pct: float
    ttl_seconds: int
    detected_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class ExecutionBundleIntent:
    opportunity_id: str
    event_id: str
    kind: str
    legs: list[BasketLeg]
    atomic_deadline_ms: int
    target_payout_usd: float
    gross_edge_usd: float
    net_edge_usd: float


@dataclass(slots=True)
class ExecutionBundleResult:
    opportunity_id: str
    status: str
    filled_legs: int
    rolled_back: bool
    realized_pnl_usd: float
    opened_legs: int
    error: str | None = None


@dataclass(slots=True)
class PaperGateMetrics:
    profit_factor: float
    max_drawdown_pct: float
    trades: int
    violations: int
    covered_days: float


@dataclass(slots=True)
class MarketMeta:
    market_id: str
    condition_id: str
    token_id: str
    token_ids: list[str]
    outcomes: list[str]
    question: str
    category: str | None
    event_slug: str | None
    status: str
    end_ts: datetime | None
    source: str
    ingested_at: datetime = field(default_factory=utcnow)
    idempotency_key: str = ""
    raw: dict[str, Any] = field(default_factory=dict)
    open_interest: float = 0.0
    volume_24h: float = 0.0
    best_bid: float = 0.0
    best_ask: float = 0.0


@dataclass(slots=True)
class PriceBar:
    market_id: str
    token_id: str
    timestamp: datetime
    price: float
    source: str
    idempotency_key: str
    ingested_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class TradePrint:
    market_id: str
    token_id: str | None
    timestamp: datetime
    side: str | None
    price: float
    size: float
    source: str
    idempotency_key: str
    raw: dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class OutcomeLabel:
    market_id: str
    outcome_yes: float | None
    resolved_at: datetime | None
    status: str
    source: str
    idempotency_key: str
    raw: dict[str, Any] = field(default_factory=dict)
    ingested_at: datetime = field(default_factory=utcnow)


@dataclass(slots=True)
class BackfillRun:
    run_id: str
    since_ts: datetime
    until_ts: datetime
    sources: list[str]
    status: str
    markets_scanned: int = 0
    markets_ingested: int = 0
    price_rows: int = 0
    trade_rows: int = 0
    error_count: int = 0
    ingested_at: datetime = field(default_factory=utcnow)
    idempotency_key: str = ""
    started_at: datetime = field(default_factory=utcnow)
    finished_at: datetime | None = None
    note: str | None = None


@dataclass(slots=True)
class DualModelScore:
    settlement_prob: float
    intraday_prob: float
    blended_prob: float
    confidence: float
