from __future__ import annotations

from datetime import datetime

from sqlalchemy import JSON, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class MarketTickORM(Base):
    __tablename__ = "market_ticks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    bid: Mapped[float] = mapped_column(Float)
    ask: Mapped[float] = mapped_column(Float)
    last_price: Mapped[float] = mapped_column(Float)
    volume_1h: Mapped[float] = mapped_column(Float)
    open_interest: Mapped[float] = mapped_column(Float)
    expiry_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)


class FeatureORM(Base):
    __tablename__ = "features"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    implied_prob: Mapped[float] = mapped_column(Float)
    spread: Mapped[float] = mapped_column(Float)
    spread_pct: Mapped[float] = mapped_column(Float, default=0.0)
    volume_1h: Mapped[float] = mapped_column(Float)
    open_interest: Mapped[float] = mapped_column(Float)
    volume_oi_ratio: Mapped[float] = mapped_column(Float, default=0.0)
    time_to_expiry_hours: Mapped[float] = mapped_column(Float)
    orderbook_imbalance: Mapped[float] = mapped_column(Float)
    momentum_20: Mapped[float] = mapped_column(Float, default=0.0)
    zscore_20: Mapped[float] = mapped_column(Float, default=0.0)
    volatility_20: Mapped[float] = mapped_column(Float, default=0.0)
    volatility_30: Mapped[float] = mapped_column(Float)
    target: Mapped[float | None] = mapped_column(Float, nullable=True)


class SignalORM(Base):
    __tablename__ = "signals"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    side: Mapped[str] = mapped_column(String(8))
    fair_prob: Mapped[float] = mapped_column(Float)
    implied_prob: Mapped[float] = mapped_column(Float)
    edge: Mapped[float] = mapped_column(Float)
    net_ev: Mapped[float] = mapped_column(Float)
    score: Mapped[int] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32), default="generated")
    detail_json: Mapped[dict] = mapped_column(JSON, default=dict)


class OrderORM(Base):
    __tablename__ = "orders"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    signal_id: Mapped[str] = mapped_column(String(64), index=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    side: Mapped[str] = mapped_column(String(8))
    price: Mapped[float] = mapped_column(Float)
    size_usd: Mapped[float] = mapped_column(Float)
    mode: Mapped[str] = mapped_column(String(16))
    trading_mode: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(32), default="new")
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class FillORM(Base):
    __tablename__ = "fills"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    order_id: Mapped[str] = mapped_column(String(64), index=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    side: Mapped[str] = mapped_column(String(8))
    fill_price: Mapped[float] = mapped_column(Float)
    size_usd: Mapped[float] = mapped_column(Float)
    fee_usd: Mapped[float] = mapped_column(Float)
    pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)
    trading_mode: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class PositionORM(Base):
    __tablename__ = "positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    side: Mapped[str] = mapped_column(String(8))
    entry_price: Mapped[float] = mapped_column(Float)
    size_usd: Mapped[float] = mapped_column(Float)
    trading_mode: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="open")
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    realized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)


class EquityCurveORM(Base):
    __tablename__ = "equity_curve"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    equity_usd: Mapped[float] = mapped_column(Float)
    daily_drawdown_pct: Mapped[float] = mapped_column(Float)
    weekly_drawdown_pct: Mapped[float] = mapped_column(Float)


class RiskEventORM(Base):
    __tablename__ = "risk_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    event_type: Mapped[str] = mapped_column(String(64), index=True)
    message: Mapped[str] = mapped_column(Text)


class ModelRegistryORM(Base):
    __tablename__ = "model_registry"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    version: Mapped[str] = mapped_column(String(64), index=True)
    kind: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(16))
    metrics_json: Mapped[dict] = mapped_column(JSON, default=dict)
    is_champion: Mapped[bool] = mapped_column(default=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)


class MarketMetaORM(Base):
    __tablename__ = "markets"

    market_id: Mapped[str] = mapped_column(String(128), primary_key=True)
    condition_id: Mapped[str] = mapped_column(String(128), index=True)
    token_id: Mapped[str] = mapped_column(String(128), index=True)
    token_ids_json: Mapped[list] = mapped_column(JSON, default=list)
    outcomes_json: Mapped[list] = mapped_column(JSON, default=list)
    question: Mapped[str] = mapped_column(Text)
    category: Mapped[str | None] = mapped_column(String(128), nullable=True)
    event_slug: Mapped[str | None] = mapped_column(String(255), nullable=True)
    open_interest: Mapped[float] = mapped_column(Float, default=0.0)
    volume_24h: Mapped[float] = mapped_column(Float, default=0.0)
    best_bid: Mapped[float] = mapped_column(Float, default=0.0)
    best_ask: Mapped[float] = mapped_column(Float, default=0.0)
    status: Mapped[str] = mapped_column(String(32), index=True)
    end_ts: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)


class StructureBundleORM(Base):
    __tablename__ = "structure_bundles"

    opportunity_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    event_id: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    status: Mapped[str] = mapped_column(String(24), index=True, default="detected")
    legs_count: Mapped[int] = mapped_column(Integer)
    target_payout_usd: Mapped[float] = mapped_column(Float)
    gross_edge_usd: Mapped[float] = mapped_column(Float)
    net_edge_usd: Mapped[float] = mapped_column(Float)
    detected_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    rolled_back: Mapped[bool] = mapped_column(default=False)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)


class StructurePositionORM(Base):
    __tablename__ = "structure_positions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    opportunity_id: Mapped[str] = mapped_column(String(80), index=True)
    event_id: Mapped[str] = mapped_column(String(255), index=True)
    kind: Mapped[str] = mapped_column(String(16), index=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    token_id: Mapped[str] = mapped_column(String(128), index=True)
    outcome_idx: Mapped[int] = mapped_column(Integer)
    outcome_name: Mapped[str] = mapped_column(String(255))
    entry_price: Mapped[float] = mapped_column(Float)
    size_shares: Mapped[float] = mapped_column(Float)
    size_usd: Mapped[float] = mapped_column(Float)
    status: Mapped[str] = mapped_column(String(16), index=True, default="open")
    opened_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    exit_price: Mapped[float | None] = mapped_column(Float, nullable=True)
    realized_pnl_usd: Mapped[float] = mapped_column(Float, default=0.0)


class PriceHistoryORM(Base):
    __tablename__ = "price_history"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    token_id: Mapped[str] = mapped_column(String(128), index=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    price: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(180), unique=True, index=True)


class TradePrintORM(Base):
    __tablename__ = "trades"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), index=True)
    token_id: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    side: Mapped[str | None] = mapped_column(String(8), nullable=True)
    price: Mapped[float] = mapped_column(Float)
    size: Mapped[float] = mapped_column(Float)
    source: Mapped[str] = mapped_column(String(32), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(180), unique=True, index=True)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)


class OutcomeORM(Base):
    __tablename__ = "outcomes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    market_id: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    outcome_yes: Mapped[float | None] = mapped_column(Float, nullable=True)
    resolved_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    status: Mapped[str] = mapped_column(String(32), index=True)
    source: Mapped[str] = mapped_column(String(32), index=True)
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    raw_json: Mapped[dict] = mapped_column(JSON, default=dict)


class BackfillRunORM(Base):
    __tablename__ = "backfill_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    run_id: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    since_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    until_ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    sources_json: Mapped[dict] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(32), index=True)
    markets_scanned: Mapped[int] = mapped_column(Integer, default=0)
    markets_ingested: Mapped[int] = mapped_column(Integer, default=0)
    price_rows: Mapped[int] = mapped_column(Integer, default=0)
    trade_rows: Mapped[int] = mapped_column(Integer, default=0)
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    source: Mapped[str] = mapped_column(String(32), index=True, default="system")
    ingested_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    idempotency_key: Mapped[str] = mapped_column(String(160), unique=True, index=True)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), index=True, nullable=True)
    note: Mapped[str | None] = mapped_column(Text, nullable=True)
