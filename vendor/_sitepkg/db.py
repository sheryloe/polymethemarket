from __future__ import annotations

from pathlib import Path

from sqlalchemy import text
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from polymethemoney.models import Base


def _resolve_database_url(database_url: str) -> str:
    # Local script runs (outside docker) commonly use .env with host=postgres.
    # When not in container, switch to localhost so CLI tools can reach mapped ports.
    if Path("/.dockerenv").exists():
        return database_url
    try:
        parsed = make_url(database_url)
    except Exception:
        return database_url
    if parsed.host != "postgres":
        return database_url
    return str(parsed.set(host="127.0.0.1"))


def create_engine_and_session(database_url: str) -> tuple[AsyncEngine, async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(_resolve_database_url(database_url), pool_pre_ping=True)
    session_factory = async_sessionmaker(engine, expire_on_commit=False)
    return engine, session_factory


async def init_db(engine: AsyncEngine) -> None:
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        await conn.execute(
            text("ALTER TABLE markets ADD COLUMN IF NOT EXISTS token_ids_json JSON DEFAULT '[]'::json")
        )
        await conn.execute(
            text("ALTER TABLE markets ADD COLUMN IF NOT EXISTS outcomes_json JSON DEFAULT '[]'::json")
        )
        await conn.execute(
            text("ALTER TABLE features ADD COLUMN IF NOT EXISTS spread_pct DOUBLE PRECISION")
        )
        await conn.execute(
            text("ALTER TABLE features ADD COLUMN IF NOT EXISTS volume_oi_ratio DOUBLE PRECISION")
        )
        await conn.execute(
            text("ALTER TABLE features ADD COLUMN IF NOT EXISTS momentum_20 DOUBLE PRECISION")
        )
        await conn.execute(
            text("ALTER TABLE features ADD COLUMN IF NOT EXISTS zscore_20 DOUBLE PRECISION")
        )
        await conn.execute(
            text("ALTER TABLE features ADD COLUMN IF NOT EXISTS volatility_20 DOUBLE PRECISION")
        )
        await conn.execute(
            text("ALTER TABLE features ADD COLUMN IF NOT EXISTS volatility_30 DOUBLE PRECISION")
        )
        await conn.execute(
            text("ALTER TABLE signals ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(64) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE signals ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(32) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(64) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE orders ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(32) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE fills ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(64) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE fills ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(32) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(64) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(32) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS entry_fee_usd DOUBLE PRECISION DEFAULT 0.0")
        )
        await conn.execute(
            text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS exit_fee_usd DOUBLE PRECISION DEFAULT 0.0")
        )
        await conn.execute(
            text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS gross_realized_pnl_usd DOUBLE PRECISION DEFAULT 0.0")
        )
        await conn.execute(
            text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS close_reason VARCHAR(64)")
        )
        await conn.execute(
            text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS fallback_mark_close BOOLEAN DEFAULT FALSE")
        )
        await conn.execute(
            text("ALTER TABLE positions ADD COLUMN IF NOT EXISTS resolved_outcome_yes DOUBLE PRECISION")
        )
        await conn.execute(
            text("ALTER TABLE equity_curve ADD COLUMN IF NOT EXISTS strategy_id VARCHAR(64) DEFAULT 'legacy'")
        )
        await conn.execute(
            text("ALTER TABLE equity_curve ADD COLUMN IF NOT EXISTS execution_mode VARCHAR(32) DEFAULT 'legacy'")
        )
        await conn.execute(text("UPDATE signals SET strategy_id='legacy' WHERE strategy_id IS NULL"))
        await conn.execute(text("UPDATE signals SET execution_mode='legacy' WHERE execution_mode IS NULL"))
        await conn.execute(text("UPDATE orders SET strategy_id='legacy' WHERE strategy_id IS NULL"))
        await conn.execute(text("UPDATE orders SET execution_mode='legacy' WHERE execution_mode IS NULL"))
        await conn.execute(text("UPDATE fills SET strategy_id='legacy' WHERE strategy_id IS NULL"))
        await conn.execute(text("UPDATE fills SET execution_mode='legacy' WHERE execution_mode IS NULL"))
        await conn.execute(text("UPDATE positions SET strategy_id='legacy' WHERE strategy_id IS NULL"))
        await conn.execute(text("UPDATE positions SET execution_mode='legacy' WHERE execution_mode IS NULL"))
        await conn.execute(text("UPDATE equity_curve SET strategy_id='legacy' WHERE strategy_id IS NULL"))
        await conn.execute(text("UPDATE equity_curve SET execution_mode='legacy' WHERE execution_mode IS NULL"))
        await conn.execute(text("UPDATE positions SET entry_fee_usd=0.0 WHERE entry_fee_usd IS NULL"))
        await conn.execute(text("UPDATE positions SET exit_fee_usd=0.0 WHERE exit_fee_usd IS NULL"))
        await conn.execute(text("UPDATE positions SET gross_realized_pnl_usd=0.0 WHERE gross_realized_pnl_usd IS NULL"))
        await conn.execute(text("UPDATE positions SET fallback_mark_close=FALSE WHERE fallback_mark_close IS NULL"))
