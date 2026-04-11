from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone

from sqlalchemy import and_, func, select, text, update
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from polymethemoney.domain import (
    STRATEGY_LEGACY,
    FeatureVector,
    MarketTick,
    OrderIntent,
    RiskState,
    Signal,
    TradingMode,
)
from polymethemoney.models import (
    BackfillRunORM,
    EquityCurveORM,
    FeatureORM,
    FillORM,
    MarketMetaORM,
    MarketTickORM,
    ModelRegistryORM,
    OutcomeORM,
    OrderORM,
    PriceHistoryORM,
    PositionORM,
    RiskEventORM,
    SignalORM,
    StructureBundleORM,
    StructurePositionORM,
    TradePrintORM,
)


class Store:
    def __init__(self, session_factory: async_sessionmaker[AsyncSession], starting_capital_usd: float) -> None:
        self.session_factory = session_factory
        self.starting_capital_usd = starting_capital_usd

    @staticmethod
    def _clip_contract_price(price: float) -> float:
        return max(0.001, min(0.999, float(price)))

    @staticmethod
    def estimate_position_unrealized(
        *,
        size_usd: float,
        entry_price: float,
        mark_price: float,
        entry_fee_usd: float = 0.0,
        exit_fee_bps: float = 0.0,
    ) -> dict[str, float]:
        clipped_entry = max(float(entry_price), 0.001)
        shares = float(size_usd) / clipped_entry
        clipped_mark = Store._clip_contract_price(mark_price)
        exit_value = shares * clipped_mark
        exit_fee_usd = max(0.0, exit_value * max(0.0, float(exit_fee_bps)) / 10000.0)
        gross_unrealized = exit_value - float(size_usd)
        net_unrealized = gross_unrealized - float(entry_fee_usd) - exit_fee_usd
        return {
            "shares": shares,
            "mark_price": clipped_mark,
            "exit_value": exit_value,
            "entry_fee_usd": float(entry_fee_usd),
            "exit_fee_usd": exit_fee_usd,
            "gross_unrealized_pnl_usd": gross_unrealized,
            "net_unrealized_pnl_usd": net_unrealized,
        }

    @staticmethod
    def estimate_position_settlement(
        *,
        size_usd: float,
        entry_price: float,
        settled_price: float,
        entry_fee_usd: float = 0.0,
    ) -> dict[str, float]:
        clipped_entry = max(float(entry_price), 0.001)
        shares = float(size_usd) / clipped_entry
        terminal_price = max(0.0, min(1.0, float(settled_price)))
        exit_value = shares * terminal_price
        gross_realized = exit_value - float(size_usd)
        net_realized = gross_realized - float(entry_fee_usd)
        return {
            "shares": shares,
            "settled_price": terminal_price,
            "exit_value": exit_value,
            "entry_fee_usd": float(entry_fee_usd),
            "exit_fee_usd": 0.0,
            "gross_realized_pnl_usd": gross_realized,
            "net_realized_pnl_usd": net_realized,
        }

    async def add_market_tick(self, tick: MarketTick) -> None:
        async with self.session_factory() as session:
            row = MarketTickORM(
                market_id=tick.market_id,
                timestamp=tick.timestamp,
                bid=tick.bid,
                ask=tick.ask,
                last_price=tick.last_price,
                volume_1h=tick.volume_1h,
                open_interest=tick.open_interest,
                expiry_ts=tick.expiry_ts,
                raw_json=tick.raw,
            )
            session.add(row)
            await session.commit()

    async def add_feature(self, fv: FeatureVector) -> None:
        async with self.session_factory() as session:
            row = FeatureORM(
                market_id=fv.market_id,
                timestamp=fv.timestamp,
                implied_prob=fv.implied_prob,
                spread=fv.spread,
                spread_pct=fv.spread_pct,
                volume_1h=fv.volume_1h,
                open_interest=fv.open_interest,
                volume_oi_ratio=fv.volume_oi_ratio,
                time_to_expiry_hours=fv.time_to_expiry_hours,
                orderbook_imbalance=fv.orderbook_imbalance,
                momentum_20=fv.momentum_20,
                zscore_20=fv.zscore_20,
                volatility_20=fv.volatility_20,
                volatility_30=fv.volatility_30,
            )
            session.add(row)
            await session.commit()

    async def add_signal(self, signal: Signal) -> None:
        async with self.session_factory() as session:
            row = SignalORM(
                id=signal.signal_id,
                strategy_id=signal.strategy_id,
                execution_mode=str(getattr(signal, "execution_mode", STRATEGY_LEGACY) or STRATEGY_LEGACY),
                market_id=signal.market_id,
                created_at=signal.created_at,
                side=signal.side.value,
                fair_prob=signal.fair_prob,
                implied_prob=signal.implied_prob,
                edge=signal.edge,
                net_ev=signal.net_ev,
                score=signal.score,
                detail_json={
                    "liquidity_score": signal.liquidity_score,
                    "model_edge_score": signal.model_edge_score,
                    "regime_score": signal.regime_score,
                    "model_confidence": signal.model_confidence,
                    "effective_edge": signal.effective_edge,
                    "ttl_seconds": signal.ttl_seconds,
                },
            )
            session.add(row)
            await session.commit()

    async def update_signal_status(self, signal_id: str, status: str) -> None:
        async with self.session_factory() as session:
            stmt = update(SignalORM).where(SignalORM.id == signal_id).values(status=status)
            await session.execute(stmt)
            await session.commit()

    async def add_order(self, order_id: str, intent: OrderIntent, trading_mode: TradingMode, status: str) -> None:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            row = OrderORM(
                id=order_id,
                strategy_id=intent.strategy_id,
                execution_mode=str(getattr(intent, "execution_mode", STRATEGY_LEGACY) or STRATEGY_LEGACY),
                signal_id=intent.signal_id,
                market_id=intent.market_id,
                side=intent.side.value,
                price=intent.price,
                size_usd=intent.size_usd,
                mode=intent.mode.value,
                trading_mode=trading_mode.value,
                status=status,
                created_at=now,
                updated_at=now,
            )
            session.add(row)
            await session.commit()

    async def update_order(self, order_id: str, status: str, error: str | None = None) -> None:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            stmt = (
                update(OrderORM)
                .where(OrderORM.id == order_id)
                .values(status=status, error=error, updated_at=now)
            )
            await session.execute(stmt)
            await session.commit()

    async def add_fill(
        self,
        order_id: str,
        market_id: str,
        side: str,
        fill_price: float,
        size_usd: float,
        fee_usd: float,
        pnl_usd: float,
        trading_mode: TradingMode,
        strategy_id: str = STRATEGY_LEGACY,
        execution_mode: str = STRATEGY_LEGACY,
    ) -> None:
        async with self.session_factory() as session:
            fill = FillORM(
                strategy_id=strategy_id,
                execution_mode=execution_mode,
                order_id=order_id,
                market_id=market_id,
                side=side,
                fill_price=fill_price,
                size_usd=size_usd,
                fee_usd=fee_usd,
                pnl_usd=pnl_usd,
                trading_mode=trading_mode.value,
                created_at=datetime.now(timezone.utc),
            )
            session.add(fill)
            await session.commit()

    async def open_position(
        self,
        market_id: str,
        side: str,
        entry_price: float,
        size_usd: float,
        mode: TradingMode,
        entry_fee_usd: float = 0.0,
        strategy_id: str = STRATEGY_LEGACY,
        execution_mode: str = STRATEGY_LEGACY,
    ) -> None:
        async with self.session_factory() as session:
            row = PositionORM(
                strategy_id=strategy_id,
                execution_mode=execution_mode,
                market_id=market_id,
                side=side,
                entry_price=entry_price,
                size_usd=size_usd,
                entry_fee_usd=max(0.0, float(entry_fee_usd)),
                exit_fee_usd=0.0,
                trading_mode=mode.value,
                status="open",
                opened_at=datetime.now(timezone.utc),
                gross_realized_pnl_usd=0.0,
                realized_pnl_usd=0.0,
                close_reason=None,
                fallback_mark_close=False,
                resolved_outcome_yes=None,
            )
            session.add(row)
            await session.commit()

    async def get_open_positions(
        self,
        trading_mode: TradingMode | None = None,
        strategy_id: str | None = None,
        execution_mode: str | None = None,
    ) -> list[PositionORM]:
        async with self.session_factory() as session:
            stmt = select(PositionORM).where(PositionORM.status == "open")
            if trading_mode is not None:
                stmt = stmt.where(PositionORM.trading_mode == trading_mode.value)
            if strategy_id is not None:
                stmt = stmt.where(PositionORM.strategy_id == strategy_id)
            if execution_mode is not None:
                stmt = stmt.where(PositionORM.execution_mode == execution_mode)
            result = await session.execute(stmt)
            return list(result.scalars().all())

    async def close_positions_for_market(
        self,
        market_id: str,
        opposite_side: str,
        exit_price: float,
        mode: TradingMode,
        exit_fee_bps: float = 0.0,
        strategy_id: str | None = None,
    ) -> list[tuple[int, float]]:
        now = datetime.now(timezone.utc)
        closed: list[tuple[int, float]] = []
        async with self.session_factory() as session:
            stmt = select(PositionORM).where(
                and_(
                    PositionORM.market_id == market_id,
                    PositionORM.side == opposite_side,
                    PositionORM.status == "open",
                    PositionORM.trading_mode == mode.value,
                )
            )
            if strategy_id is not None:
                stmt = stmt.where(PositionORM.strategy_id == strategy_id)
            rows = list((await session.execute(stmt)).scalars().all())
            for row in rows:
                mtm = self.estimate_position_unrealized(
                    size_usd=float(row.size_usd),
                    entry_price=float(row.entry_price),
                    mark_price=exit_price,
                    entry_fee_usd=float(getattr(row, "entry_fee_usd", 0.0) or 0.0),
                    exit_fee_bps=exit_fee_bps,
                )
                gross_pnl = float(mtm["gross_unrealized_pnl_usd"])
                pnl = float(mtm["net_unrealized_pnl_usd"])
                row.status = "closed"
                row.closed_at = now
                row.exit_fee_usd = float(mtm["exit_fee_usd"])
                row.gross_realized_pnl_usd = gross_pnl
                row.realized_pnl_usd = pnl
                closed.append((row.id, pnl))
            await session.commit()
        return closed

    async def close_all_open_positions(self, mode: TradingMode, strategy_id: str | None = None) -> int:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            filters = [PositionORM.status == "open", PositionORM.trading_mode == mode.value]
            if strategy_id is not None:
                filters.append(PositionORM.strategy_id == strategy_id)
            stmt = update(PositionORM).where(and_(*filters)).values(status="closed", closed_at=now)
            result = await session.execute(stmt)
            await session.commit()
            return int(result.rowcount or 0)

    async def reset_paper_open_positions(self, strategy_ids: list[str] | None = None) -> dict[str, int]:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            filters = [PositionORM.status == "open", PositionORM.trading_mode == TradingMode.PAPER.value]
            if strategy_ids:
                filters.append(PositionORM.strategy_id.in_(strategy_ids))
            pos_stmt = update(PositionORM).where(and_(*filters)).values(
                status="closed",
                closed_at=now,
                entry_fee_usd=0.0,
                exit_fee_usd=0.0,
                gross_realized_pnl_usd=0.0,
                realized_pnl_usd=0.0,
                close_reason="reset_paper",
                fallback_mark_close=False,
                resolved_outcome_yes=None,
            )
            pos_result = await session.execute(pos_stmt)

            structure_leg_stmt = (
                update(StructurePositionORM)
                .where(StructurePositionORM.status == "open")
                .values(
                    status="closed",
                    closed_at=now,
                    exit_price=StructurePositionORM.entry_price,
                    realized_pnl_usd=0.0,
                )
            )
            structure_leg_result = await session.execute(structure_leg_stmt)

            structure_bundle_stmt = (
                update(StructureBundleORM)
                .where(StructureBundleORM.status == "open")
                .values(status="closed", closed_at=now, rolled_back=False, error="demo_start_reset")
            )
            structure_bundle_result = await session.execute(structure_bundle_stmt)

            await session.commit()
            return {
                "positions_closed": int(pos_result.rowcount or 0),
                "structure_legs_closed": int(structure_leg_result.rowcount or 0),
                "structure_bundles_closed": int(structure_bundle_result.rowcount or 0),
            }

    async def close_position_at_mark(
        self,
        position_id: int,
        mark_price: float,
        mode: TradingMode,
        exit_fee_bps: float = 0.0,
        strategy_id: str | None = None,
        close_reason: str = "mark_close",
        fallback_mark_close: bool = False,
    ) -> dict | None:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            filters = [
                PositionORM.id == position_id,
                PositionORM.status == "open",
                PositionORM.trading_mode == mode.value,
            ]
            if strategy_id is not None:
                filters.append(PositionORM.strategy_id == strategy_id)
            stmt = select(PositionORM).where(and_(*filters))
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            mtm = self.estimate_position_unrealized(
                size_usd=float(row.size_usd),
                entry_price=float(row.entry_price),
                mark_price=mark_price,
                entry_fee_usd=float(getattr(row, "entry_fee_usd", 0.0) or 0.0),
                exit_fee_bps=exit_fee_bps,
            )
            gross_pnl = float(mtm["gross_unrealized_pnl_usd"])
            pnl = float(mtm["net_unrealized_pnl_usd"])
            row.status = "closed"
            row.closed_at = now
            row.exit_fee_usd = float(mtm["exit_fee_usd"])
            row.gross_realized_pnl_usd = gross_pnl
            row.realized_pnl_usd = pnl
            row.close_reason = close_reason
            row.fallback_mark_close = bool(fallback_mark_close)
            row.resolved_outcome_yes = None
            await session.commit()
            return {
                "position_id": int(row.id),
                "strategy_id": str(row.strategy_id),
                "execution_mode": str(getattr(row, "execution_mode", STRATEGY_LEGACY) or STRATEGY_LEGACY),
                "market_id": str(row.market_id),
                "side": str(row.side),
                "size_usd": float(row.size_usd),
                "entry_price": float(row.entry_price),
                "entry_fee_usd": float(getattr(row, "entry_fee_usd", 0.0) or 0.0),
                "exit_fee_usd": float(mtm["exit_fee_usd"]),
                "gross_realized_pnl_usd": gross_pnl,
                "exit_price": float(mtm["mark_price"]),
                "realized_pnl_usd": float(pnl),
                "close_reason": close_reason,
                "fallback_mark_close": bool(fallback_mark_close),
            }

    async def close_position_at_settlement(
        self,
        position_id: int,
        outcome_yes: float,
        mode: TradingMode,
        strategy_id: str | None = None,
        close_reason: str = "resolved_outcome",
    ) -> dict | None:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            filters = [
                PositionORM.id == position_id,
                PositionORM.status == "open",
                PositionORM.trading_mode == mode.value,
            ]
            if strategy_id is not None:
                filters.append(PositionORM.strategy_id == strategy_id)
            stmt = select(PositionORM).where(and_(*filters))
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            terminal_yes = max(0.0, min(1.0, float(outcome_yes)))
            settled_price = terminal_yes if str(row.side).upper() == "YES" else (1.0 - terminal_yes)
            settlement = self.estimate_position_settlement(
                size_usd=float(row.size_usd),
                entry_price=float(row.entry_price),
                settled_price=settled_price,
                entry_fee_usd=float(getattr(row, "entry_fee_usd", 0.0) or 0.0),
            )
            row.status = "closed"
            row.closed_at = now
            row.exit_fee_usd = 0.0
            row.gross_realized_pnl_usd = float(settlement["gross_realized_pnl_usd"])
            row.realized_pnl_usd = float(settlement["net_realized_pnl_usd"])
            row.close_reason = close_reason
            row.fallback_mark_close = False
            row.resolved_outcome_yes = terminal_yes
            await session.commit()
            return {
                "position_id": int(row.id),
                "strategy_id": str(row.strategy_id),
                "execution_mode": str(getattr(row, "execution_mode", STRATEGY_LEGACY) or STRATEGY_LEGACY),
                "market_id": str(row.market_id),
                "side": str(row.side),
                "size_usd": float(row.size_usd),
                "entry_price": float(row.entry_price),
                "entry_fee_usd": float(getattr(row, "entry_fee_usd", 0.0) or 0.0),
                "exit_fee_usd": 0.0,
                "gross_realized_pnl_usd": float(settlement["gross_realized_pnl_usd"]),
                "exit_price": float(settlement["settled_price"]),
                "realized_pnl_usd": float(settlement["net_realized_pnl_usd"]),
                "close_reason": close_reason,
                "fallback_mark_close": False,
                "resolved_outcome_yes": terminal_yes,
            }

    async def record_equity_curve(
        self,
        state: RiskState,
        equity_usd: float,
        strategy_id: str = STRATEGY_LEGACY,
        execution_mode: str = STRATEGY_LEGACY,
    ) -> None:
        async with self.session_factory() as session:
            row = EquityCurveORM(
                strategy_id=strategy_id,
                execution_mode=execution_mode,
                timestamp=datetime.now(timezone.utc),
                equity_usd=equity_usd,
                daily_drawdown_pct=state.daily_drawdown_pct,
                weekly_drawdown_pct=state.weekly_drawdown_pct,
            )
            session.add(row)
            await session.commit()

    async def add_risk_event(self, event_type: str, message: str) -> None:
        async with self.session_factory() as session:
            row = RiskEventORM(timestamp=datetime.now(timezone.utc), event_type=event_type, message=message)
            session.add(row)
            await session.commit()

    async def record_model_result(self, version: str, kind: str, status: str, metrics: dict, is_champion: bool) -> None:
        async with self.session_factory() as session:
            if is_champion:
                await session.execute(
                    update(ModelRegistryORM).where(ModelRegistryORM.kind == kind).values(is_champion=False)
                )
            row = ModelRegistryORM(
                version=version,
                kind=kind,
                status=status,
                metrics_json=metrics,
                is_champion=is_champion,
                created_at=datetime.now(timezone.utc),
            )
            session.add(row)
            await session.commit()

    async def upsert_market_metadata(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        async with self.session_factory() as session:
            stmt = insert(MarketMetaORM).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[MarketMetaORM.market_id],
                set_={
                    "condition_id": stmt.excluded.condition_id,
                    "token_id": stmt.excluded.token_id,
                    "token_ids_json": stmt.excluded.token_ids_json,
                    "outcomes_json": stmt.excluded.outcomes_json,
                    "question": stmt.excluded.question,
                    "category": stmt.excluded.category,
                    "event_slug": stmt.excluded.event_slug,
                    "open_interest": stmt.excluded.open_interest,
                    "volume_24h": stmt.excluded.volume_24h,
                    "best_bid": stmt.excluded.best_bid,
                    "best_ask": stmt.excluded.best_ask,
                    "status": stmt.excluded.status,
                    "end_ts": stmt.excluded.end_ts,
                    "source": stmt.excluded.source,
                    "ingested_at": stmt.excluded.ingested_at,
                    "idempotency_key": stmt.excluded.idempotency_key,
                    "raw_json": stmt.excluded.raw_json,
                },
            )
            await session.execute(stmt)
            await session.commit()
            return len(rows)

    async def add_price_history_rows(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        batch_size = 1000
        inserted = 0
        async with self.session_factory() as session:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                stmt = insert(PriceHistoryORM).values(batch)
                stmt = stmt.on_conflict_do_nothing(index_elements=[PriceHistoryORM.idempotency_key])
                result = await session.execute(stmt)
                inserted += int(result.rowcount or 0)
            await session.commit()
            return inserted

    async def add_trade_rows(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        batch_size = 1000
        inserted = 0
        async with self.session_factory() as session:
            for start in range(0, len(rows), batch_size):
                batch = rows[start : start + batch_size]
                stmt = insert(TradePrintORM).values(batch)
                stmt = stmt.on_conflict_do_nothing(index_elements=[TradePrintORM.idempotency_key])
                result = await session.execute(stmt)
                inserted += int(result.rowcount or 0)
            await session.commit()
            return inserted

    async def upsert_outcomes(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        async with self.session_factory() as session:
            stmt = insert(OutcomeORM).values(rows)
            stmt = stmt.on_conflict_do_update(
                index_elements=[OutcomeORM.market_id],
                set_={
                    "outcome_yes": stmt.excluded.outcome_yes,
                    "resolved_at": stmt.excluded.resolved_at,
                    "status": stmt.excluded.status,
                    "source": stmt.excluded.source,
                    "ingested_at": stmt.excluded.ingested_at,
                    "idempotency_key": stmt.excluded.idempotency_key,
                    "raw_json": stmt.excluded.raw_json,
                },
            )
            await session.execute(stmt)
            await session.commit()
            return len(rows)

    async def start_backfill_run(
        self,
        run_id: str,
        since_ts: datetime,
        until_ts: datetime,
        sources: list[str],
    ) -> None:
        now = datetime.now(timezone.utc)
        payload = {
            "run_id": run_id,
            "since_ts": since_ts,
            "until_ts": until_ts,
            "sources_json": {"sources": sources},
            "status": "running",
            "markets_scanned": 0,
            "markets_ingested": 0,
            "price_rows": 0,
            "trade_rows": 0,
            "error_count": 0,
            "source": "system",
            "ingested_at": now,
            "idempotency_key": f"backfill:{run_id}",
            "started_at": now,
            "finished_at": None,
            "note": None,
        }
        async with self.session_factory() as session:
            stmt = insert(BackfillRunORM).values(payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[BackfillRunORM.run_id],
                set_={
                    "since_ts": stmt.excluded.since_ts,
                    "until_ts": stmt.excluded.until_ts,
                    "sources_json": stmt.excluded.sources_json,
                    "status": stmt.excluded.status,
                    "ingested_at": stmt.excluded.ingested_at,
                    "started_at": stmt.excluded.started_at,
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def finish_backfill_run(
        self,
        run_id: str,
        status: str,
        markets_scanned: int,
        markets_ingested: int,
        price_rows: int,
        trade_rows: int,
        error_count: int,
        note: str | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            stmt = (
                update(BackfillRunORM)
                .where(BackfillRunORM.run_id == run_id)
                .values(
                    status=status,
                    markets_scanned=markets_scanned,
                    markets_ingested=markets_ingested,
                    price_rows=price_rows,
                    trade_rows=trade_rows,
                    error_count=error_count,
                    note=note,
                    finished_at=now,
                    ingested_at=now,
                )
            )
            await session.execute(stmt)
            await session.commit()

    async def latest_backfill_run(self) -> dict | None:
        async with self.session_factory() as session:
            stmt = select(BackfillRunORM).order_by(BackfillRunORM.started_at.desc()).limit(1)
            row = (await session.execute(stmt)).scalar_one_or_none()
            if row is None:
                return None
            return {
                "run_id": row.run_id,
                "since_ts": row.since_ts,
                "until_ts": row.until_ts,
                "status": row.status,
                "markets_scanned": row.markets_scanned,
                "markets_ingested": row.markets_ingested,
                "price_rows": row.price_rows,
                "trade_rows": row.trade_rows,
                "error_count": row.error_count,
                "started_at": row.started_at,
                "finished_at": row.finished_at,
            }

    async def latest_model_versions(self) -> dict[str, str]:
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(ModelRegistryORM.kind, ModelRegistryORM.version)
                    .where(ModelRegistryORM.status == "ok")
                    .order_by(ModelRegistryORM.created_at.desc())
                )
            ).all()
            versions: dict[str, str] = {}
            for kind, version in rows:
                if kind not in versions:
                    versions[str(kind)] = str(version)
            return versions

    async def latest_training_baseline(self) -> dict[str, str]:
        async with self.session_factory() as session:
            rows = (
                await session.execute(
                    select(ModelRegistryORM.kind, ModelRegistryORM.metrics_json)
                    .where(ModelRegistryORM.status == "ok")
                    .order_by(ModelRegistryORM.created_at.desc())
                )
            ).all()
            baseline: dict[str, str] = {}
            for kind, metrics in rows:
                key = str(kind)
                if key in baseline:
                    continue
                if isinstance(metrics, dict) and isinstance(metrics.get("train_max_ts"), str):
                    baseline[key] = str(metrics["train_max_ts"])
            return baseline

    async def list_markets_for_training(self, since: datetime) -> list[dict]:
        async with self.session_factory() as session:
            stmt = (
                select(
                    MarketMetaORM.market_id,
                    MarketMetaORM.token_id,
                    MarketMetaORM.question,
                    MarketMetaORM.category,
                    MarketMetaORM.event_slug,
                    MarketMetaORM.open_interest,
                    MarketMetaORM.volume_24h,
                    MarketMetaORM.best_bid,
                    MarketMetaORM.best_ask,
                    MarketMetaORM.end_ts,
                    MarketMetaORM.status,
                    OutcomeORM.outcome_yes,
                    OutcomeORM.resolved_at,
                    OutcomeORM.status.label("outcome_status"),
                )
                .join(OutcomeORM, OutcomeORM.market_id == MarketMetaORM.market_id, isouter=True)
                .where(
                    and_(
                        (MarketMetaORM.end_ts.is_(None) | (MarketMetaORM.end_ts >= since)),
                    )
                )
            )
            rows = (await session.execute(stmt)).mappings().all()
            return [dict(row) for row in rows]

    async def get_price_history(self, market_id: str, since: datetime) -> list[dict]:
        async with self.session_factory() as session:
            stmt = (
                select(PriceHistoryORM.timestamp, PriceHistoryORM.price, PriceHistoryORM.token_id)
                .where(and_(PriceHistoryORM.market_id == market_id, PriceHistoryORM.timestamp >= since))
                .order_by(PriceHistoryORM.timestamp.asc())
            )
            rows = (await session.execute(stmt)).all()
            return [
                {"timestamp": row[0], "price": float(row[1]), "token_id": row[2]}
                for row in rows
            ]

    async def get_trades(self, market_id: str, since: datetime) -> list[dict]:
        async with self.session_factory() as session:
            stmt = (
                select(TradePrintORM.timestamp, TradePrintORM.side, TradePrintORM.price, TradePrintORM.size)
                .where(and_(TradePrintORM.market_id == market_id, TradePrintORM.timestamp >= since))
                .order_by(TradePrintORM.timestamp.asc())
            )
            rows = (await session.execute(stmt)).all()
            return [
                {
                    "timestamp": row[0],
                    "side": row[1],
                    "price": float(row[2]),
                    "size": float(row[3]),
                }
                for row in rows
            ]

    async def latest_market_prices(self, market_ids: list[str]) -> dict[str, float]:
        if not market_ids:
            return {}
        sql = text(
            """
            WITH latest AS (
                SELECT market_id, MAX(timestamp) AS max_ts
                FROM market_ticks
                WHERE market_id = ANY(:market_ids)
                GROUP BY market_id
            )
            SELECT t.market_id, t.last_price
            FROM market_ticks t
            JOIN latest l
              ON l.market_id = t.market_id
             AND l.max_ts = t.timestamp
            """
        )
        async with self.session_factory() as session:
            rows = (await session.execute(sql, {"market_ids": market_ids})).all()
            return {str(row[0]): float(row[1]) for row in rows}

    async def latest_market_expiries(self, market_ids: list[str]) -> dict[str, datetime | None]:
        if not market_ids:
            return {}
        sql = text(
            """
            WITH latest AS (
                SELECT market_id, MAX(timestamp) AS max_ts
                FROM market_ticks
                WHERE market_id = ANY(:market_ids)
                GROUP BY market_id
            )
            SELECT t.market_id, t.expiry_ts
            FROM market_ticks t
            JOIN latest l
              ON l.market_id = t.market_id
             AND l.max_ts = t.timestamp
            """
        )
        async with self.session_factory() as session:
            rows = (await session.execute(sql, {"market_ids": market_ids})).all()
            return {str(row[0]): row[1] for row in rows}

    async def latest_market_outcomes(
        self,
        market_ids: list[str],
    ) -> dict[str, dict[str, object]]:
        if not market_ids:
            return {}
        async with self.session_factory() as session:
            stmt = select(
                OutcomeORM.market_id,
                OutcomeORM.outcome_yes,
                OutcomeORM.resolved_at,
                OutcomeORM.status,
            ).where(OutcomeORM.market_id.in_(market_ids))
            rows = (await session.execute(stmt)).all()
            return {
                str(row[0]): {
                    "outcome_yes": None if row[1] is None else float(row[1]),
                    "resolved_at": row[2],
                    "status": str(row[3] or ""),
                }
                for row in rows
            }

    async def market_questions(self, market_ids: list[str]) -> dict[str, str]:
        if not market_ids:
            return {}
        async with self.session_factory() as session:
            stmt = select(MarketMetaORM.market_id, MarketMetaORM.question).where(MarketMetaORM.market_id.in_(market_ids))
            rows = (await session.execute(stmt)).all()
            return {
                str(row[0]): str(row[1] or "")
                for row in rows
            }

    async def upsert_structure_bundle(self, payload: dict) -> None:
        async with self.session_factory() as session:
            stmt = insert(StructureBundleORM).values(payload)
            stmt = stmt.on_conflict_do_update(
                index_elements=[StructureBundleORM.opportunity_id],
                set_={
                    "event_id": stmt.excluded.event_id,
                    "kind": stmt.excluded.kind,
                    "status": stmt.excluded.status,
                    "legs_count": stmt.excluded.legs_count,
                    "target_payout_usd": stmt.excluded.target_payout_usd,
                    "gross_edge_usd": stmt.excluded.gross_edge_usd,
                    "net_edge_usd": stmt.excluded.net_edge_usd,
                    "detected_at": stmt.excluded.detected_at,
                    "opened_at": stmt.excluded.opened_at,
                    "closed_at": stmt.excluded.closed_at,
                    "rolled_back": stmt.excluded.rolled_back,
                    "error": stmt.excluded.error,
                },
            )
            await session.execute(stmt)
            await session.commit()

    async def add_structure_positions(self, rows: list[dict]) -> int:
        if not rows:
            return 0
        async with self.session_factory() as session:
            session.add_all([StructurePositionORM(**row) for row in rows])
            await session.commit()
            return len(rows)

    async def open_structure_bundles(self) -> list[StructureBundleORM]:
        async with self.session_factory() as session:
            stmt = select(StructureBundleORM).where(StructureBundleORM.status == "open").order_by(StructureBundleORM.opened_at.asc())
            rows = (await session.execute(stmt)).scalars().all()
            return list(rows)

    async def open_structure_positions(self, opportunity_id: str | None = None) -> list[StructurePositionORM]:
        async with self.session_factory() as session:
            stmt = select(StructurePositionORM).where(StructurePositionORM.status == "open")
            if opportunity_id is not None:
                stmt = stmt.where(StructurePositionORM.opportunity_id == opportunity_id)
            rows = (await session.execute(stmt)).scalars().all()
            return list(rows)

    async def structure_open_legs_count(self) -> int:
        async with self.session_factory() as session:
            stmt = select(func.count(StructurePositionORM.id)).where(StructurePositionORM.status == "open")
            value = await session.scalar(stmt)
            return int(value or 0)

    async def structure_open_notional(self) -> float:
        async with self.session_factory() as session:
            stmt = select(func.coalesce(func.sum(StructurePositionORM.size_usd), 0.0)).where(StructurePositionORM.status == "open")
            value = await session.scalar(stmt)
            return float(value or 0.0)

    async def close_structure_bundle(
        self,
        opportunity_id: str,
        exit_prices: dict[str, float],
        status: str = "closed",
        rolled_back: bool = False,
        error: str | None = None,
    ) -> float:
        now = datetime.now(timezone.utc)
        total_realized = 0.0
        async with self.session_factory() as session:
            stmt = select(StructurePositionORM).where(
                and_(
                    StructurePositionORM.opportunity_id == opportunity_id,
                    StructurePositionORM.status == "open",
                )
            )
            rows = list((await session.execute(stmt)).scalars().all())
            for row in rows:
                exit_price = float(exit_prices.get(str(row.token_id), row.entry_price))
                shares = float(row.size_shares)
                proceeds = shares * max(0.001, min(0.999, exit_price))
                realized = proceeds - float(row.size_usd)
                row.status = "closed"
                row.closed_at = now
                row.exit_price = exit_price
                row.realized_pnl_usd = realized
                total_realized += realized
            bundle_stmt = (
                update(StructureBundleORM)
                .where(StructureBundleORM.opportunity_id == opportunity_id)
                .values(status=status, rolled_back=rolled_back, error=error, closed_at=now)
            )
            await session.execute(bundle_stmt)
            await session.commit()
        return total_realized

    async def mark_structure_bundle_open(self, opportunity_id: str) -> None:
        now = datetime.now(timezone.utc)
        async with self.session_factory() as session:
            stmt = (
                update(StructureBundleORM)
                .where(StructureBundleORM.opportunity_id == opportunity_id)
                .values(status="open", opened_at=now)
            )
            await session.execute(stmt)
            await session.commit()

    async def structure_status_snapshot(self) -> dict[str, float | int]:
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        async with self.session_factory() as session:
            open_pairs_stmt = select(func.count(StructureBundleORM.opportunity_id)).where(
                and_(StructureBundleORM.status == "open", StructureBundleORM.kind == "pair")
            )
            open_baskets_stmt = select(func.count(StructureBundleORM.opportunity_id)).where(
                and_(StructureBundleORM.status == "open", StructureBundleORM.kind == "basket")
            )
            day_stmt = select(func.coalesce(func.sum(StructurePositionORM.realized_pnl_usd), 0.0)).where(
                and_(StructurePositionORM.status == "closed", StructurePositionORM.closed_at >= day_ago)
            )
            week_stmt = select(func.coalesce(func.sum(StructurePositionORM.realized_pnl_usd), 0.0)).where(
                and_(StructurePositionORM.status == "closed", StructurePositionORM.closed_at >= week_ago)
            )
            return {
                "open_pairs": int((await session.scalar(open_pairs_stmt)) or 0),
                "open_baskets": int((await session.scalar(open_baskets_stmt)) or 0),
                "day_realized": float((await session.scalar(day_stmt)) or 0.0),
                "week_realized": float((await session.scalar(week_stmt)) or 0.0),
            }

    async def build_training_rows_from_features(
        self,
        lookback_days: int,
        horizon_minutes: int,
        min_move: float,
        max_spread_pct: float | None = None,
        min_volume_oi_ratio: float | None = None,
        min_tte_hours: float | None = None,
        max_sample_weight: float | None = None,
    ) -> list[dict[str, float | int | str]]:
        sql = text(
            """
            SELECT
                f.market_id,
                f.timestamp AS ts,
                f.implied_prob,
                f.spread,
                f.spread_pct,
                f.volume_1h,
                f.open_interest,
                f.volume_oi_ratio,
                f.time_to_expiry_hours,
                f.orderbook_imbalance,
                f.momentum_20,
                f.zscore_20,
                f.volatility_20,
                f.volatility_30,
                (
                    SELECT f2.implied_prob
                    FROM features f2
                    WHERE f2.market_id = f.market_id
                      AND f2.timestamp >= f.timestamp + make_interval(mins => :horizon_minutes)
                    ORDER BY f2.timestamp ASC
                    LIMIT 1
                ) AS future_prob
            FROM features f
            WHERE f.timestamp >= NOW() - make_interval(days => :lookback_days)
            ORDER BY f.market_id ASC, f.timestamp ASC
            """
        )
        rows: list[dict[str, float | int | str]] = []
        async with self.session_factory() as session:
            result = await session.execute(
                sql,
                {
                    "lookback_days": int(max(1, lookback_days)),
                    "horizon_minutes": int(max(1, horizon_minutes)),
                },
            )
            for row in result.mappings().all():
                try:
                    implied_prob = float(row["implied_prob"])
                    future_prob = float(row["future_prob"])
                    spread = float(row["spread"])
                    spread_pct = float(row["spread_pct"] or 0.0)
                    volume_1h = float(row["volume_1h"])
                    open_interest = float(row["open_interest"])
                    volume_oi_ratio = float(row["volume_oi_ratio"] or 0.0)
                    tte_hours = float(row["time_to_expiry_hours"])
                    imbalance = float(row["orderbook_imbalance"])
                    momentum_20 = float(row["momentum_20"] or 0.0)
                    zscore_20 = float(row["zscore_20"] or 0.0)
                    volatility_20 = float(row["volatility_20"] or 0.0)
                    volatility_30 = float(row["volatility_30"])
                except (TypeError, ValueError):
                    continue
                if max_spread_pct is not None and spread_pct > max_spread_pct:
                    continue
                if min_volume_oi_ratio is not None and volume_oi_ratio < min_volume_oi_ratio:
                    continue
                if min_tte_hours is not None and tte_hours < min_tte_hours:
                    continue
                delta = future_prob - implied_prob
                if abs(delta) < min_move:
                    continue
                target = 1 if delta > 0 else 0
                # Larger realized move and tighter spread get slightly higher weight.
                weight = max(0.5, abs(delta) / max(spread, 0.001))
                if max_sample_weight is not None:
                    weight = min(weight, max_sample_weight)
                weight = min(5.0, weight)
                ts = row["ts"]
                ts_text = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                rows.append(
                    {
                        "timestamp": ts_text,
                        "implied_prob": implied_prob,
                        "spread": spread,
                        "spread_pct": spread_pct,
                        "volume_1h": volume_1h,
                        "open_interest": open_interest,
                        "volume_oi_ratio": volume_oi_ratio,
                        "time_to_expiry_hours": tte_hours,
                        "orderbook_imbalance": imbalance,
                        "momentum_20": momentum_20,
                        "zscore_20": zscore_20,
                        "volatility_20": volatility_20,
                        "volatility_30": volatility_30,
                        "target": target,
                        "sample_weight": round(weight, 6),
                    }
                )
        return rows

    async def build_training_rows_from_settlements(
        self,
        lookback_days: int,
        max_spread_pct: float | None = None,
        min_volume_oi_ratio: float | None = None,
        min_tte_hours: float | None = None,
        max_sample_weight: float | None = None,
    ) -> list[dict[str, float | int | str]]:
        sql = text(
            """
            WITH ranked_features AS (
                SELECT
                    f.market_id,
                    f.timestamp AS ts,
                    f.implied_prob,
                    f.spread,
                    f.spread_pct,
                    f.volume_1h,
                    f.open_interest,
                    f.volume_oi_ratio,
                    f.time_to_expiry_hours,
                    f.orderbook_imbalance,
                    f.momentum_20,
                    f.zscore_20,
                    f.volatility_20,
                    f.volatility_30,
                    o.outcome_yes,
                    ROW_NUMBER() OVER (
                        PARTITION BY f.market_id, date_trunc('minute', f.timestamp)
                        ORDER BY f.timestamp DESC
                    ) AS rn
                FROM features f
                JOIN outcomes o
                  ON o.market_id = f.market_id
                WHERE f.timestamp >= NOW() - make_interval(days => :lookback_days)
                  AND o.outcome_yes IS NOT NULL
                  AND o.resolved_at IS NOT NULL
                  AND f.timestamp <= o.resolved_at
            )
            SELECT
                market_id,
                ts,
                implied_prob,
                spread,
                spread_pct,
                volume_1h,
                open_interest,
                volume_oi_ratio,
                time_to_expiry_hours,
                orderbook_imbalance,
                momentum_20,
                zscore_20,
                volatility_20,
                volatility_30,
                outcome_yes
            FROM ranked_features
            WHERE rn = 1
            ORDER BY market_id ASC, ts ASC
            """
        )
        rows: list[dict[str, float | int | str]] = []
        async with self.session_factory() as session:
            result = await session.execute(
                sql,
                {
                    "lookback_days": int(max(1, lookback_days)),
                },
            )
            for row in result.mappings().all():
                try:
                    implied_prob = float(row["implied_prob"])
                    spread = float(row["spread"])
                    spread_pct = float(row["spread_pct"] or 0.0)
                    volume_1h = float(row["volume_1h"])
                    open_interest = float(row["open_interest"])
                    volume_oi_ratio = float(row["volume_oi_ratio"] or 0.0)
                    tte_hours = float(row["time_to_expiry_hours"])
                    imbalance = float(row["orderbook_imbalance"])
                    momentum_20 = float(row["momentum_20"] or 0.0)
                    zscore_20 = float(row["zscore_20"] or 0.0)
                    volatility_20 = float(row["volatility_20"] or 0.0)
                    volatility_30 = float(row["volatility_30"])
                    outcome_yes = float(row["outcome_yes"])
                except (TypeError, ValueError):
                    continue

                if outcome_yes not in {0.0, 1.0}:
                    continue
                if max_spread_pct is not None and spread_pct > max_spread_pct:
                    continue
                if min_volume_oi_ratio is not None and volume_oi_ratio < min_volume_oi_ratio:
                    continue
                if min_tte_hours is not None and tte_hours < min_tte_hours:
                    continue

                time_focus = max(0.0, min(1.0, 1.0 - (tte_hours / 0.10)))
                spread_bonus = max(0.0, min(1.0, 1.0 - (spread_pct / max(max_spread_pct or 0.05, 0.05))))
                flow_bonus = max(0.0, min(1.0, volume_oi_ratio / 0.25))
                imbalance_bonus = max(0.0, min(1.0, abs(imbalance)))
                weight = 1.0 + (1.25 * time_focus) + (0.35 * flow_bonus) + (0.25 * spread_bonus) + (0.15 * imbalance_bonus)
                if max_sample_weight is not None:
                    weight = min(weight, max_sample_weight)
                weight = min(5.0, max(0.5, weight))

                ts = row["ts"]
                ts_text = ts.isoformat() if hasattr(ts, "isoformat") else str(ts)
                rows.append(
                    {
                        "timestamp": ts_text,
                        "implied_prob": implied_prob,
                        "spread": spread,
                        "spread_pct": spread_pct,
                        "volume_1h": volume_1h,
                        "open_interest": open_interest,
                        "volume_oi_ratio": volume_oi_ratio,
                        "time_to_expiry_hours": tte_hours,
                        "orderbook_imbalance": imbalance,
                        "momentum_20": momentum_20,
                        "zscore_20": zscore_20,
                        "volatility_20": volatility_20,
                        "volatility_30": volatility_30,
                        "target": int(outcome_yes >= 0.5),
                        "sample_weight": round(weight, 6),
                    }
                )
        return rows

    async def realized_pnl_window(self, since: datetime, strategy_id: str | None = None) -> float:
        async with self.session_factory() as session:
            filters = [PositionORM.status == "closed", PositionORM.closed_at >= since]
            if strategy_id is not None:
                filters.append(PositionORM.strategy_id == strategy_id)
            stmt = select(func.coalesce(func.sum(PositionORM.realized_pnl_usd), 0.0)).where(and_(*filters))
            value = await session.scalar(stmt)
            if strategy_id is not None:
                return float(value or 0.0)
            structure_stmt = select(func.coalesce(func.sum(StructurePositionORM.realized_pnl_usd), 0.0)).where(
                and_(StructurePositionORM.status == "closed", StructurePositionORM.closed_at >= since)
            )
            structure_value = await session.scalar(structure_stmt)
            return float(value or 0.0) + float(structure_value or 0.0)

    async def paper_trade_outcomes(self, since: datetime, strategy_id: str | None = None) -> list[float]:
        async with self.session_factory() as session:
            filters = [
                FillORM.trading_mode == TradingMode.PAPER.value,
                FillORM.created_at >= since,
                FillORM.pnl_usd != 0.0,
            ]
            if strategy_id is not None:
                filters.append(FillORM.strategy_id == strategy_id)
            stmt = select(FillORM.pnl_usd).where(and_(*filters))
            rows = (await session.execute(stmt)).all()
            return [float(v[0]) for v in rows]

    async def latest_equity(self, strategy_id: str | None = None) -> float:
        async with self.session_factory() as session:
            stmt = select(EquityCurveORM.equity_usd)
            if strategy_id is not None:
                stmt = stmt.where(EquityCurveORM.strategy_id == strategy_id)
            stmt = stmt.order_by(EquityCurveORM.id.desc()).limit(1)
            last = await session.scalar(stmt)
            if last is None:
                return self.starting_capital_usd
            return float(last)

    async def open_position_exposure(
        self,
        trading_mode: TradingMode,
        *,
        strategy_id: str | None = None,
        execution_mode: str | None = None,
        market_id: str | None = None,
        side: str | None = None,
    ) -> dict[str, float | int]:
        async with self.session_factory() as session:
            filters = [
                PositionORM.status == "open",
                PositionORM.trading_mode == trading_mode.value,
            ]
            if strategy_id is not None:
                filters.append(PositionORM.strategy_id == strategy_id)
            if execution_mode is not None:
                filters.append(PositionORM.execution_mode == execution_mode)
            if market_id is not None:
                filters.append(PositionORM.market_id == market_id)
            if side is not None:
                filters.append(PositionORM.side == side)
            count_stmt = select(func.count(PositionORM.id)).where(and_(*filters))
            notional_stmt = select(func.coalesce(func.sum(PositionORM.size_usd), 0.0)).where(and_(*filters))
            return {
                "count": int((await session.scalar(count_stmt)) or 0),
                "notional_usd": float((await session.scalar(notional_stmt)) or 0.0),
            }

    async def fill_notional_since(
        self,
        since: datetime,
        *,
        trading_mode: TradingMode,
        strategy_id: str | None = None,
        execution_mode: str | None = None,
    ) -> float:
        async with self.session_factory() as session:
            filters = [
                PositionORM.opened_at >= since,
                PositionORM.trading_mode == trading_mode.value,
            ]
            if strategy_id is not None:
                filters.append(PositionORM.strategy_id == strategy_id)
            if execution_mode is not None:
                filters.append(PositionORM.execution_mode == execution_mode)
            stmt = select(func.coalesce(func.sum(PositionORM.size_usd), 0.0)).where(and_(*filters))
            value = await session.scalar(stmt)
            return float(value or 0.0)

    async def strategy_compare_summary(
        self,
        window_minutes: int,
        primary_strategy_id: str,
        secondary_strategy_id: str,
    ) -> dict[str, float | int]:
        since = datetime.now(timezone.utc) - timedelta(minutes=max(1, int(window_minutes)))
        async with self.session_factory() as session:
            signal_stmt = select(
                SignalORM.strategy_id,
                SignalORM.market_id,
                SignalORM.side,
                SignalORM.created_at,
            ).where(
                and_(
                    SignalORM.created_at >= since,
                    SignalORM.strategy_id.in_([primary_strategy_id, secondary_strategy_id]),
                    SignalORM.status == "filled",
                )
            )
            signal_rows = (await session.execute(signal_stmt)).all()

            close_stmt = select(
                PositionORM.strategy_id,
                PositionORM.closed_at,
                PositionORM.realized_pnl_usd,
            ).where(
                and_(
                    PositionORM.status == "closed",
                    PositionORM.closed_at >= since,
                    PositionORM.strategy_id.in_([primary_strategy_id, secondary_strategy_id]),
                )
            )
            close_rows = (await session.execute(close_stmt)).all()

            fallback_stmt = select(func.count(PositionORM.id)).where(
                and_(
                    PositionORM.status == "closed",
                    PositionORM.closed_at >= since,
                    PositionORM.strategy_id.in_([primary_strategy_id, secondary_strategy_id]),
                    PositionORM.fallback_mark_close.is_(True),
                )
            )
            fallback_count = int((await session.scalar(fallback_stmt)) or 0)

        bucket_map: dict[tuple[datetime, str], dict[str, str]] = {}
        for strategy_id, market_id, side_value, created_at in signal_rows:
            if created_at is None:
                continue
            bucket = created_at.replace(second=0, microsecond=0)
            slot = bucket_map.setdefault((bucket, str(market_id)), {})
            slot[str(strategy_id)] = str(side_value)
        both_slots = 0
        same_side = 0
        for slot in bucket_map.values():
            if primary_strategy_id in slot and secondary_strategy_id in slot:
                both_slots += 1
                if slot[primary_strategy_id] == slot[secondary_strategy_id]:
                    same_side += 1

        pnl_a: dict[datetime, float] = {}
        pnl_b: dict[datetime, float] = {}
        for strategy_id, closed_at, realized_pnl in close_rows:
            if closed_at is None:
                continue
            bucket = closed_at.replace(second=0, microsecond=0)
            target = pnl_a if str(strategy_id) == primary_strategy_id else pnl_b
            target[bucket] = target.get(bucket, 0.0) + float(realized_pnl or 0.0)

        union_buckets = sorted(set(pnl_a) | set(pnl_b))
        corr = 0.0
        if len(union_buckets) >= 2:
            xs = [pnl_a.get(bucket, 0.0) for bucket in union_buckets]
            ys = [pnl_b.get(bucket, 0.0) for bucket in union_buckets]
            mean_x = sum(xs) / len(xs)
            mean_y = sum(ys) / len(ys)
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
            var_x = sum((x - mean_x) ** 2 for x in xs)
            var_y = sum((y - mean_y) ** 2 for y in ys)
            if var_x > 0 and var_y > 0:
                corr = cov / math.sqrt(var_x * var_y)

        return {
            "same_side_overlap": (same_side / both_slots) if both_slots > 0 else 0.0,
            "shared_signal_slots": both_slots,
            "pnl_correlation": max(-1.0, min(1.0, float(corr))),
            "fallback_mark_close_count": fallback_count,
        }

    async def status_snapshot(self, strategy_id: str | None = None) -> dict[str, float | int]:
        now = datetime.now(timezone.utc)
        day_ago = now - timedelta(days=1)
        week_ago = now - timedelta(days=7)
        async with self.session_factory() as session:
            open_filters = [PositionORM.status == "open"]
            closed_filters = [PositionORM.status == "closed"]
            day_filters = [PositionORM.status == "closed", PositionORM.closed_at >= day_ago]
            week_filters = [PositionORM.status == "closed", PositionORM.closed_at >= week_ago]
            total_filters = [PositionORM.status == "closed"]
            if strategy_id is not None:
                open_filters.append(PositionORM.strategy_id == strategy_id)
                closed_filters.append(PositionORM.strategy_id == strategy_id)
                day_filters.append(PositionORM.strategy_id == strategy_id)
                week_filters.append(PositionORM.strategy_id == strategy_id)
                total_filters.append(PositionORM.strategy_id == strategy_id)

            open_positions_stmt = select(func.count(PositionORM.id)).where(and_(*open_filters))
            open_positions = int((await session.scalar(open_positions_stmt)) or 0)
            if strategy_id is None:
                structure_open_stmt = select(func.count(StructurePositionORM.id)).where(StructurePositionORM.status == "open")
                open_positions += int((await session.scalar(structure_open_stmt)) or 0)

            closed_positions_stmt = select(func.count(PositionORM.id)).where(and_(*closed_filters))
            closed_positions = int((await session.scalar(closed_positions_stmt)) or 0)
            if strategy_id is None:
                structure_closed_stmt = select(func.count(StructurePositionORM.id)).where(StructurePositionORM.status == "closed")
                closed_positions += int((await session.scalar(structure_closed_stmt)) or 0)

            day_pnl_stmt = select(func.coalesce(func.sum(PositionORM.realized_pnl_usd), 0.0)).where(and_(*day_filters))
            week_pnl_stmt = select(func.coalesce(func.sum(PositionORM.realized_pnl_usd), 0.0)).where(and_(*week_filters))
            total_pnl_stmt = select(func.coalesce(func.sum(PositionORM.realized_pnl_usd), 0.0)).where(and_(*total_filters))
            wins_stmt = select(func.count(PositionORM.id)).where(and_(*total_filters, PositionORM.realized_pnl_usd > 0))
            losses_stmt = select(func.count(PositionORM.id)).where(and_(*total_filters, PositionORM.realized_pnl_usd < 0))

            day_pnl = float((await session.scalar(day_pnl_stmt)) or 0.0)
            week_pnl = float((await session.scalar(week_pnl_stmt)) or 0.0)
            total_pnl = float((await session.scalar(total_pnl_stmt)) or 0.0)
            wins = int((await session.scalar(wins_stmt)) or 0)
            losses = int((await session.scalar(losses_stmt)) or 0)
            if strategy_id is None:
                day_structure_stmt = select(func.coalesce(func.sum(StructurePositionORM.realized_pnl_usd), 0.0)).where(
                    and_(StructurePositionORM.status == "closed", StructurePositionORM.closed_at >= day_ago)
                )
                week_structure_stmt = select(func.coalesce(func.sum(StructurePositionORM.realized_pnl_usd), 0.0)).where(
                    and_(StructurePositionORM.status == "closed", StructurePositionORM.closed_at >= week_ago)
                )
                total_structure_stmt = select(func.coalesce(func.sum(StructurePositionORM.realized_pnl_usd), 0.0)).where(
                    StructurePositionORM.status == "closed"
                )
                day_pnl += float((await session.scalar(day_structure_stmt)) or 0.0)
                week_pnl += float((await session.scalar(week_structure_stmt)) or 0.0)
                total_pnl += float((await session.scalar(total_structure_stmt)) or 0.0)
        decisions = wins + losses
        return {
            "open_positions": open_positions,
            "closed_positions": closed_positions,
            "day_pnl": day_pnl,
            "week_pnl": week_pnl,
            "total_pnl": total_pnl,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / decisions) if decisions > 0 else 0.0,
        }

    async def data_freshness_snapshot(self) -> dict[str, datetime | int | None]:
        now = datetime.now(timezone.utc)
        since_5m = now - timedelta(minutes=5)
        async with self.session_factory() as session:
            latest_tick_ts = await session.scalar(select(func.max(MarketTickORM.timestamp)))
            latest_feature_ts = await session.scalar(select(func.max(FeatureORM.timestamp)))
            latest_signal_ts = await session.scalar(select(func.max(SignalORM.created_at)))
            ticks_5m = await session.scalar(
                select(func.count(MarketTickORM.id)).where(MarketTickORM.timestamp >= since_5m)
            )
        return {
            "latest_tick_ts": latest_tick_ts,
            "latest_feature_ts": latest_feature_ts,
            "latest_signal_ts": latest_signal_ts,
            "ticks_5m": int(ticks_5m or 0),
        }

    async def signal_window_summary(
        self,
        window_minutes: int = 360,
        strategy_id: str | None = None,
    ) -> dict[str, float | int | list[tuple[str, int]]]:
        now = datetime.now(timezone.utc)
        minutes = max(1, int(window_minutes))
        since = now - timedelta(minutes=minutes)
        async with self.session_factory() as session:
            signal_base = [SignalORM.created_at >= since]
            if strategy_id is not None:
                signal_base.append(SignalORM.strategy_id == strategy_id)
            total_stmt = select(func.count(SignalORM.id)).where(and_(*signal_base))
            filled_stmt = select(func.count(SignalORM.id)).where(and_(*signal_base, SignalORM.status == "filled"))
            filled_yes_stmt = select(func.count(SignalORM.id)).where(
                and_(*signal_base, SignalORM.status == "filled", SignalORM.side == "YES")
            )
            filled_no_stmt = select(func.count(SignalORM.id)).where(
                and_(*signal_base, SignalORM.status == "filled", SignalORM.side == "NO")
            )
            yes_stmt = select(func.count(SignalORM.id)).where(and_(*signal_base, SignalORM.side == "YES"))
            no_stmt = select(func.count(SignalORM.id)).where(and_(*signal_base, SignalORM.side == "NO"))

            total_signals = int((await session.scalar(total_stmt)) or 0)
            filled_signals = int((await session.scalar(filled_stmt)) or 0)
            filled_yes_signals = int((await session.scalar(filled_yes_stmt)) or 0)
            filled_no_signals = int((await session.scalar(filled_no_stmt)) or 0)
            yes_signals = int((await session.scalar(yes_stmt)) or 0)
            no_signals = int((await session.scalar(no_stmt)) or 0)

            reject_stmt = (
                select(SignalORM.status, func.count(SignalORM.id))
                .where(and_(*signal_base, SignalORM.status.like("rejected:%")))
                .group_by(SignalORM.status)
                .order_by(func.count(SignalORM.id).desc())
                .limit(3)
            )
            reject_rows = (await session.execute(reject_stmt)).all()
            reject_top = [(str(status), int(count)) for status, count in reject_rows]

            position_base = [PositionORM.status == "closed", PositionORM.closed_at >= since]
            if strategy_id is not None:
                position_base.append(PositionORM.strategy_id == strategy_id)
            realized_main_stmt = select(func.coalesce(func.sum(PositionORM.realized_pnl_usd), 0.0)).where(and_(*position_base))
            wins_stmt = select(func.count(PositionORM.id)).where(and_(*position_base, PositionORM.realized_pnl_usd > 0))
            losses_stmt = select(func.count(PositionORM.id)).where(and_(*position_base, PositionORM.realized_pnl_usd < 0))
            no_fill_base = [FillORM.created_at >= since, FillORM.trading_mode == TradingMode.PAPER.value, FillORM.side == "NO"]
            if strategy_id is not None:
                no_fill_base.append(FillORM.strategy_id == strategy_id)
            no_realized_stmt = select(func.coalesce(func.sum(FillORM.pnl_usd), 0.0)).where(and_(*no_fill_base))
            realized_pnl = float((await session.scalar(realized_main_stmt)) or 0.0)
            wins = int((await session.scalar(wins_stmt)) or 0)
            losses = int((await session.scalar(losses_stmt)) or 0)
            if strategy_id is None:
                realized_structure_stmt = select(func.coalesce(func.sum(StructurePositionORM.realized_pnl_usd), 0.0)).where(
                    and_(StructurePositionORM.status == "closed", StructurePositionORM.closed_at >= since)
                )
                realized_pnl += float((await session.scalar(realized_structure_stmt)) or 0.0)
            no_realized_pnl = float((await session.scalar(no_realized_stmt)) or 0.0)

        fill_rate = (filled_signals / total_signals) if total_signals > 0 else 0.0
        yes_ratio = (yes_signals / total_signals) if total_signals > 0 else 0.0
        no_ratio = (no_signals / total_signals) if total_signals > 0 else 0.0
        decisions = wins + losses
        return {
            "window_minutes": minutes,
            "total_signals": total_signals,
            "filled_signals": filled_signals,
            "filled_yes_signals": filled_yes_signals,
            "filled_no_signals": filled_no_signals,
            "fill_rate": fill_rate,
            "yes_signals": yes_signals,
            "no_signals": no_signals,
            "yes_ratio": yes_ratio,
            "no_ratio": no_ratio,
            "realized_pnl": realized_pnl,
            "no_realized_pnl": no_realized_pnl,
            "wins": wins,
            "losses": losses,
            "win_rate": (wins / decisions) if decisions > 0 else 0.0,
            "reject_top": reject_top,
        }
