from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from polymethemoney.config import Settings
from polymethemoney.domain import (
    EXECUTION_MODE_FUNDED_PAPER,
    STRATEGY_LEGACY,
    FeatureVector,
    Side,
    Signal,
    StrategySpec,
    get_strategy_spec,
)
from polymethemoney.services.execution_engine import ExecutionEngine
from polymethemoney.services.model_engine import ModelEngine
from polymethemoney.state import RuntimeState
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)


class ContrarianEngine:
    def __init__(
        self,
        settings: Settings,
        feature_queue: asyncio.Queue[FeatureVector],
        model_engine: ModelEngine,
        execution_engine: ExecutionEngine,
        store: Store,
        runtime_state: RuntimeState | None = None,
        strategy_id: str = STRATEGY_LEGACY,
        model_variant: str = "model_a",
        direction_mode: str = "contrarian",
        strategy_spec: StrategySpec | None = None,
    ) -> None:
        self.settings = settings
        self.feature_queue = feature_queue
        self.model_engine = model_engine
        self.execution_engine = execution_engine
        self.store = store
        self.runtime_state = runtime_state
        self.strategy_spec = strategy_spec or get_strategy_spec(strategy_id)
        self.strategy_id = self.strategy_spec.strategy_id
        self._latest_by_market: dict[str, FeatureVector] = {}
        self._last_emit_slot: dict[str, str] = {}
        self._emit_lock = asyncio.Lock()

    async def run(self) -> None:
        await asyncio.gather(self._consume_features(), self._emit_loop())

    async def _consume_features(self) -> None:
        while True:
            feature = await self.feature_queue.get()
            is_new_market = feature.market_id not in self._latest_by_market
            self._latest_by_market[feature.market_id] = feature
            if is_new_market:
                await self._emit_if_slot_open(feature.market_id)

    async def _emit_loop(self) -> None:
        while True:
            now = datetime.now(timezone.utc)
            slot_second = max(0, min(55, int(self.settings.strategy_slot_second_utc)))
            next_run = now.replace(second=slot_second, microsecond=0)
            if next_run <= now:
                next_run += timedelta(minutes=1)
            await asyncio.sleep(max(0.05, (next_run - now).total_seconds()))
            try:
                await self._emit_due_signals(datetime.now(timezone.utc))
            except Exception:
                logger.exception("strategy emit failed: %s", self.strategy_id)

    async def _emit_if_slot_open(self, market_id: str) -> None:
        now = datetime.now(timezone.utc)
        if now.second < int(self.settings.strategy_slot_second_utc):
            return
        await self._emit_market_if_due(market_id, now)

    async def _emit_due_signals(self, now: datetime) -> None:
        if self.runtime_state is not None and self.runtime_state.paused:
            return
        for market_id in sorted(self._latest_by_market.keys()):
            await self._emit_market_if_due(market_id, now)

    async def _emit_market_if_due(self, market_id: str, now: datetime) -> None:
        async with self._emit_lock:
            feature = self._latest_by_market.get(market_id)
            if feature is None:
                return
            if feature.timestamp < now - timedelta(seconds=120):
                return
            slot_key = now.strftime("%Y%m%d%H%M")
            if self._last_emit_slot.get(market_id) == slot_key:
                return

            expiry_map = await self.store.latest_market_expiries([market_id])
            expiry = expiry_map.get(market_id)
            grace = timedelta(seconds=max(0, int(self.settings.contrarian_expiry_grace_seconds)))
            if expiry is not None and now >= expiry + grace:
                self._latest_by_market.pop(market_id, None)
                self._last_emit_slot.pop(market_id, None)
                return

            open_positions = await self.store.get_open_positions(
                self.runtime_state.trading_mode if self.runtime_state is not None else None,
                strategy_id=self.strategy_id,
            )
            if len(open_positions) >= max(1, int(self.strategy_spec.max_positions)):
                return

            signal = self._build_strategy_signal(feature, now)
            if signal is None:
                return
            await self.execution_engine.on_signal(signal)
            self._last_emit_slot[market_id] = slot_key
            logger.info(
                "strategy signal strategy=%s market=%s side=%s fair=%.4f implied=%.4f edge=%.6f net_ev=%.6f",
                self.strategy_id,
                signal.market_id,
                signal.side.value,
                signal.fair_prob,
                signal.implied_prob,
                signal.edge,
                signal.net_ev,
            )

    def _build_strategy_signal(self, fv: FeatureVector, now: datetime) -> Signal | None:
        prediction = self.model_engine.predict_strategy(self.strategy_id, fv)
        fair_yes = max(0.001, min(0.999, float(prediction.fair_probability)))
        implied_yes = max(0.001, min(0.999, float(fv.implied_prob)))
        confidence = float(max(0.05, min(0.99, float(prediction.confidence))))
        prediction_gap = abs(fair_yes - implied_yes)
        if not self._passes_strategy_gate(fv, fair_yes, implied_yes, confidence, prediction_gap):
            return None

        if fair_yes >= implied_yes:
            side = Side.YES
            fair = fair_yes
            implied = implied_yes
        else:
            side = Side.NO
            fair = 1.0 - fair_yes
            implied = 1.0 - implied_yes

        fair = max(0.001, min(0.999, fair))
        implied = max(0.001, min(0.999, implied))
        edge = fair - implied
        fee_cost = (float(self.settings.taker_fee_bps) / 10000.0) * 0.5
        slippage_cost = float(self.settings.slippage_bps) / 10000.0
        net_ev = edge - fee_cost - slippage_cost
        scale = max(1e-4, float(self.settings.signal_edge_score_scale))
        model_edge_score = max(0.0, min(45.0, (net_ev / scale) * 45.0))
        score = int(max(0, min(100, model_edge_score)))
        return Signal(
            strategy_id=self.strategy_id,
            market_id=fv.market_id,
            side=side,
            fair_prob=fair,
            implied_prob=implied,
            edge=edge,
            net_ev=net_ev,
            score=score,
            ttl_seconds=self.settings.order_ttl_seconds,
            liquidity_score=max(0.0, min(1.0, 1.0 - (float(fv.spread) / max(float(self.settings.max_spread), 0.001)))),
            model_edge_score=model_edge_score,
            regime_score=0.0,
            execution_mode=self.strategy_spec.execution_mode,
            model_confidence=confidence,
            effective_edge=edge,
            created_at=now,
        )

    def _passes_strategy_gate(
        self,
        fv: FeatureVector,
        fair_yes: float,
        implied_yes: float,
        confidence: float,
        prediction_gap: float,
    ) -> bool:
        tte_seconds = float(fv.time_to_expiry_hours) * 3600.0
        if tte_seconds < float(self.settings.funded_min_tte_seconds):
            return False
        if tte_seconds > float(self.settings.funded_max_tte_seconds):
            return False
        if float(fv.spread) > float(self.settings.funded_max_spread):
            return False
        if self.strategy_spec.execution_mode == EXECUTION_MODE_FUNDED_PAPER:
            if prediction_gap < float(self.settings.funded_min_prediction_gap):
                return False
            if confidence < float(self.settings.funded_min_model_confidence):
                return False
            side_edge = fair_yes - implied_yes if fair_yes >= implied_yes else (1.0 - fair_yes) - (1.0 - implied_yes)
            fee_cost = (float(self.settings.taker_fee_bps) / 10000.0) * 0.5
            slippage_cost = float(self.settings.slippage_bps) / 10000.0
            if (side_edge - fee_cost - slippage_cost) <= 0:
                return False
        return True
