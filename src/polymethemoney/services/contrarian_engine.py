from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone

from polymethemoney.config import Settings
from polymethemoney.domain import STRATEGY_LEGACY, FeatureVector, Side, Signal
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
    ) -> None:
        self.settings = settings
        self.feature_queue = feature_queue
        self.model_engine = model_engine
        self.execution_engine = execution_engine
        self.store = store
        self.runtime_state = runtime_state
        self.strategy_id = strategy_id
        self.model_variant = model_variant
        self.direction_mode = direction_mode
        self._latest_by_market: dict[str, FeatureVector] = {}
        self._last_emit_at: dict[str, datetime] = {}

    async def run(self) -> None:
        await asyncio.gather(self._consume_features(), self._emit_loop())

    async def _consume_features(self) -> None:
        while True:
            feature = await self.feature_queue.get()
            self._latest_by_market[feature.market_id] = feature

    async def _emit_loop(self) -> None:
        interval = max(5, int(self.settings.contrarian_entry_interval_seconds))
        while True:
            await asyncio.sleep(interval)
            try:
                await self._emit_due_signals(interval)
            except Exception:
                logger.exception("contrarian emit failed")

    async def _emit_due_signals(self, interval: int) -> None:
        if self.runtime_state is not None and self.runtime_state.paused:
            return
        if not self._latest_by_market:
            return

        now = datetime.now(timezone.utc)
        grace = timedelta(seconds=max(0, int(self.settings.contrarian_expiry_grace_seconds)))
        freshness = timedelta(seconds=max(interval * 2, 120))
        market_ids = list(self._latest_by_market.keys())

        try:
            expiries = await self.store.latest_market_expiries(market_ids)
        except Exception:
            logger.exception("contrarian expiry lookup failed")
            expiries = {}

        mode = self.runtime_state.trading_mode if self.runtime_state is not None else None
        try:
            open_positions = await self.store.get_open_positions(mode, strategy_id=self.strategy_id)
        except Exception:
            open_positions = []
        if self.settings.ab_test_enabled:
            max_positions = max(1, int(self.settings.model_max_positions))
        else:
            max_positions = max(1, int(self.settings.contrarian_max_positions))
        if len(open_positions) >= max_positions:
            return

        for market_id in sorted(market_ids):
            feature = self._latest_by_market.get(market_id)
            if feature is None:
                continue
            if feature.timestamp < now - freshness:
                continue
            expiry = expiries.get(market_id)
            if expiry is not None and now >= expiry + grace:
                self._latest_by_market.pop(market_id, None)
                self._last_emit_at.pop(market_id, None)
                continue
            last_emit = self._last_emit_at.get(market_id)
            if last_emit is not None and (now - last_emit).total_seconds() < interval:
                continue

            signal = self._build_strategy_signal(feature, now)
            if signal is None:
                continue
            await self.execution_engine.on_signal(signal)
            self._last_emit_at[market_id] = now
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
        fair_yes, confidence = self._strategy_prediction(fv)
        implied_yes = max(0.001, min(0.999, float(fv.implied_prob)))

        if self.direction_mode == "direct":
            if fair_yes >= implied_yes:
                side = Side.YES
                fair = fair_yes
                implied = implied_yes
            else:
                side = Side.NO
                fair = 1.0 - fair_yes
                implied = 1.0 - implied_yes
        else:
            if fair_yes >= implied_yes:
                side = Side.NO
                fair = 1.0 - fair_yes
                implied = 1.0 - implied_yes
            else:
                side = Side.YES
                fair = fair_yes
                implied = implied_yes

        fair = max(0.001, min(0.999, fair))
        implied = max(0.001, min(0.999, implied))
        edge = fair - implied
        fee_cost = (self.settings.taker_fee_bps / 10000.0) * 0.5
        slippage_cost = self.settings.slippage_bps / 10000.0
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
            liquidity_score=0.0,
            model_edge_score=model_edge_score,
            regime_score=0.0,
            model_confidence=confidence,
            effective_edge=edge,
            created_at=now,
        )

    def _strategy_prediction(self, fv: FeatureVector) -> tuple[float, float]:
        if self.model_variant == "model_b":
            prediction = self.model_engine.predict_model_b(fv)
        else:
            prediction = self.model_engine.predict_model_a(fv)
        fair = max(0.001, min(0.999, float(prediction.fair_probability)))
        confidence = float(max(0.05, min(0.95, float(prediction.confidence))))
        return fair, confidence
