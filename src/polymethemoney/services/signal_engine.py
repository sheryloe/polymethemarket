from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

from polymethemoney.config import Settings
from polymethemoney.domain import FeatureVector, Side, Signal
from polymethemoney.services.execution_engine import ExecutionEngine
from polymethemoney.services.model_engine import ModelEngine
from polymethemoney.state import RuntimeState, TimedReasonEvent

logger = logging.getLogger(__name__)


class SignalEngine:
    _YES_PRIORITY_POLICY = "YES_PRIORITY"
    _BALANCED_POLICY = "BALANCED"

    def __init__(
        self,
        settings: Settings,
        feature_queue: asyncio.Queue[FeatureVector],
        model_engine: ModelEngine,
        execution_engine: ExecutionEngine,
        runtime_state: RuntimeState | None = None,
    ) -> None:
        self.settings = settings
        self.feature_queue = feature_queue
        self.model_engine = model_engine
        self.execution_engine = execution_engine
        self.runtime_state = runtime_state
        self._last_implied_prob_by_market: dict[str, float] = {}
        self._trend_deltas: deque[float] = deque(maxlen=max(60, self.settings.trend_window_size))
        self._last_signal_at_by_market_side: dict[tuple[str, str], datetime] = {}
        self._recent_candidate_sides: deque[str] = deque(maxlen=max(20, self.settings.signal_side_balance_window))
        self._recent_approved_sides: deque[str] = deque(maxlen=300)

    async def run(self) -> None:
        while True:
            feature = await self.feature_queue.get()
            signal = self._generate_signal(feature)
            if signal is None:
                continue
            await self.execution_engine.on_signal(signal)

    def _generate_signal(self, fv: FeatureVector) -> Signal | None:
        self._update_trend_state(fv)
        if not self._passes_liquidity_filter(fv):
            return self._reject_candidate("liquidity_filter")

        prediction = self.model_engine.predict_dual_score(fv)
        fair_yes = prediction.blended_prob
        model_confidence = prediction.confidence
        implied_yes = fv.implied_prob

        if model_confidence < self.settings.signal_min_confidence:
            return self._reject_candidate("min_confidence")

        quality = self._quality_score(fv, model_confidence)
        if quality < self.settings.signal_min_quality:
            return self._reject_candidate("min_quality")

        edge_yes = fair_yes - implied_yes
        alpha_bonus = self._alpha_bonus(fv)
        raw_yes = edge_yes + alpha_bonus
        raw_no = -edge_yes - alpha_bonus

        effective_yes = self._effective_edge(raw_yes, model_confidence, quality, Side.YES, fv)
        effective_no = self._effective_edge(raw_no, model_confidence, quality, Side.NO, fv)

        fee_cost = (self.settings.taker_fee_bps / 10000.0) * 0.5
        slippage_cost = self.settings.slippage_bps / 10000.0
        net_yes = effective_yes - fee_cost - slippage_cost
        net_no = effective_no - fee_cost - slippage_cost

        side_policy = (self.settings.signal_side_policy or self._BALANCED_POLICY).strip().upper()
        if side_policy == self._YES_PRIORITY_POLICY:
            margin = max(0.0, self.settings.signal_yes_priority_margin)
            if net_no >= net_yes + margin:
                side = Side.NO
                net_ev = net_no
                effective_edge = effective_no
                fair = 1.0 - fair_yes
                implied = 1.0 - implied_yes
                edge = -edge_yes
            else:
                side = Side.YES
                net_ev = net_yes
                effective_edge = effective_yes
                fair = fair_yes
                implied = implied_yes
                edge = edge_yes
        else:
            if net_no > net_yes:
                side = Side.NO
                net_ev = net_no
                effective_edge = effective_no
                fair = 1.0 - fair_yes
                implied = 1.0 - implied_yes
                edge = -edge_yes
            else:
                side = Side.YES
                net_ev = net_yes
                effective_edge = effective_yes
                fair = fair_yes
                implied = implied_yes
                edge = edge_yes

        self._record_candidate_side(side.value)
        if not self._passes_contract_price(implied):
            return self._reject_candidate("min_contract_price")
        if not self._passes_net_ev_filter(implied=implied, net_ev=net_ev):
            return self._reject_candidate("net_ev_filter")
        if not self._passes_reentry_cooldown(fv.market_id, side):
            return self._reject_candidate("reentry_cooldown")

        model_edge_score = self._edge_score(net_ev)
        liquidity_score = self._liquidity_score(fv)
        regime_score = self.model_engine.regime_score(fv)
        quality_score = min(10.0, max(0.0, quality * 10.0))
        score = int(round(model_edge_score + liquidity_score + regime_score + quality_score))
        score = max(0, min(100, score))

        signal = Signal(
            market_id=fv.market_id,
            side=side,
            fair_prob=max(0.001, min(0.999, fair)),
            implied_prob=max(0.001, min(0.999, implied)),
            edge=edge,
            net_ev=net_ev,
            score=score,
            ttl_seconds=self.settings.order_ttl_seconds,
            liquidity_score=liquidity_score,
            model_edge_score=model_edge_score,
            regime_score=regime_score,
            model_confidence=model_confidence,
            effective_edge=effective_edge,
            created_at=datetime.now(timezone.utc),
        )
        self._record_candidate_accept(side.value)
        return signal

    def _alpha_bonus(self, fv: FeatureVector) -> float:
        trend_bias, trend_confidence = self._trend_bias()
        trend_score = trend_bias * trend_confidence
        revert_score = -fv.orderbook_imbalance
        weighted = (
            (self.settings.signal_alpha_trend_weight * trend_score)
            + (self.settings.signal_alpha_revert_weight * revert_score)
        )
        cap = max(0.0, self.settings.signal_alpha_bonus_cap)
        bonus = weighted * cap
        return max(-cap, min(cap, bonus))

    def _effective_edge(
        self,
        raw_edge: float,
        model_confidence: float,
        quality: float,
        side: Side,
        fv: FeatureVector,
    ) -> float:
        effective = raw_edge * (0.4 + 0.6 * model_confidence)
        effective = self._trend_adjusted_edge(side, effective)
        effective *= (0.5 + 0.5 * quality)
        return effective

    def _edge_score(self, net_ev: float) -> float:
        scale = max(1e-4, self.settings.signal_edge_score_scale)
        score = (net_ev / scale) * 45.0
        return max(0.0, min(45.0, score))

    def _quality_score(self, fv: FeatureVector, model_confidence: float) -> float:
        spread_penalty = min(1.0, fv.spread / max(self.settings.max_spread, 1e-6))
        vol_ref = max(1e-4, self.settings.signal_vol_ref)
        vol_penalty = min(1.0, fv.volatility_30 / vol_ref)
        spread_weight = max(0.0, min(1.0, self.settings.signal_spread_penalty_weight))
        vol_weight = max(0.0, min(1.0, self.settings.signal_vol_penalty_weight))
        liquidity = self._liquidity_score(fv) / 30.0
        quality = model_confidence
        quality *= (1.0 - (spread_penalty * spread_weight))
        quality *= (1.0 - (vol_penalty * vol_weight))
        quality *= 0.7 + (0.3 * liquidity)
        return max(0.0, min(1.0, quality))

    def _passes_net_ev_filter(self, implied: float, net_ev: float) -> bool:
        base_min = max(0.0, self.settings.signal_min_net_ev)
        floor = max(0.001, min(0.499, self.settings.signal_tail_prob_floor))
        ceiling = max(0.501, min(0.999, self.settings.signal_tail_prob_ceiling))
        tail_extra = max(0.0, self.settings.signal_tail_extra_net_ev)
        in_tail = implied <= floor or implied >= ceiling
        required = base_min + (tail_extra if in_tail else 0.0)
        return net_ev > required

    def _passes_contract_price(self, implied: float) -> bool:
        min_price = max(0.001, min(0.499, self.settings.signal_min_contract_price))
        return implied >= min_price

    def _passes_reentry_cooldown(self, market_id: str, side: Side) -> bool:
        cooldown = max(0, self.settings.signal_reentry_cooldown_seconds)
        if cooldown <= 0:
            return True
        key = (market_id, side.value)
        now = datetime.now(timezone.utc)
        previous = self._last_signal_at_by_market_side.get(key)
        if previous is not None:
            elapsed = (now - previous).total_seconds()
            if elapsed < cooldown:
                return False
        self._last_signal_at_by_market_side[key] = now
        return True

    def _passes_liquidity_filter(self, fv: FeatureVector) -> bool:
        return (
            fv.open_interest >= self.settings.min_open_interest_usd
            and fv.spread <= self.settings.max_spread
            and fv.volume_1h >= self.settings.min_hourly_volume_usd
        )

    def _liquidity_score(self, fv: FeatureVector) -> float:
        oi_score = min(1.0, fv.open_interest / max(self.settings.min_open_interest_usd * 2.0, 1.0))
        vol_score = min(1.0, fv.volume_1h / max(self.settings.min_hourly_volume_usd * 2.0, 1.0))
        spread_score = max(0.0, 1.0 - (fv.spread / max(self.settings.max_spread * 1.5, 1e-6)))
        composite = 0.4 * oi_score + 0.4 * vol_score + 0.2 * spread_score
        return max(0.0, min(30.0, composite * 30.0))

    def _update_trend_state(self, fv: FeatureVector) -> None:
        prev = self._last_implied_prob_by_market.get(fv.market_id)
        self._last_implied_prob_by_market[fv.market_id] = fv.implied_prob
        if prev is None:
            return
        delta = fv.implied_prob - prev
        if abs(delta) < 1e-6:
            return
        liquidity_weight = min(
            1.5,
            0.5 + (fv.volume_1h / max(self.settings.min_hourly_volume_usd * 2.0, 1.0)),
        )
        self._trend_deltas.append(delta * liquidity_weight)

    def _trend_adjusted_edge(self, side: Side, effective_edge: float) -> float:
        if not self.settings.trend_adaptive_enabled:
            return effective_edge
        trend_bias, trend_confidence = self._trend_bias()
        if trend_confidence <= 0:
            return effective_edge
        direction = 1.0 if side == Side.YES else -1.0
        alignment = direction * trend_bias
        if alignment >= 0:
            multiplier = 1.0 + (alignment * trend_confidence * self.settings.trend_align_boost)
        else:
            multiplier = 1.0 + (alignment * trend_confidence * self.settings.trend_revert_penalty)
        multiplier = max(0.35, min(1.75, multiplier))
        return effective_edge * multiplier

    def _trend_bias(self) -> tuple[float, float]:
        n = len(self._trend_deltas)
        if n < 20:
            return 0.0, 0.0
        mean_delta = sum(self._trend_deltas) / n
        variance = sum((value - mean_delta) ** 2 for value in self._trend_deltas) / max(1, n - 1)
        stdev = variance**0.5
        if stdev <= 1e-9:
            return 0.0, min(1.0, n / 200.0)
        z_score = mean_delta / stdev
        bias = max(-1.0, min(1.0, z_score / 3.0))
        confidence = min(1.0, n / 200.0)
        return bias, confidence

    def _record_candidate_side(self, side: str) -> None:
        self._recent_candidate_sides.append(side)
        if self.runtime_state is not None:
            self.runtime_state.recent_signal_candidate_sides.append(side)

    def _record_candidate_accept(self, side: str) -> None:
        self._recent_approved_sides.append(side)
        if self.runtime_state is not None:
            self.runtime_state.recent_approved_signal_sides.append(side)
            self.runtime_state.recent_signal_candidate_rejects.append("accepted")

    def _reject_candidate(self, reason: str) -> Signal | None:
        if self.runtime_state is not None:
            self.runtime_state.recent_signal_candidate_rejects.append(reason)
            if reason.startswith("no_guard"):
                self.runtime_state.recent_no_guard_rejects.append(
                    TimedReasonEvent(timestamp=datetime.now(timezone.utc), reason=reason)
                )
        return None
