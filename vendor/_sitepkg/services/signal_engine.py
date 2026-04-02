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
    _NO_RATIO_WINDOW = 300

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
        self._recent_signal_sides: deque[str] = deque(maxlen=max(10, self.settings.signal_side_balance_window))
        self._recent_candidate_sides: deque[str] = deque(maxlen=max(20, self.settings.signal_side_balance_window))
        self._recent_approved_sides: deque[str] = deque(maxlen=self._NO_RATIO_WINDOW)

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
        trend_bias, _ = self._trend_bias()
        prediction = self.model_engine.predict(fv)
        fair_yes = prediction.fair_probability
        model_confidence = prediction.confidence
        implied_yes = fv.implied_prob
        fair_yes = self._regime_shrunk_fair(fair_yes, implied_yes, fv, model_confidence)
        min_confidence = self._dynamic_min_confidence(fv)
        if model_confidence < min_confidence:
            return self._reject_candidate("min_confidence")
        edge_yes = fair_yes - implied_yes
        fair_no = 1.0 - fair_yes
        implied_no = 1.0 - implied_yes
        edge_no = fair_no - implied_no

        risk_yes = self._risk_adjusted_edge(edge_yes, fv)
        risk_no = self._risk_adjusted_edge(edge_no, fv)
        effective_yes = self._build_effective_edge(Side.YES, fv, risk_yes, model_confidence)
        effective_no = self._build_effective_edge(Side.NO, fv, risk_no, model_confidence)
        fee_cost = (self.settings.taker_fee_bps / 10000.0) * 0.5
        slippage_cost = self.settings.slippage_bps / 10000.0
        net_yes = effective_yes - fee_cost - slippage_cost
        net_no = effective_no - fee_cost - slippage_cost

        no_guard_reject = self._evaluate_no_guard(
            net_ev=net_no,
            model_confidence=model_confidence,
            trend_bias=trend_bias,
            orderbook_imbalance=fv.orderbook_imbalance,
        )
        side_policy = (self.settings.signal_side_policy or self._YES_PRIORITY_POLICY).strip().upper()
        if side_policy == self._BALANCED_POLICY:
            prefer_no = effective_no > effective_yes
        else:
            prefer_no = effective_no > effective_yes and no_guard_reject is None

        if prefer_no:
            side = Side.NO
            edge = edge_no
            implied = implied_no
            fair = fair_no
            effective_edge = effective_no
            net_ev = net_no
        else:
            side = Side.YES
            edge = edge_yes
            implied = implied_yes
            fair = fair_yes
            effective_edge = effective_yes
            net_ev = net_yes
            if effective_no > effective_yes and no_guard_reject:
                self._record_no_guard_reject(no_guard_reject)

        self._record_candidate_side(side.value)
        if not self._passes_time_to_expiry(fv, edge):
            return self._reject_candidate("time_to_expiry_guard")
        if self._is_dominant_side_hard_blocked(side):
            return self._reject_candidate("dominant_side_hard_block")
        if self._mid_band_low_edge(implied, net_ev):
            return self._reject_candidate("mid_band_low_edge")
        if not self._passes_contract_price(implied):
            return self._reject_candidate("min_contract_price")
        if not self._passes_net_ev_filter(implied=implied, net_ev=net_ev):
            return self._reject_candidate("net_ev_filter")
        if not self._passes_reentry_cooldown(fv.market_id, side):
            return self._reject_candidate("reentry_cooldown")

        model_edge_score = max(0.0, min(45.0, (net_ev / max(1e-6, self.settings.signal_edge_score_scale)) * 45.0))
        liquidity_score = self._liquidity_score(fv)
        regime_score = self.model_engine.regime_score(fv)
        score = int(round(model_edge_score + liquidity_score + regime_score))
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
        self._recent_signal_sides.append(side.value)
        self._record_candidate_accept(side.value)
        return signal

    def _build_effective_edge(self, side: Side, fv: FeatureVector, raw_edge: float, model_confidence: float) -> float:
        # Confidence-aware shrinkage to reduce overreaction on noisy estimates.
        effective = raw_edge * (0.4 + 0.6 * model_confidence)
        effective = self._micro_revert_adjusted_edge(side, fv, effective)
        effective = self._trend_adjusted_edge(side, effective)
        effective = self._side_balance_adjusted_edge(side, effective)
        return effective

    def _passes_net_ev_filter(self, implied: float, net_ev: float) -> bool:
        base_min = max(0.0, self.settings.signal_min_net_ev)
        risk_min = max(0.0, self.settings.signal_min_risk_adj_edge)
        base_min = max(base_min, risk_min)
        floor = max(0.001, min(0.499, self.settings.signal_tail_prob_floor))
        ceiling = max(0.501, min(0.999, self.settings.signal_tail_prob_ceiling))
        tail_extra = max(0.0, self.settings.signal_tail_extra_net_ev)
        in_tail = implied <= floor or implied >= ceiling
        required = base_min + (tail_extra if in_tail else 0.0)
        return net_ev > required

    def _passes_contract_price(self, implied: float) -> bool:
        min_price = max(0.001, min(0.499, self.settings.signal_min_contract_price))
        return implied >= min_price

    def _passes_time_to_expiry(self, fv: FeatureVector, edge: float) -> bool:
        min_tte = max(0.0, self.settings.signal_min_tte_hours)
        if fv.time_to_expiry_hours < min_tte:
            return False
        long_tte = max(min_tte, self.settings.signal_long_tte_hours)
        if fv.time_to_expiry_hours >= long_tte:
            return abs(edge) >= max(0.0, self.settings.signal_long_tte_min_edge)
        return True

    def _mid_band_low_edge(self, implied: float, net_ev: float) -> bool:
        band = max(0.0, min(0.2, self.settings.signal_mid_band))
        if band <= 0:
            return False
        if abs(implied - 0.5) > band:
            return False
        return net_ev < self.settings.signal_mid_band_min_net_ev

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
            0.5
            + (fv.volume_1h / max(self.settings.min_hourly_volume_usd * 2.0, 1.0)),
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

    def _risk_adjusted_edge(self, raw_edge: float, fv: FeatureVector) -> float:
        vol_ref = max(1e-6, self.settings.signal_risk_vol_ref)
        spread_ref = max(1e-6, self.settings.signal_risk_spread_ref)
        vol_penalty = min(1.0, fv.volatility_30 / vol_ref)
        spread_penalty = min(1.0, fv.spread / spread_ref)
        oi_ratio = fv.open_interest / max(1.0, self.settings.min_open_interest_usd)
        vol_ratio = fv.volume_1h / max(1.0, self.settings.min_hourly_volume_usd)
        liq_score = min(1.0, 0.5 * oi_ratio + 0.5 * vol_ratio)
        liq_penalty = 1.0 - liq_score

        penalty = (
            (self.settings.signal_risk_vol_penalty_weight * vol_penalty)
            + (self.settings.signal_risk_spread_penalty_weight * spread_penalty)
            + (self.settings.signal_risk_liq_penalty_weight * liq_penalty)
        )
        penalty = max(0.0, min(0.85, penalty))
        return raw_edge * (1.0 - penalty)

    def _regime_shrunk_fair(
        self,
        fair_yes: float,
        implied_yes: float,
        fv: FeatureVector,
        model_confidence: float,
    ) -> float:
        vol_ref = max(1e-6, self.settings.signal_risk_vol_ref)
        vol_penalty = min(1.0, fv.volatility_30 / vol_ref)
        liq_factor = min(1.0, 0.5 * (fv.open_interest / max(1.0, self.settings.min_open_interest_usd))
                         + 0.5 * (fv.volume_1h / max(1.0, self.settings.min_hourly_volume_usd)))
        base = max(self.settings.signal_regime_shrink_min, min(self.settings.signal_regime_shrink_max, model_confidence))
        shrink = base * (1.0 - 0.5 * vol_penalty) * (0.6 + 0.4 * liq_factor)
        shrink = max(self.settings.signal_regime_shrink_min, min(self.settings.signal_regime_shrink_max, shrink))
        return implied_yes + ((fair_yes - implied_yes) * shrink)

    def _dynamic_min_confidence(self, fv: FeatureVector) -> float:
        base = max(0.0, self.settings.signal_min_confidence)
        vol_ref = max(1e-6, self.settings.signal_risk_vol_ref)
        vol_penalty = min(1.0, fv.volatility_30 / vol_ref)
        bump = max(0.0, self.settings.signal_vol_confidence_penalty) * vol_penalty
        return min(0.95, base + bump)

    def _evaluate_no_guard(
        self,
        net_ev: float,
        model_confidence: float,
        trend_bias: float,
        orderbook_imbalance: float,
    ) -> str | None:
        if net_ev < self.settings.signal_no_min_net_ev:
            return "no_guard_net_ev"
        if model_confidence < self.settings.signal_no_min_confidence:
            return "no_guard_confidence"
        trend_ok = trend_bias <= self.settings.signal_no_regime_trend_bias
        imbalance_ok = orderbook_imbalance <= self.settings.signal_no_regime_imbalance
        if not (trend_ok or imbalance_ok):
            return "no_guard_regime"
        total = len(self._recent_approved_sides)
        if total >= 20:
            no_count = sum(1 for side in self._recent_approved_sides if side == Side.NO.value)
            projected_ratio = (no_count + 1) / max(1, total + 1)
            ratio_cap = max(0.05, min(0.95, self.settings.signal_no_max_ratio))
            if projected_ratio > ratio_cap:
                return "no_guard_ratio"
        return None

    def _micro_revert_adjusted_edge(self, side: Side, fv: FeatureVector, effective_edge: float) -> float:
        if not self.settings.micro_revert_enabled:
            return effective_edge
        if fv.volatility_30 > self.settings.micro_revert_max_vol:
            return effective_edge
        threshold = max(0.01, self.settings.micro_revert_imb_threshold)
        if abs(fv.orderbook_imbalance) < threshold:
            return effective_edge
        preferred_side = Side.NO if fv.orderbook_imbalance > 0 else Side.YES
        boost = max(0.0, min(0.6, self.settings.micro_revert_boost))
        if side == preferred_side:
            multiplier = 1.0 + boost
        else:
            multiplier = 1.0 - (boost * 0.8)
        multiplier = max(0.35, min(1.75, multiplier))
        return effective_edge * multiplier

    def _side_balance_adjusted_edge(self, side: Side, effective_edge: float) -> float:
        if not self.settings.signal_side_balance_enabled:
            return effective_edge
        if self.settings.signal_dominant_side_hard_block:
            return effective_edge
        dominant_side, dominant_ratio = self._dominant_side_ratio()
        if dominant_side is None:
            return effective_edge
        ratio_cap = max(0.5, min(0.99, self.settings.signal_max_side_ratio))
        if dominant_ratio <= ratio_cap:
            return effective_edge
        if side.value != dominant_side:
            return effective_edge
        penalty = max(0.0, min(0.95, self.settings.signal_dominant_side_penalty))
        # Prevent near-zero collapse that can stall signal flow when penalty is set too high.
        return effective_edge * max(0.25, 1.0 - penalty)

    def _is_dominant_side_hard_blocked(self, side: Side) -> bool:
        if not self.settings.signal_side_balance_enabled:
            return False
        if not self.settings.signal_dominant_side_hard_block:
            return False
        dominant_side, dominant_ratio = self._dominant_side_ratio()
        if dominant_side is None:
            return False
        ratio_cap = max(0.5, min(0.99, self.settings.signal_max_side_ratio))
        return dominant_ratio > ratio_cap and side.value == dominant_side

    def _dominant_side_ratio(self) -> tuple[str | None, float]:
        window = max(20, self.settings.signal_side_balance_window)
        history = list(self._recent_candidate_sides)[-window:]
        total = len(history)
        if total < 20:
            return None, 0.0
        yes_count = sum(1 for value in history if value == Side.YES.value)
        no_count = total - yes_count
        dominant_side = Side.YES.value if yes_count >= no_count else Side.NO.value
        dominant_ratio = max(yes_count, no_count) / max(1, total)
        return dominant_side, dominant_ratio

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
        return None

    def _record_no_guard_reject(self, reason: str) -> None:
        if self.runtime_state is not None:
            self.runtime_state.recent_no_guard_rejects.append(
                TimedReasonEvent(timestamp=datetime.now(timezone.utc), reason=reason)
            )
