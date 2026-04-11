from __future__ import annotations

import csv
import logging
import math
import warnings
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from lightgbm import LGBMClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from polymethemoney.config import Settings
from polymethemoney.domain import (
    STRATEGY_EXPIRY_ANCHOR,
    STRATEGY_MODEL_A,
    STRATEGY_MODEL_B,
    STRATEGY_TAPE_RIDER,
    DualModelScore,
    FeatureVector,
)
from polymethemoney.storage import Store

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ModelPrediction:
    fair_probability: float
    confidence: float


@dataclass(slots=True)
class _TargetBundle:
    target: str
    model_lgbm: LGBMClassifier
    model_logit: Pipeline
    calibrator: IsotonicRegression
    weight_lgbm: float
    weight_logit: float
    version: str
    train_rows: int
    positive_rate: float


class ModelEngine:
    FEATURE_COLUMNS = [
        "implied_prob",
        "spread",
        "spread_pct",
        "volume_1h",
        "open_interest",
        "volume_oi_ratio",
        "time_to_expiry_hours",
        "orderbook_imbalance",
        "momentum_20",
        "zscore_20",
        "volatility_20",
        "volatility_30",
    ]
    CSV_COLUMNS = ["timestamp", *FEATURE_COLUMNS, "target", "sample_weight"]

    def __init__(self, settings: Settings, store: Store, training_file: Path | None = None) -> None:
        self.settings = settings
        self.store = store
        self.legacy_training_file = training_file or Path("data/training.csv")
        self.training_intraday_file = Path(self.settings.training_intraday_file)
        self.training_settlement_file = Path(self.settings.training_settlement_file)
        self.intraday_bundle: _TargetBundle | None = None
        self.settlement_bundle: _TargetBundle | None = None
        self.model_version: str = "bootstrap-heuristic"

    async def retrain_incremental(self, target: str = "both") -> None:
        await self._retrain(kind="incremental", target=target)

    async def retrain_full(self, target: str = "both") -> None:
        await self._retrain(kind="full", target=target)

    async def _retrain(self, kind: str, target: str) -> None:
        targets = self._resolve_targets(target)
        versions: dict[str, str] = {}
        for model_target in targets:
            min_rows = self._min_rows_for_target(model_target)
            rows = await self._ensure_training_data(model_target)
            if rows < min_rows:
                await self.store.record_model_result(
                    version=self.model_version,
                    kind=model_target,
                    status="skipped",
                    metrics={
                        "reason": "insufficient_rows",
                        "rows": int(rows),
                        "required": int(min_rows),
                        "retrain_kind": kind,
                    },
                    is_champion=False,
                )
                logger.warning("skip %s retrain: rows=%s required=%s", model_target, rows, min_rows)
                continue

            file_path = self._training_file_for_target(model_target)
            X, y, timestamps, sample_weight = self._load_training_data(file_path)
            if X.shape[0] < min_rows:
                await self.store.record_model_result(
                    version=self.model_version,
                    kind=model_target,
                    status="skipped",
                    metrics={
                        "reason": "insufficient_loaded_rows",
                        "rows": int(X.shape[0]),
                        "required": int(min_rows),
                        "retrain_kind": kind,
                    },
                    is_champion=False,
                )
                continue

            bundle, metrics = self._train_bundle(
                target=model_target,
                kind=kind,
                X=X,
                y=y,
                timestamps=timestamps,
                sample_weight=sample_weight,
            )
            if bundle is None:
                await self.store.record_model_result(
                    version=self.model_version,
                    kind=model_target,
                    status="skipped",
                    metrics={**metrics, "retrain_kind": kind},
                    is_champion=False,
                )
                continue

            if model_target == "intraday":
                self.intraday_bundle = bundle
            else:
                self.settlement_bundle = bundle
            versions[model_target] = bundle.version
            await self.store.record_model_result(
                version=bundle.version,
                kind=model_target,
                status="ok",
                metrics={**metrics, "retrain_kind": kind},
                is_champion=True,
            )
            logger.info("model retrained target=%s kind=%s version=%s metrics=%s", model_target, kind, bundle.version, metrics)

        self.model_version = self._compose_version(versions)

    def predict(self, fv: FeatureVector) -> ModelPrediction:
        score = self.predict_dual_score(fv)
        return ModelPrediction(fair_probability=score.blended_prob, confidence=score.confidence)

    def predict_dual_score(self, fv: FeatureVector) -> DualModelScore:
        intraday_up_prob, intraday_conf = self._predict_target(fv, self.intraday_bundle, "intraday")
        settlement_prob, settlement_conf = self._predict_target(fv, self.settlement_bundle, "settlement")

        settlement_weight = self._settlement_weight(
            time_to_expiry_hours=fv.time_to_expiry_hours,
            settlement_confidence=settlement_conf,
            settlement_rows=self.settlement_bundle.train_rows if self.settlement_bundle else 0,
        )
        settlement_anchor = (fv.implied_prob * (1.0 - settlement_weight)) + (settlement_prob * settlement_weight)
        intraday_base_rate = self.intraday_bundle.positive_rate if self.intraday_bundle else 0.5
        intraday_fair = self._intraday_fair_probability(fv, intraday_up_prob, intraday_base_rate)
        trend_bias = self._trend_bias(fv)
        trend_strength = self._trend_strength(fv)
        trend_anchor = self._trend_anchor_probability(fv, trend_bias, trend_strength)
        settlement_anchor = self._blend_probability(
            settlement_anchor,
            trend_anchor,
            min(0.72, 0.20 + (0.20 * trend_strength)),
        )
        intraday_fair = self._blend_probability(
            intraday_fair,
            trend_anchor,
            min(0.88, 0.35 + (0.35 * trend_strength) + (0.10 * (1.0 - settlement_weight))),
        )
        blended = (settlement_anchor * settlement_weight) + (intraday_fair * (1.0 - settlement_weight))
        blended = self._blend_probability(
            blended,
            trend_anchor,
            min(0.85, 0.25 + (0.35 * trend_strength)),
        )

        disagreement = abs(settlement_anchor - intraday_fair)
        dist = abs(blended - fv.implied_prob)
        base_conf = 0.65 * intraday_conf + 0.35 * settlement_conf
        confidence = base_conf * (1.0 - min(1.0, disagreement * 1.5)) * min(1.0, 0.25 + (dist / 0.14))
        confidence = confidence * (0.60 + 0.40 * trend_strength) + (0.08 * abs(trend_bias))
        confidence = float(max(0.05, min(0.99, confidence)))
        return DualModelScore(
            settlement_prob=float(max(0.001, min(0.999, settlement_anchor))),
            intraday_prob=float(max(0.001, min(0.999, intraday_fair))),
            blended_prob=float(max(0.001, min(0.999, blended))),
            confidence=confidence,
        )

    def predict_fair_probability(self, fv: FeatureVector) -> float:
        return self.predict(fv).fair_probability

    def predict_strategy(self, strategy_id: str, fv: FeatureVector) -> ModelPrediction:
        if strategy_id == STRATEGY_EXPIRY_ANCHOR:
            return self.predict_expiry_anchor(fv)
        if strategy_id == STRATEGY_TAPE_RIDER:
            return self.predict_tape_rider(fv)
        if strategy_id == STRATEGY_MODEL_B:
            return self.predict_model_b(fv)
        return self.predict_model_a(fv)

    def predict_expiry_anchor(self, fv: FeatureVector) -> ModelPrediction:
        base = self.predict_dual_score(fv)
        derived = self.derive_strategy_features(fv)
        bias = 0.04 * math.tanh(
            (1.4 * float(fv.orderbook_imbalance))
            + (0.9 * float(fv.volume_oi_ratio))
            - (0.7 * float(derived["spread_to_tte"]))
            + (0.5 * float(derived["mid_prob_flag"]))
        )
        fair = self._clip_probability(
            (0.65 * float(base.settlement_prob))
            + (0.20 * float(base.intraday_prob))
            + (0.15 * float(fv.implied_prob))
            + bias
        )
        confidence = self._clip_probability(
            (0.45 * float(base.confidence))
            + 0.25
            + (0.15 * (1.0 - float(derived["spread_to_tte"])))
            + (0.15 * float(derived["mid_prob_flag"]))
        )
        return ModelPrediction(fair_probability=fair, confidence=confidence)

    def predict_tape_rider(self, fv: FeatureVector) -> ModelPrediction:
        derived = self.derive_strategy_features(fv)
        bias = 0.12 * math.tanh(
            (2.6 * float(derived["momentum_vol_adj"]))
            + (1.4 * float(derived["imbalance_momentum_align"]))
            + (0.9 * float(derived["vol_ratio_20_30"]))
            + (0.6 * float(derived["volume_pressure"]))
            - (0.7 * float(derived["spread_to_tte"]))
        )
        fair = self._clip_probability(float(fv.implied_prob) + bias)
        shock_flag = float(derived["shock_flag"])
        confidence = 0.20 + (0.35 * abs(bias) / 0.12) + (0.15 * abs(float(derived["imbalance_momentum_align"]))) + (0.10 * (1.0 - shock_flag))
        confidence = float(max(0.05, min(0.99, confidence)))
        return ModelPrediction(fair_probability=fair, confidence=confidence)

    def predict_model_a(self, fv: FeatureVector) -> ModelPrediction:
        base = self.predict_dual_score(fv)
        derived = self.derive_strategy_features(fv)
        weights = {
            "dist_from_mid": self.settings.model_a_dist_from_mid_weight,
            "abs_dist_from_mid": self.settings.model_a_abs_dist_from_mid_weight,
            "spread_to_mid": self.settings.model_a_spread_to_mid_weight,
            "spread_to_tte": self.settings.model_a_spread_to_tte_weight,
            "log_volume_1h": self.settings.model_a_log_volume_1h_weight,
            "log_open_interest": self.settings.model_a_log_open_interest_weight,
            "volume_pressure": self.settings.model_a_volume_pressure_weight,
            "imbalance_abs": self.settings.model_a_imbalance_abs_weight,
            "imbalance_momentum_align": self.settings.model_a_imbalance_momentum_align_weight,
            "imbalance_zscore_align": self.settings.model_a_imbalance_zscore_align_weight,
            "momentum_abs": self.settings.model_a_momentum_abs_weight,
            "zscore_abs": self.settings.model_a_zscore_abs_weight,
            "vol_ratio_20_30": self.settings.model_a_vol_ratio_20_30_weight,
            "momentum_vol_adj": self.settings.model_a_momentum_vol_adj_weight,
            "zscore_vol_adj": self.settings.model_a_zscore_vol_adj_weight,
            "expiry_pressure": self.settings.model_a_expiry_pressure_weight,
            "tail_prob_flag": self.settings.model_a_tail_prob_flag_weight,
            "mid_prob_flag": self.settings.model_a_mid_prob_flag_weight,
            "revert_pressure": self.settings.model_a_revert_pressure_weight,
            "shock_flag": self.settings.model_a_shock_flag_weight,
        }
        adjustment = sum(float(weights[name]) * float(derived[name]) for name in weights)
        fair = self._clip_probability(float(base.blended_prob) + adjustment)
        confidence = float(base.confidence)
        confidence += min(0.16, abs(adjustment) * 2.6)
        confidence -= min(0.05, float(derived["shock_flag"]) * 0.05)
        confidence = float(max(0.05, min(0.99, confidence)))
        return ModelPrediction(fair_probability=fair, confidence=confidence)

    def predict_model_b(self, fv: FeatureVector) -> ModelPrediction:
        derived = self.derive_strategy_features(fv)
        implied = self._clip_probability(float(fv.implied_prob))
        direction = 1.0 if implied >= 0.5 else -1.0

        market_prior = implied
        trend_prob = self._clip_probability(
            implied
            + (0.13 * math.tanh(
                (2.4 * float(derived["momentum_vol_adj"]))
                + (1.3 * float(derived["imbalance_momentum_align"]))
                + (0.9 * float(derived["vol_ratio_20_30"]))
            ))
        )
        mean_revert_prob = self._clip_probability(
            implied
            + (0.11 * math.tanh(
                (-1.4 * float(derived["zscore_vol_adj"]))
                + (1.2 * float(derived["revert_pressure"]))
                - (0.8 * float(derived["shock_flag"]))
            ))
        )
        expiry_prob = self._clip_probability(
            (implied * (1.0 - (0.28 * float(derived["expiry_pressure"]))))
            + (0.5 * (0.28 * float(derived["expiry_pressure"])))
            - (0.04 * float(derived["spread_to_tte"]))
            + (0.025 * float(derived["mid_prob_flag"]) * direction)
        )
        flow_strength = 0.10 * math.tanh(
            (1.2 * float(derived["volume_pressure"]))
            + (0.9 * float(derived["log_volume_1h"]))
            + (0.7 * float(derived["log_open_interest"]))
            + (0.6 * float(derived["imbalance_abs"]))
        )
        flow_prob = self._clip_probability(implied + (direction * flow_strength))

        probabilities = np.array(
            [market_prior, trend_prob, mean_revert_prob, expiry_prob, flow_prob],
            dtype=np.float64,
        )
        weights = np.array([0.30, 0.23, 0.17, 0.15, 0.15], dtype=np.float64)
        fair = self._clip_probability(float(np.average(probabilities, weights=weights)))

        dispersion = float(np.std(probabilities))
        edge = abs(fair - implied)
        confidence = 0.18 + min(0.42, edge / 0.12) + max(0.0, 0.30 - min(0.30, dispersion * 1.8))
        confidence = float(max(0.05, min(0.99, confidence)))
        return ModelPrediction(fair_probability=fair, confidence=confidence)

    def derive_strategy_features(self, fv: FeatureVector) -> dict[str, float]:
        implied = self._clip_probability(float(fv.implied_prob))
        dist_from_mid = self._clip_symmetric((implied - 0.5) / 0.25, 1.0)
        spread_to_mid = max(0.0, min(1.0, float(fv.spread) / max(0.02, implied * 0.08)))
        spread_to_tte = max(
            0.0,
            min(1.0, float(fv.spread) / max(0.002, max(float(fv.time_to_expiry_hours), 0.01) * 0.08)),
        )
        log_volume_1h = max(0.0, min(1.0, math.log1p(max(0.0, float(fv.volume_1h))) / 12.0))
        log_open_interest = max(0.0, min(1.0, math.log1p(max(0.0, float(fv.open_interest))) / 14.0))
        volume_pressure = self._clip_symmetric((float(fv.volume_oi_ratio) - 0.15) / 0.25, 1.0)
        imbalance = self._clip_symmetric(float(fv.orderbook_imbalance), 1.0)
        momentum_abs = max(0.0, min(1.0, abs(float(fv.momentum_20)) / 0.03))
        zscore_abs = max(0.0, min(1.0, abs(float(fv.zscore_20)) / 3.0))
        vol_ratio_raw = float(fv.volatility_20) / max(float(fv.volatility_30), 1e-6)
        vol_ratio_20_30 = self._clip_symmetric((vol_ratio_raw - 1.0) / 0.75, 1.0)
        momentum_vol_adj = self._clip_symmetric(float(fv.momentum_20) / max(float(fv.volatility_30) * 3.0, 0.01), 1.0)
        zscore_vol_adj = self._clip_symmetric((float(fv.zscore_20) / 3.0) * max(float(fv.volatility_30) / 0.04, 0.25), 1.0)
        expiry_pressure = max(0.0, min(1.0, math.exp(-max(float(fv.time_to_expiry_hours), 0.0) / 0.20)))
        tail_prob_flag = 1.0 if abs(implied - 0.5) >= 0.35 else 0.0
        mid_prob_flag = 1.0 if abs(implied - 0.5) <= 0.08 else 0.0
        imbalance_momentum_align = self._clip_symmetric(imbalance * math.tanh(float(fv.momentum_20) / 0.02), 1.0)
        imbalance_zscore_align = self._clip_symmetric(imbalance * math.tanh(float(fv.zscore_20) / 2.5), 1.0)
        revert_pressure = self._clip_symmetric(
            (-math.tanh(float(fv.zscore_20) / 2.0)) * (1.0 - (0.5 * momentum_abs)),
            1.0,
        )
        shock_flag = 1.0 if (float(fv.volatility_20) > float(fv.volatility_30) * 1.35) or (momentum_abs > 0.85) else 0.0

        return {
            "dist_from_mid": dist_from_mid,
            "abs_dist_from_mid": abs(dist_from_mid),
            "spread_to_mid": spread_to_mid,
            "spread_to_tte": spread_to_tte,
            "log_volume_1h": log_volume_1h,
            "log_open_interest": log_open_interest,
            "volume_pressure": volume_pressure,
            "imbalance_abs": abs(imbalance),
            "imbalance_momentum_align": imbalance_momentum_align,
            "imbalance_zscore_align": imbalance_zscore_align,
            "momentum_abs": momentum_abs,
            "zscore_abs": zscore_abs,
            "vol_ratio_20_30": vol_ratio_20_30,
            "momentum_vol_adj": momentum_vol_adj,
            "zscore_vol_adj": zscore_vol_adj,
            "expiry_pressure": expiry_pressure,
            "tail_prob_flag": tail_prob_flag,
            "mid_prob_flag": mid_prob_flag,
            "revert_pressure": revert_pressure,
            "shock_flag": shock_flag,
        }

    def regime_score(self, fv: FeatureVector) -> float:
        vol_factor = min(1.0, fv.volatility_30 / 0.10)
        expiry_factor = min(1.0, fv.time_to_expiry_hours / 168.0)
        trend_factor = 0.55 + (0.45 * abs(self._trend_bias(fv)))
        stable = max(0.0, 1.0 - 0.7 * vol_factor + 0.3 * expiry_factor)
        return max(0.0, min(25.0, 25.0 * stable * trend_factor))

    async def _ensure_training_data(self, target: str) -> int:
        file_path = self._training_file_for_target(target)
        if file_path.exists():
            X, _, _, _ = self._load_training_data(file_path)
            if X.shape[0] >= self._min_rows_for_target(target):
                return int(X.shape[0])

        if target == "intraday" and self.settings.auto_build_training_from_db:
            generated_rows = await self._rebuild_intraday_from_runtime_features(file_path)
            if generated_rows > 0:
                X, _, _, _ = self._load_training_data(file_path)
                return int(X.shape[0])
        if target == "settlement" and self.settings.auto_build_training_from_db:
            generated_rows = await self._rebuild_settlement_from_resolved_features(file_path)
            if generated_rows > 0:
                X, _, _, _ = self._load_training_data(file_path)
                return int(X.shape[0])

        # Legacy fallback for intraday only.
        if target == "intraday" and self.legacy_training_file.exists():
            X, _, _, _ = self._load_training_data(self.legacy_training_file)
            if X.shape[0] > 0:
                file_path.parent.mkdir(parents=True, exist_ok=True)
                file_path.write_text(self.legacy_training_file.read_text(encoding="utf-8"), encoding="utf-8")
                return int(X.shape[0])

        return 0

    async def _rebuild_intraday_from_runtime_features(self, file_path: Path) -> int:
        rows = await self.store.build_training_rows_from_features(
            lookback_days=self.settings.training_lookback_days,
            horizon_minutes=self.settings.training_horizon_minutes,
            min_move=self.settings.training_min_move,
            max_spread_pct=self.settings.training_max_spread_pct,
            min_volume_oi_ratio=self.settings.training_min_vol_oi_ratio,
            min_tte_hours=self.settings.training_min_tte_hours,
            max_sample_weight=self.settings.training_max_weight,
        )
        if len(rows) < self.settings.model_min_rows:
            relaxed = await self.store.build_training_rows_from_features(
                lookback_days=self.settings.training_lookback_days,
                horizon_minutes=self.settings.training_horizon_minutes,
                min_move=0.0,
                max_spread_pct=self.settings.training_max_spread_pct,
                min_volume_oi_ratio=self.settings.training_min_vol_oi_ratio,
                min_tte_hours=self.settings.training_min_tte_hours,
                max_sample_weight=self.settings.training_max_weight,
            )
            if len(relaxed) > len(rows):
                rows = self._rebalance_rows(relaxed, max_negative_multiplier=8.0)
        if not rows:
            return 0
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    async def _rebuild_settlement_from_resolved_features(self, file_path: Path) -> int:
        rows = await self.store.build_training_rows_from_settlements(
            lookback_days=self.settings.training_lookback_days,
            max_spread_pct=self.settings.training_max_spread_pct,
            min_volume_oi_ratio=self.settings.training_min_vol_oi_ratio,
            min_tte_hours=self.settings.training_min_tte_hours,
            max_sample_weight=self.settings.training_max_weight,
        )
        if len(rows) < self._min_rows_for_target("settlement"):
            relaxed = await self.store.build_training_rows_from_settlements(
                lookback_days=self.settings.training_lookback_days,
                max_spread_pct=self.settings.training_max_spread_pct,
                min_volume_oi_ratio=None,
                min_tte_hours=None,
                max_sample_weight=self.settings.training_max_weight,
            )
            if len(relaxed) > len(rows):
                rows = relaxed
        if not rows:
            return 0
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.CSV_COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def _train_bundle(
        self,
        target: str,
        kind: str,
        X: np.ndarray,
        y: np.ndarray,
        timestamps: np.ndarray,
        sample_weight: np.ndarray,
    ) -> tuple[_TargetBundle | None, dict]:
        order = np.argsort(timestamps)
        X = X[order]
        y = y[order]
        timestamps = timestamps[order]
        sample_weight = sample_weight[order]
        sample_weight = self._apply_recency_weight(sample_weight, timestamps, target)

        folds = self._build_walk_forward_folds(timestamps)
        required_folds = 2
        if target == "settlement":
            required_folds = 1
        if len(folds) < required_folds:
            folds = self._build_fallback_folds(len(y))
        if len(folds) < required_folds and target == "settlement":
            holdout = self._build_holdout_fold(len(y), train_ratio=0.8)
            if holdout is not None:
                folds = [holdout]
        if len(folds) < required_folds:
            return None, {"reason": "insufficient_folds", "rows": int(len(y))}

        oof_lgbm = np.full(len(y), np.nan, dtype=np.float64)
        oof_logit = np.full(len(y), np.nan, dtype=np.float64)
        min_train_rows = 120 if target != "settlement" else 20
        min_valid_rows = 40 if target != "settlement" else 10
        for train_idx, valid_idx in folds:
            if len(train_idx) < min_train_rows or len(valid_idx) < min_valid_rows:
                continue
            x_train = X[train_idx]
            y_train = y[train_idx]
            w_train = sample_weight[train_idx]
            x_valid = X[valid_idx]
            model_lgbm = self._new_lgbm(kind)
            model_lgbm.fit(x_train, y_train, sample_weight=w_train)
            model_logit = self._new_logit()
            model_logit.fit(x_train, y_train, logit__sample_weight=w_train)
            oof_lgbm[valid_idx] = self._predict_lgbm(model_lgbm, x_valid)[:, 1]
            oof_logit[valid_idx] = model_logit.predict_proba(x_valid)[:, 1]

        valid_mask = (~np.isnan(oof_lgbm)) & (~np.isnan(oof_logit))
        if target == "settlement":
            min_oof_rows = 10
        else:
            min_oof_rows = max(100, self._min_rows_for_target(target) // 2)
        if int(np.sum(valid_mask)) < min_oof_rows:
            return None, {"reason": "insufficient_oof", "rows": int(np.sum(valid_mask))}

        y_valid = y[valid_mask]
        lgbm_pred = np.clip(oof_lgbm[valid_mask], 1e-6, 1 - 1e-6)
        logit_pred = np.clip(oof_logit[valid_mask], 1e-6, 1 - 1e-6)
        lgbm_loss = float(log_loss(y_valid, lgbm_pred))
        logit_loss = float(log_loss(y_valid, logit_pred))
        weight_lgbm, weight_logit = self._inverse_loss_weights(lgbm_loss, logit_loss)
        blended_valid = np.clip(
            (weight_lgbm * lgbm_pred) + (weight_logit * logit_pred),
            1e-6,
            1 - 1e-6,
        )
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(blended_valid, y_valid, sample_weight=sample_weight[valid_mask])
        calibrated_valid = np.clip(np.asarray(calibrator.predict(blended_valid), dtype=np.float64), 1e-6, 1 - 1e-6)

        final_lgbm = self._new_lgbm(kind)
        final_lgbm.fit(X, y, sample_weight=sample_weight)
        final_logit = self._new_logit()
        final_logit.fit(X, y, logit__sample_weight=sample_weight)
        version = f"{target}-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}"
        bundle = _TargetBundle(
            target=target,
            model_lgbm=final_lgbm,
            model_logit=final_logit,
            calibrator=calibrator,
            weight_lgbm=weight_lgbm,
            weight_logit=weight_logit,
            version=version,
            train_rows=int(X.shape[0]),
            positive_rate=float(np.mean(y)),
        )
        metrics = {
            "rows": int(X.shape[0]),
            "oof_rows": int(np.sum(valid_mask)),
            "folds": len(folds),
            "lgbm_logloss": round(lgbm_loss, 6),
            "logit_logloss": round(logit_loss, 6),
            "ensemble_logloss": round(float(log_loss(y_valid, calibrated_valid)), 6),
            "brier": round(float(np.mean((calibrated_valid - y_valid) ** 2)), 6),
            "ece": round(self._expected_calibration_error(calibrated_valid, y_valid, bins=12), 6),
            "weight_lgbm": round(weight_lgbm, 4),
            "weight_logit": round(weight_logit, 4),
            "sample_weight_mean": round(float(np.mean(sample_weight)), 6),
            "sample_weight_max": round(float(np.max(sample_weight)), 6),
            "trend_recency_half_life_days": float(self.settings.trend_recency_half_life_days),
            "train_min_ts": datetime.fromtimestamp(float(np.min(timestamps)), tz=timezone.utc).isoformat(),
            "train_max_ts": datetime.fromtimestamp(float(np.max(timestamps)), tz=timezone.utc).isoformat(),
        }
        return bundle, metrics

    def _predict_target(self, fv: FeatureVector, bundle: _TargetBundle | None, target: str) -> tuple[float, float]:
        if bundle is None:
            if target == "intraday":
                return self._heuristic_intraday_probability(fv), 0.35
            return self._heuristic_settlement_probability(fv), 0.25
        x = np.array([[self._feature_value(fv, name) for name in self.FEATURE_COLUMNS]], dtype=np.float64)
        raw_lgbm = float(self._predict_lgbm(bundle.model_lgbm, x)[0, 1])
        raw_logit = float(bundle.model_logit.predict_proba(x)[0, 1])
        raw_blended = bundle.weight_lgbm * raw_lgbm + bundle.weight_logit * raw_logit
        calibrated = float(bundle.calibrator.predict([raw_blended])[0])
        if target == "settlement":
            # Settlement model often has much fewer rows than intraday.
            # Shrink noisy outputs toward current implied probability.
            row_reliability = min(1.0, bundle.train_rows / 1000.0)
            calibrated = fv.implied_prob + ((calibrated - fv.implied_prob) * row_reliability)
        disagreement = abs(raw_lgbm - raw_logit)
        confidence = (1.0 - min(1.0, disagreement * 2.0)) * min(1.0, 0.25 + abs(calibrated - fv.implied_prob) / 0.14)
        return float(max(0.001, min(0.999, calibrated))), float(max(0.05, min(0.99, confidence)))

    def _heuristic_intraday_probability(self, fv: FeatureVector) -> float:
        trend_bias = self._trend_bias(fv)
        trend_strength = self._trend_strength(fv)
        liquidity_scale = 0.50 + min(0.50, float(fv.volume_1h) / max(1.0, float(fv.open_interest) + 1.0))
        expiry_scale = max(0.40, min(1.0, 1.0 - (float(fv.time_to_expiry_hours) / 24.0)))
        shift = trend_bias * 0.12 * liquidity_scale * expiry_scale
        fair = fv.implied_prob + shift + (trend_bias * trend_strength * 0.02)
        return max(0.001, min(0.999, fair))

    def _heuristic_settlement_probability(self, fv: FeatureVector) -> float:
        trend_bias = self._trend_bias(fv)
        trend_strength = self._trend_strength(fv)
        vol_dampen = max(0.45, 1.0 - min(1.0, fv.volatility_30 / 0.07))
        fair = fv.implied_prob + (trend_bias * 0.07 * vol_dampen) + (trend_bias * trend_strength * 0.015)
        return max(0.001, min(0.999, fair))

    @staticmethod
    def _settlement_weight(time_to_expiry_hours: float, settlement_confidence: float, settlement_rows: int) -> float:
        if time_to_expiry_hours <= 0:
            return 0.05
        row_factor = min(1.0, settlement_rows / 1500.0)
        conf_factor = max(0.2, min(1.0, settlement_confidence))
        time_factor = max(0.15, min(0.75, time_to_expiry_hours / 120.0))
        return float(max(0.05, min(0.60, time_factor * row_factor * conf_factor)))

    @staticmethod
    def _intraday_fair_probability(fv: FeatureVector, intraday_up_prob: float, baseline_up_rate: float) -> float:
        # Intraday model predicts short-term up/down direction, not absolute event probability.
        # Convert to a bounded drift around current implied probability.
        baseline = max(0.05, min(0.95, baseline_up_rate))
        centered = max(0.0, min(1.0, intraday_up_prob)) - baseline
        norm = max(0.10, max(baseline, 1.0 - baseline))
        directional = max(-1.0, min(1.0, centered / norm))
        liquidity_scale = min(1.0, 0.4 + (fv.volume_1h / max(1.0, fv.open_interest + 1.0)))
        vol_penalty = max(0.5, 1.0 - min(1.0, fv.volatility_30 / 0.06))
        max_shift = 0.09
        shift = directional * max_shift * liquidity_scale * vol_penalty
        return max(0.001, min(0.999, fv.implied_prob + shift))

    def _trend_bias(self, fv: FeatureVector) -> float:
        vol_20 = max(0.001, float(fv.volatility_20))
        vol_30 = max(0.001, float(fv.volatility_30))
        momentum = math.tanh(float(fv.momentum_20) / max(0.0025, vol_20 * 1.8))
        imbalance = math.tanh(max(-1.0, min(1.0, float(fv.orderbook_imbalance))) * 1.3)
        zscore = math.tanh(float(fv.zscore_20) / 2.5)
        acceleration = math.tanh((float(fv.momentum_20) - (float(fv.zscore_20) * 0.01)) / max(0.003, vol_30 * 2.2))
        raw = (0.48 * momentum) + (0.26 * imbalance) + (0.16 * zscore) + (0.10 * acceleration)
        return max(-1.0, min(1.0, raw))

    def _trend_strength(self, fv: FeatureVector) -> float:
        bias = abs(self._trend_bias(fv))
        liquidity = min(1.0, float(fv.volume_1h) / max(1.0, float(fv.open_interest) + 1.0))
        expiry_scale = max(0.35, min(1.0, 1.0 - (float(fv.time_to_expiry_hours) / 24.0)))
        return max(0.0, min(1.0, (0.60 * bias) + (0.20 * liquidity) + (0.20 * expiry_scale)))

    @staticmethod
    def _blend_probability(base: float, anchor: float, weight: float) -> float:
        w = max(0.0, min(1.0, weight))
        return (base * (1.0 - w)) + (anchor * w)

    def _trend_anchor_probability(self, fv: FeatureVector, trend_bias: float, trend_strength: float) -> float:
        liquidity_scale = 0.45 + min(0.55, float(fv.volume_1h) / max(1.0, float(fv.open_interest) + 1.0))
        expiry_scale = max(0.35, min(1.0, 1.0 - (float(fv.time_to_expiry_hours) / 18.0)))
        vol_scale = max(0.40, 1.0 - min(1.0, float(fv.volatility_30) / 0.12))
        max_shift = 0.11 * liquidity_scale * expiry_scale * vol_scale
        anchor = float(fv.implied_prob) + (trend_bias * max_shift) + (trend_bias * trend_strength * 0.025)
        return max(0.001, min(0.999, anchor))

    def _training_file_for_target(self, target: str) -> Path:
        if target == "settlement":
            return self.training_settlement_file
        return self.training_intraday_file

    def _min_rows_for_target(self, target: str) -> int:
        if target == "settlement":
            return max(30, self.settings.model_min_rows // 16)
        return self.settings.model_min_rows

    @staticmethod
    def _resolve_targets(target: str) -> list[str]:
        t = target.strip().lower()
        if t in {"both", "all"}:
            return ["intraday", "settlement"]
        if t in {"intraday", "settlement"}:
            return [t]
        return ["intraday", "settlement"]

    def _compose_version(self, versions: dict[str, str]) -> str:
        intraday = versions.get("intraday", self.intraday_bundle.version if self.intraday_bundle else "none")
        settlement = versions.get("settlement", self.settlement_bundle.version if self.settlement_bundle else "none")
        return f"intraday:{intraday}|settlement:{settlement}"

    @staticmethod
    def _rebalance_rows(rows: list[dict[str, float | int | str]], max_negative_multiplier: float) -> list[dict[str, float | int | str]]:
        positives = [row for row in rows if int(row["target"]) == 1]
        negatives = [row for row in rows if int(row["target"]) == 0]
        if not positives or not negatives:
            return rows
        max_negatives = int(len(positives) * max_negative_multiplier)
        if len(negatives) <= max_negatives:
            return rows
        rng = np.random.default_rng(42)
        indices = rng.choice(len(negatives), size=max_negatives, replace=False)
        selected_neg = [negatives[int(idx)] for idx in indices]
        reduced = [*positives, *selected_neg]
        reduced.sort(key=lambda row: str(row.get("timestamp", "")))
        return reduced

    def _feature_value(self, fv: FeatureVector, name: str) -> float:
        raw = float(asdict(fv)[name])
        return self._clip_feature(name, raw)

    def _load_training_data(self, file_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        rows_X: list[list[float]] = []
        rows_y: list[float] = []
        rows_ts: list[float] = []
        rows_w: list[float] = []
        with file_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            for idx, row in enumerate(reader):
                try:
                    features = [self._clip_feature(c, float(row[c])) for c in self.FEATURE_COLUMNS]
                    target = float(row["target"])
                except (KeyError, TypeError, ValueError):
                    continue
                ts = self._parse_timestamp(row.get("timestamp"), fallback=float(idx))
                weight = self._parse_weight(row.get("sample_weight"))
                rows_X.append(features)
                rows_y.append(target)
                rows_ts.append(ts)
                rows_w.append(weight)
        if not rows_X:
            empty = np.empty((0,), dtype=np.float64)
            return (
                np.empty((0, len(self.FEATURE_COLUMNS)), dtype=np.float64),
                empty,
                empty,
                empty,
            )
        return (
            np.array(rows_X, dtype=np.float64),
            np.array(rows_y, dtype=np.float64),
            np.array(rows_ts, dtype=np.float64),
            np.array(rows_w, dtype=np.float64),
        )

    @staticmethod
    def _parse_timestamp(raw: str | None, fallback: float) -> float:
        if not raw:
            return fallback
        text = str(raw).strip()
        if not text:
            return fallback
        try:
            return float(text)
        except ValueError:
            pass
        try:
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except ValueError:
            return fallback

    @staticmethod
    def _parse_weight(raw: str | None) -> float:
        if raw is None:
            return 1.0
        try:
            value = float(raw)
        except (TypeError, ValueError):
            return 1.0
        return max(0.1, min(10.0, value))

    def _clip_feature(self, name: str, value: float) -> float:
        if name == "spread_pct":
            return min(value, self.settings.feature_clip_spread_pct_max)
        if name == "volume_oi_ratio":
            return min(value, self.settings.feature_clip_vol_oi_max)
        if name == "momentum_20":
            return self._clip_symmetric(value, self.settings.feature_clip_momentum_max)
        if name == "zscore_20":
            return self._clip_symmetric(value, self.settings.feature_clip_zscore_max)
        if name == "volatility_20":
            return min(value, self.settings.feature_clip_vol20_max)
        if name == "volatility_30":
            return min(value, self.settings.feature_clip_vol30_max)
        return value

    @staticmethod
    def _clip_symmetric(value: float, bound: float) -> float:
        limit = max(0.0, float(bound))
        if limit <= 0:
            return value
        return max(-limit, min(limit, value))

    @staticmethod
    def _clip_probability(value: float) -> float:
        return float(max(0.001, min(0.999, value)))

    def _build_walk_forward_folds(self, timestamps: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
        if timestamps.size == 0:
            return []
        min_ts = float(np.min(timestamps))
        max_ts = float(np.max(timestamps))
        day_sec = 86400.0
        train_span = max(7.0, float(self.settings.train_days)) * day_sec
        valid_span = max(1.0, float(self.settings.valid_days)) * day_sec
        if max_ts - min_ts < (train_span + valid_span):
            return []
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        cursor = min_ts + train_span
        end_cursor = max_ts - valid_span
        while cursor <= end_cursor:
            train_start = cursor - train_span
            train_mask = (timestamps >= train_start) & (timestamps < cursor)
            valid_mask = (timestamps >= cursor) & (timestamps < (cursor + valid_span))
            train_idx = np.where(train_mask)[0]
            valid_idx = np.where(valid_mask)[0]
            if train_idx.size > 0 and valid_idx.size > 0:
                folds.append((train_idx, valid_idx))
            cursor += valid_span
        return folds

    @staticmethod
    def _build_fallback_folds(n_rows: int) -> list[tuple[np.ndarray, np.ndarray]]:
        if n_rows < 240:
            return []
        folds: list[tuple[np.ndarray, np.ndarray]] = []
        split_points = [0.70, 0.78, 0.86]
        for ratio in split_points:
            split = int(n_rows * ratio)
            # Short-horizon BTC datasets are often compact; keep a slightly
            # larger fallback validation window so OOF calibration can proceed.
            valid_end = min(n_rows, split + max(40, int(n_rows * 0.10)))
            if split < 120 or valid_end - split < 40:
                continue
            train_idx = np.arange(0, split)
            valid_idx = np.arange(split, valid_end)
            folds.append((train_idx, valid_idx))
        return folds

    @staticmethod
    def _build_holdout_fold(n_rows: int, train_ratio: float) -> tuple[np.ndarray, np.ndarray] | None:
        if n_rows < 30:
            return None
        split = int(n_rows * train_ratio)
        split = max(20, min(split, n_rows - 10))
        train_idx = np.arange(0, split)
        valid_idx = np.arange(split, n_rows)
        if train_idx.size < 20 or valid_idx.size < 10:
            return None
        return train_idx, valid_idx

    @staticmethod
    def _new_logit() -> Pipeline:
        return Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("logit", LogisticRegression(max_iter=2000, class_weight="balanced")),
            ]
        )

    @staticmethod
    def _new_lgbm(kind: str) -> LGBMClassifier:
        return LGBMClassifier(
            objective="binary",
            n_estimators=320 if kind == "full" else 180,
            learning_rate=0.04,
            max_depth=-1,
            num_leaves=31,
            min_child_samples=25,
            subsample=0.85,
            colsample_bytree=0.8,
            reg_alpha=0.2,
            reg_lambda=0.6,
            random_state=42,
            verbosity=-1,
        )

    @staticmethod
    def _predict_lgbm(model: LGBMClassifier, X: np.ndarray) -> np.ndarray:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="X does not have valid feature names, but LGBMClassifier was fitted with feature names",
            )
            return model.predict_proba(X)

    @staticmethod
    def _inverse_loss_weights(loss_a: float, loss_b: float) -> tuple[float, float]:
        inv_a = 1.0 / max(loss_a, 1e-6)
        inv_b = 1.0 / max(loss_b, 1e-6)
        total = inv_a + inv_b
        if total <= 0:
            return 0.5, 0.5
        return inv_a / total, inv_b / total

    @staticmethod
    def _expected_calibration_error(prob: np.ndarray, y_true: np.ndarray, bins: int = 10) -> float:
        edges = np.linspace(0, 1, bins + 1)
        ece = 0.0
        n = len(prob)
        if n == 0:
            return 1.0
        for i in range(bins):
            left = edges[i]
            right = edges[i + 1]
            mask = (prob >= left) & (prob < right if i < bins - 1 else prob <= right)
            if not np.any(mask):
                continue
            conf = float(np.mean(prob[mask]))
            acc = float(np.mean(y_true[mask]))
            weight = float(np.sum(mask) / n)
            ece += abs(conf - acc) * weight
        return ece

    def _apply_recency_weight(self, sample_weight: np.ndarray, timestamps: np.ndarray, target: str) -> np.ndarray:
        if sample_weight.size == 0 or timestamps.size == 0:
            return sample_weight
        half_life_days = max(0.25, float(self.settings.trend_recency_half_life_days))
        if target == "settlement":
            half_life_days = half_life_days * 2.0
        max_ts = float(np.max(timestamps))
        age_days = np.maximum(0.0, (max_ts - timestamps) / 86400.0)
        recency_factor = np.power(0.5, age_days / half_life_days)
        weighted = sample_weight * np.clip(recency_factor, 0.05, 1.25)
        return np.clip(weighted, 0.05, 20.0)
