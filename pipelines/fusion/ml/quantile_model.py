"""LightGBM quantile regression calibration head.

Sits on top of the structural prior. Inputs are the same feature vector that
the structural estimator uses, plus the structural estimate itself. The model
learns residual corrections and provides calibrated prediction intervals.

Three models are trained simultaneously:
  - q10: 10th percentile (conservative lower bound)
  - q50: median (point estimate)
  - q90: 90th percentile (optimistic upper bound)

The gap between q10 and q90 is the uncertainty band. If the model is well-
calibrated, ~80% of true values should fall within this band.

Usage:
  trainer = QuantileTrainer()
  trainer.train(df)
  trainer.save(path)

  model = QuantileModel.load(path)
  lo, mid, hi = model.predict(feature_dict)
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


FEATURE_COLS = [
    "sdi",
    "sku_diversity",
    "inventory_value",
    "refill_signal",
    "store_size_sqft",
    "pos_sophistication",
    "coverage_score",
    "extraction_confidence",
    "geo_demand_score",
    "competition_density",
    "residential_ratio",
    "poi_footfall_score",
    "catchment_population_1km",
    "structural_daily",
    "demand_daily",
    "supply_daily",
]

CATEGORICAL_COLS = ["tier", "shop_type", "road_class"]

QUANTILES = [0.10, 0.50, 0.90]

DEFAULT_PARAMS = {
    "n_estimators": 300,
    "max_depth": 5,
    "learning_rate": 0.05,
    "min_child_samples": 20,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "verbose": -1,
}


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in CATEGORICAL_COLS:
        if col in df.columns:
            df[col] = df[col].astype("category")
    return df


def _feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    return _encode_categoricals(df)[FEATURE_COLS + CATEGORICAL_COLS]


class QuantileTrainer:
    """Trains three quantile LightGBM models and evaluates calibration."""

    def __init__(self, params: dict | None = None):
        self.params = {**DEFAULT_PARAMS, **(params or {})}
        self.models: dict[float, lgb.LGBMRegressor] = {}
        self.metrics: dict[str, Any] = {}

    def train(
        self,
        df: pd.DataFrame,
        target_col: str = "daily_sales_truth",
        test_size: float = 0.2,
        seed: int = 42,
    ) -> dict[str, Any]:
        X = _feature_matrix(df)
        y = df[target_col]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=seed,
        )

        for q in QUANTILES:
            model = lgb.LGBMRegressor(
                objective="quantile",
                alpha=q,
                **self.params,
            )
            model.fit(
                X_train, y_train,
                categorical_feature=CATEGORICAL_COLS,
            )
            self.models[q] = model

        # --- Evaluate calibration ---
        preds = {
            q: self.models[q].predict(X_test)
            for q in QUANTILES
        }

        # Coverage: % of true values falling within [q10, q90].
        in_band = (y_test.values >= preds[0.10]) & (y_test.values <= preds[0.90])
        coverage = float(in_band.mean())

        # Median absolute error.
        mae = float(np.abs(y_test.values - preds[0.50]).mean())

        # Mean band width (relative to median prediction).
        band_width = float(np.mean((preds[0.90] - preds[0.10]) / np.maximum(preds[0.50], 1)))

        # Feature importance (from median model).
        importances = dict(zip(
            FEATURE_COLS + CATEGORICAL_COLS,
            self.models[0.50].feature_importances_.tolist(),
        ))

        self.metrics = {
            "coverage_80": round(coverage, 4),
            "target_coverage": 0.80,
            "mae": round(mae, 0),
            "mean_relative_band_width": round(band_width, 4),
            "test_size": len(X_test),
            "train_size": len(X_train),
            "feature_importance": importances,
        }

        return self.metrics

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for q, model in self.models.items():
            with open(path / f"q{int(q * 100)}.pkl", "wb") as f:
                pickle.dump(model, f)
        with open(path / "metrics.json", "w") as f:
            json.dump(self.metrics, f, indent=2)

    @staticmethod
    def load(path: str | Path) -> "QuantileModel":
        return QuantileModel.load(path)


class QuantileModel:
    """Loaded quantile model for inference."""

    def __init__(self, models: dict[float, lgb.LGBMRegressor], metrics: dict):
        self.models = models
        self.metrics = metrics

    @classmethod
    def load(cls, path: str | Path) -> "QuantileModel":
        path = Path(path)
        models = {}
        for q in QUANTILES:
            pkl = path / f"q{int(q * 100)}.pkl"
            with open(pkl, "rb") as f:
                models[q] = pickle.load(f)  # noqa: S301
        metrics = {}
        metrics_path = path / "metrics.json"
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text())
        return cls(models=models, metrics=metrics)

    def predict(self, features: dict) -> tuple[float, float, float]:
        """Predict (q10, q50, q90) from a single-row feature dict."""
        df = pd.DataFrame([features])
        X = _feature_matrix(df)
        lo = float(self.models[0.10].predict(X)[0])
        mid = float(self.models[0.50].predict(X)[0])
        hi = float(self.models[0.90].predict(X)[0])
        # Ensure ordering.
        lo = max(lo, 500)
        hi = max(hi, lo + 100)
        mid = max(min(mid, hi), lo)
        return lo, mid, hi
