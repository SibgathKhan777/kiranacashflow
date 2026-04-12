"""Train the quantile calibration head and save to models/.

Usage:
    python -m pipelines.fusion.ml.train          # from project root
    python pipelines/fusion/ml/train.py           # also works
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from pipelines.fusion.ml.data_generator import generate_dataset  # noqa: E402
from pipelines.fusion.ml.quantile_model import QuantileTrainer  # noqa: E402


def main():
    model_dir = ROOT / "models" / "quantile_lgbm"

    print("Generating 2000 synthetic training samples...")
    df = generate_dataset(2000, fraud_fraction=0.05, seed=42)

    print("Training quantile models (q10, q50, q90)...")
    trainer = QuantileTrainer()
    metrics = trainer.train(df)

    print(f"\n--- Calibration metrics ---")
    print(f"  80% coverage (target 0.80): {metrics['coverage_80']:.3f}")
    print(f"  Median absolute error:      ₹{metrics['mae']:,.0f}")
    print(f"  Mean band width (relative): {metrics['mean_relative_band_width']:.2%}")
    print(f"  Train / test split:         {metrics['train_size']} / {metrics['test_size']}")

    print(f"\n--- Top feature importances ---")
    imp = sorted(metrics["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    for name, score in imp[:8]:
        print(f"  {name:30s} {score:>5}")

    trainer.save(model_dir)
    print(f"\nModels saved to {model_dir}/")


if __name__ == "__main__":
    main()
