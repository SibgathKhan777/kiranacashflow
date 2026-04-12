"""Synthetic training data generator for the quantile calibration head.

Generates realistic kirana-shop feature vectors with ground-truth daily sales,
drawn from the structural priors with controlled noise. This lets us train
the quantile head *before* we have real labeled data: the model learns to
compress and calibrate the structural estimates, and once real labels come in
(even 30-50), we fine-tune.

Design choices:
  - Tier distribution reflects India's actual shop density (~15% metro, ~20%
    tier1, ~30% tier2, ~25% tier3, ~10% rural).
  - Ground truth is generated from the structural prior + multiplicative
    lognormal noise (σ=0.3), which means the model can't just memorize the
    prior — it has to learn to denoise.
  - We include adversarial "fraud" samples (~5%) where SDI is artificially
    inflated but footfall is low, so the model learns that high SDI in a
    low-demand area doesn't translate to high sales.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


LOOKUPS = Path(__file__).resolve().parents[3] / "data" / "lookups"
TIER_PRIORS = json.loads((LOOKUPS / "tier_priors.json").read_text())
CATEGORY_MARGINS = json.loads((LOOKUPS / "category_margins.json").read_text())

TIERS = ["metro", "tier1", "tier2", "tier3", "rural"]
TIER_WEIGHTS = [0.15, 0.20, 0.30, 0.25, 0.10]

ROAD_CLASSES = ["arterial", "collector", "residential", "alley"]
ROAD_CLASS_WEIGHTS = [0.10, 0.25, 0.45, 0.20]

SHOP_TYPES = ["packaged_kirana", "loose_goods", "street_stall"]
SHOP_TYPE_WEIGHTS = [0.70, 0.15, 0.15]


def _blended_margin(category_mix: dict[str, float]) -> tuple[float, float]:
    lo = hi = 0.0
    total = sum(category_mix.values()) or 1.0
    for cat, weight in category_mix.items():
        margins = CATEGORY_MARGINS.get(cat)
        if not margins:
            continue
        lo += (weight / total) * margins["margin_low"]
        hi += (weight / total) * margins["margin_high"]
    return max(lo, 0.04), max(hi, 0.08)


def _effective_turnover_days(category_mix: dict[str, float]) -> float:
    total = sum(category_mix.values()) or 1.0
    days = 0.0
    for cat, weight in category_mix.items():
        margins = CATEGORY_MARGINS.get(cat)
        if not margins:
            continue
        days += (weight / total) * margins["turnover_days"]
    return max(days, 5.0)


def generate_dataset(
    n: int = 2000,
    fraud_fraction: float = 0.05,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate n synthetic kirana-shop samples with ground-truth daily sales.

    Returns a DataFrame with columns matching the feature vector consumed by
    the quantile head, plus `daily_sales_truth` as the label.
    """
    rng = np.random.default_rng(seed)
    rows: list[dict] = []

    n_fraud = int(n * fraud_fraction)
    n_clean = n - n_fraud

    for i in range(n):
        is_fraud = i >= n_clean

        tier = rng.choice(TIERS, p=TIER_WEIGHTS)
        tp = TIER_PRIORS[tier]
        shop_type = rng.choice(SHOP_TYPES, p=SHOP_TYPE_WEIGHTS)
        road_class = rng.choice(ROAD_CLASSES, p=ROAD_CLASS_WEIGHTS)

        # -- Vision features --
        sdi = float(rng.beta(4, 3))          # skewed toward 0.5-0.7
        sku_div = float(rng.beta(3, 3))
        refill = float(rng.beta(3, 3))

        if shop_type == "loose_goods":
            category_mix = {
                "staples": 0.15, "fmcg_packaged": 0.10, "personal_care": 0.05,
                "tobacco_paan": 0.05, "beverages": 0.05, "dairy": 0.05,
                "loose_goods": 0.55,
            }
        elif shop_type == "street_stall":
            category_mix = {
                "staples": 0.10, "fmcg_packaged": 0.15, "personal_care": 0.10,
                "tobacco_paan": 0.20, "beverages": 0.15, "dairy": 0.05,
                "loose_goods": 0.25,
            }
        else:
            category_mix = {
                "staples": 0.25, "fmcg_packaged": 0.35, "personal_care": 0.15,
                "tobacco_paan": 0.05, "beverages": 0.10, "dairy": 0.05,
                "loose_goods": 0.05,
            }

        base_inv = 80_000 + sdi * 400_000
        if shop_type == "loose_goods":
            base_inv *= 1.5
        elif shop_type == "street_stall":
            base_inv *= 0.5
        inventory = float(base_inv * rng.lognormal(0, 0.2))

        store_size = float(rng.lognormal(np.log(120), 0.4))
        store_size = max(30, min(store_size, 1000))

        pos_soph = float(rng.beta(2, 5))  # most shops have low POS
        if tier == "metro":
            pos_soph = min(pos_soph + 0.2, 1.0)

        coverage = float(rng.choice([0.33, 0.67, 1.0], p=[0.15, 0.35, 0.50]))
        extraction_conf = float(rng.beta(5, 2))  # skew toward high confidence

        # -- Geo features --
        tier_pop = tp["daily_footfall_base"]
        geo_demand = float(rng.beta(4, 3))
        competition = float(rng.beta(3, 4))
        residential_ratio = float(rng.beta(5, 3))
        poi_footfall = float(rng.beta(3, 3))
        catchment_pop = int(tier_pop * 100 * (0.5 + geo_demand))

        # -- Structural estimate (demand-side) --
        road_mult = {"arterial": 1.4, "collector": 1.15, "residential": 1.0, "alley": 0.75}[road_class]
        residential_boost = 0.85 + 0.3 * residential_ratio
        comp_penalty = max(0.6, 1.0 - 0.4 * max(competition - 0.5, 0))
        effective_footfall = tier_pop * (0.5 + geo_demand) * road_mult * residential_boost * comp_penalty
        conversion = 0.35 + 0.25 * sku_div
        ticket_size = tp["ticket_size_median"]
        demand_daily = effective_footfall * conversion * ticket_size

        # -- Structural estimate (supply-side) --
        turnover_days = _effective_turnover_days(category_mix)
        refill_scale = 0.5 + 0.8 * refill
        supply_daily = (inventory / turnover_days) * refill_scale

        # -- Blend --
        w_s = 0.35 + 0.4 * extraction_conf
        structural_daily = (1 - w_s) * demand_daily + w_s * supply_daily

        # -- Ground truth: structural + noise --
        if is_fraud:
            # Fraud: SDI inflated, but true sales are low (from demand side only).
            sdi = min(sdi + 0.3, 0.98)
            refill = max(refill - 0.3, 0.05)
            inventory *= 2.0  # staged inventory
            truth = demand_daily * float(rng.lognormal(0, 0.15))  # truth follows demand, not supply
        else:
            truth = structural_daily * float(rng.lognormal(0, 0.25))

        truth = max(truth, 500)  # floor

        margin_lo, margin_hi = _blended_margin(category_mix)

        rows.append({
            "tier": tier,
            "shop_type": shop_type,
            "road_class": road_class,
            "sdi": round(sdi, 4),
            "sku_diversity": round(sku_div, 4),
            "inventory_value": round(inventory, 0),
            "refill_signal": round(refill, 4),
            "store_size_sqft": round(store_size, 1),
            "pos_sophistication": round(pos_soph, 4),
            "coverage_score": round(coverage, 3),
            "extraction_confidence": round(extraction_conf, 4),
            "geo_demand_score": round(geo_demand, 4),
            "competition_density": round(competition, 4),
            "residential_ratio": round(residential_ratio, 4),
            "poi_footfall_score": round(poi_footfall, 4),
            "catchment_population_1km": catchment_pop,
            "structural_daily": round(structural_daily, 0),
            "demand_daily": round(demand_daily, 0),
            "supply_daily": round(supply_daily, 0),
            "margin_lo": round(margin_lo, 4),
            "margin_hi": round(margin_hi, 4),
            "is_fraud": is_fraud,
            "daily_sales_truth": round(truth, 0),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset(2000)
    out = LOOKUPS.parent / "synthetic_train.csv"
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} samples to {out}")
    print(f"  Tiers: {df.tier.value_counts().to_dict()}")
    print(f"  Types: {df.shop_type.value_counts().to_dict()}")
    print(f"  Fraud: {df.is_fraud.sum()}")
    print(f"  Daily sales: ₹{df.daily_sales_truth.median():,.0f} median, "
          f"₹{df.daily_sales_truth.mean():,.0f} mean")
