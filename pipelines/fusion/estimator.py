"""Structural-prior cash-flow estimator.

This is the *economic logic core* of the system. It deliberately does NOT use
a neural network. Instead it computes two independent estimates of daily sales:

  (a) DEMAND-SIDE:  daily_sales ≈ footfall × conversion × ticket_size
  (b) SUPPLY-SIDE:  daily_sales ≈ inventory_value / effective_turnover_days

Both estimates are derived from the vision + geo features. If they agree, we
are more confident; if they disagree, we raise `prior_disagreement` which the
fraud layer will pick up.

The Phase-2 quantile-regression head will sit *on top of* this structural
estimate as a calibration layer — not replace it. The structural bones stay
even when the ML body changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from api.schemas import (
    CashFlowEstimate,
    GeoFeatures,
    OwnerDeclared,
    VisionFeatures,
)


LOOKUPS = Path(__file__).resolve().parents[2] / "data" / "lookups"
TIER_PRIORS = json.loads((LOOKUPS / "tier_priors.json").read_text())
CATEGORY_MARGINS = json.loads((LOOKUPS / "category_margins.json").read_text())


@dataclass
class _StructuralBreakdown:
    demand_side_daily: float
    supply_side_daily: float
    blended_daily: float
    ticket_size: float
    effective_footfall: float
    blended_margin_low: float
    blended_margin_high: float


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


def _structural(
    vision: VisionFeatures,
    geo: GeoFeatures,
    owner: OwnerDeclared,
) -> _StructuralBreakdown:
    tier = TIER_PRIORS[geo.tier]

    # ---- DEMAND-SIDE ----
    # Footfall = tier base × geo demand score × road multiplier × residential boost
    road_mult = {"arterial": 1.4, "collector": 1.15, "residential": 1.0, "alley": 0.75}[
        geo.road_class
    ]
    residential_boost = 0.85 + 0.3 * geo.residential_ratio  # residential areas => reliable daily buyers
    competition_penalty = max(0.6, 1.0 - 0.4 * max(geo.competition_density - 0.5, 0))

    effective_footfall = (
        tier["daily_footfall_base"]
        * (0.5 + geo.geo_demand_score)  # 0.5x..1.5x
        * road_mult
        * residential_boost
        * competition_penalty
    )

    # Conversion rate: how many passers-by actually buy. Small shops convert
    # a surprisingly high fraction of regulars; we assume 35-60% depending on
    # SKU diversity (more diversity = more reasons to walk in).
    conversion = 0.35 + 0.25 * vision.sku_diversity

    # Ticket size: tier median, nudged by category mix (tobacco/paan => small
    # tickets; beverages/FMCG => larger tickets).
    ticket_size = tier["ticket_size_median"]
    mix = vision.category_mix
    ticket_size *= 1.0 - 0.2 * mix.get("tobacco_paan", 0) + 0.15 * mix.get("beverages", 0)

    demand_side_daily = effective_footfall * conversion * ticket_size

    # ---- SUPPLY-SIDE ----
    turnover_days = _effective_turnover_days(mix)
    # A healthy shop sells through its working capital every `turnover_days`.
    # Daily sales ≈ inventory_value / turnover_days, but scale down by refill
    # signal: if refill_signal is weak, inventory is not actually moving.
    refill_scale = 0.5 + 0.8 * vision.refill_signal  # 0.5..1.3
    supply_side_daily = (vision.inventory_value_est / turnover_days) * refill_scale

    # ---- BLEND ----
    # Weight by extraction confidence of vision (more trust in vision =>
    # supply side counts more).
    w_supply = 0.35 + 0.4 * vision.extraction_confidence
    w_demand = 1.0 - w_supply
    blended = w_demand * demand_side_daily + w_supply * supply_side_daily

    # Owner-declared shop size sanity nudge: if owner says 40 sqft and we see
    # huge inventory, don't let the estimate run away.
    if owner.shop_size_sqft and owner.shop_size_sqft > 0:
        sqft_cap = owner.shop_size_sqft * 120  # ₹120/sqft/day soft ceiling
        blended = min(blended, sqft_cap * 1.5)

    lo_m, hi_m = _blended_margin(mix)

    return _StructuralBreakdown(
        demand_side_daily=demand_side_daily,
        supply_side_daily=supply_side_daily,
        blended_daily=blended,
        ticket_size=ticket_size,
        effective_footfall=effective_footfall,
        blended_margin_low=lo_m,
        blended_margin_high=hi_m,
    )


_QUANTILE_MODEL = None


def _load_quantile_model():
    global _QUANTILE_MODEL
    if _QUANTILE_MODEL is not None:
        return _QUANTILE_MODEL
    model_path = Path(__file__).resolve().parents[2] / "models" / "quantile_lgbm"
    if not model_path.exists():
        return None
    try:
        from pipelines.fusion.ml.quantile_model import QuantileModel
        _QUANTILE_MODEL = QuantileModel.load(model_path)
        return _QUANTILE_MODEL
    except Exception:
        return None


def _quantile_head(
    point: float,
    vision: VisionFeatures,
    geo: GeoFeatures,
    struct: _StructuralBreakdown,
) -> tuple[float, float, float]:
    """Quantile regression calibration head.

    Uses the trained LightGBM model if available; otherwise falls back to
    a simple spread-based heuristic. Returns (lo, mid, hi) daily sales.
    """
    model = _load_quantile_model()
    if model is not None:
        features = {
            "sdi": vision.shelf_density_index,
            "sku_diversity": vision.sku_diversity,
            "inventory_value": vision.inventory_value_est,
            "refill_signal": vision.refill_signal,
            "store_size_sqft": vision.store_size_sqft,
            "pos_sophistication": vision.pos_sophistication,
            "coverage_score": vision.coverage_score,
            "extraction_confidence": vision.extraction_confidence,
            "geo_demand_score": geo.geo_demand_score,
            "competition_density": geo.competition_density,
            "residential_ratio": geo.residential_ratio,
            "poi_footfall_score": geo.poi_footfall_score,
            "catchment_population_1km": geo.catchment_population_1km,
            "structural_daily": struct.blended_daily,
            "demand_daily": struct.demand_side_daily,
            "supply_daily": struct.supply_side_daily,
            "tier": geo.tier,
            "shop_type": vision.shop_type,
            "road_class": geo.road_class,
        }
        lo, mid, hi = model.predict(features)
        return lo, mid, hi

    # Fallback: widen the band inversely to extraction confidence.
    spread = 0.20 + (1.0 - vision.extraction_confidence) * 0.35
    return point * (1 - spread), point, point * (1 + spread)


def estimate_cashflow(
    vision: VisionFeatures,
    geo: GeoFeatures,
    owner: OwnerDeclared,
) -> CashFlowEstimate:
    struct = _structural(vision, geo, owner)
    point = struct.blended_daily

    daily_lo, daily_mid, daily_hi = _quantile_head(point, vision, geo, struct)

    # Working days per month ≈ 28 (allow for closures, festivals).
    monthly_rev_lo = daily_lo * 28
    monthly_rev_hi = daily_hi * 28

    monthly_inc_lo = monthly_rev_lo * struct.blended_margin_low
    monthly_inc_hi = monthly_rev_hi * struct.blended_margin_high

    # Prior disagreement: |demand - supply| / max(demand, supply)
    denom = max(struct.demand_side_daily, struct.supply_side_daily, 1.0)
    disagreement = abs(struct.demand_side_daily - struct.supply_side_daily) / denom

    return CashFlowEstimate(
        daily_sales_range=[round(daily_lo, 0), round(daily_hi, 0)],
        monthly_revenue_range=[round(monthly_rev_lo, 0), round(monthly_rev_hi, 0)],
        monthly_income_range=[round(monthly_inc_lo, 0), round(monthly_inc_hi, 0)],
        structural_point_estimate=round(point, 0),
        model_point_estimate=round(daily_mid, 0),
        prior_disagreement=round(min(disagreement, 1.0), 3),
    )
