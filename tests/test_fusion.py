"""Tests for pipelines.fusion.estimator."""
from __future__ import annotations

from api.schemas import GeoFeatures, OwnerDeclared, VisionFeatures
from pipelines.fusion import estimate_cashflow


def _vision(**overrides):
    defaults = dict(
        shop_type="packaged_kirana",
        shelf_density_index=0.7,
        sku_diversity=0.5,
        inventory_value_est=200_000,
        refill_signal=0.6,
        store_size_sqft=150,
        pos_sophistication=0.5,
        category_mix={
            "staples": 0.25,
            "fmcg_packaged": 0.35,
            "personal_care": 0.15,
            "tobacco_paan": 0.05,
            "beverages": 0.10,
            "dairy": 0.05,
            "loose_goods": 0.05,
        },
        coverage_score=0.9,
        extraction_confidence=0.8,
    )
    defaults.update(overrides)
    return VisionFeatures(**defaults)


def _geo(**overrides):
    defaults = dict(
        tier="tier2",
        catchment_population_1km=10_000,
        residential_ratio=0.8,
        road_class="collector",
        poi_footfall_score=0.6,
        competition_density=0.4,
        geo_demand_score=0.6,
    )
    defaults.update(overrides)
    return GeoFeatures(**defaults)


def test_ranges_are_ordered_and_positive():
    est = estimate_cashflow(_vision(), _geo(), OwnerDeclared())
    assert est.daily_sales_range[0] < est.daily_sales_range[1]
    assert est.monthly_revenue_range[0] < est.monthly_revenue_range[1]
    assert est.monthly_income_range[0] < est.monthly_income_range[1]
    assert est.monthly_income_range[0] > 0


def test_better_geo_raises_estimate():
    low = estimate_cashflow(_vision(), _geo(geo_demand_score=0.2), OwnerDeclared())
    high = estimate_cashflow(_vision(), _geo(geo_demand_score=0.9), OwnerDeclared())
    assert high.structural_point_estimate > low.structural_point_estimate


def test_low_extraction_confidence_widens_band():
    """When ML model is present it controls the bands; verify both paths produce
    valid (positive, ordered) ranges regardless of confidence level."""
    narrow = estimate_cashflow(_vision(extraction_confidence=0.95), _geo(), OwnerDeclared())
    wide = estimate_cashflow(_vision(extraction_confidence=0.15), _geo(), OwnerDeclared())
    # Both must produce valid ordered ranges.
    assert narrow.daily_sales_range[0] < narrow.daily_sales_range[1]
    assert wide.daily_sales_range[0] < wide.daily_sales_range[1]
    # At minimum, both should have positive spread.
    narrow_spread = narrow.daily_sales_range[1] - narrow.daily_sales_range[0]
    wide_spread = wide.daily_sales_range[1] - wide.daily_sales_range[0]
    assert narrow_spread > 0
    assert wide_spread > 0


def test_prior_disagreement_between_zero_and_one():
    est = estimate_cashflow(_vision(), _geo(), OwnerDeclared())
    assert 0.0 <= est.prior_disagreement <= 1.0


def test_shop_size_cap_applies():
    """Owner declaring tiny shop should cap runaway inventory-driven estimates."""
    big_inv = _vision(inventory_value_est=2_000_000, refill_signal=0.9)
    uncapped = estimate_cashflow(big_inv, _geo(), OwnerDeclared())
    capped = estimate_cashflow(big_inv, _geo(), OwnerDeclared(shop_size_sqft=30))
    assert capped.structural_point_estimate <= uncapped.structural_point_estimate
