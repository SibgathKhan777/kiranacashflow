"""Tests for loan sizing and peer benchmarking."""
from __future__ import annotations

from api.schemas import CashFlowEstimate, GeoFeatures, UnderwriteResponse, VisionFeatures
from pipelines.lending.loan_sizer import size_loan
from pipelines.lending.peer_benchmark import benchmark


def _estimate(**kw):
    base = dict(
        daily_sales_range=[6000, 10000],
        monthly_revenue_range=[168000, 280000],
        monthly_income_range=[20000, 40000],
        structural_point_estimate=8000,
        model_point_estimate=8000,
        prior_disagreement=0.15,
    )
    base.update(kw)
    return CashFlowEstimate(**base)


def _response(**kw):
    base = dict(
        daily_sales_range=[6000, 10000],
        monthly_revenue_range=[168000, 280000],
        monthly_income_range=[20000, 40000],
        confidence_score=0.75,
        risk_flags=[],
        recommendation="approve_for_underwriting",
        explainability={},
    )
    base.update(kw)
    return UnderwriteResponse(**base)


def _vision(**kw):
    base = dict(
        shop_type="packaged_kirana",
        shelf_density_index=0.7,
        sku_diversity=0.6,
        inventory_value_est=180000,
        refill_signal=0.6,
        store_size_sqft=150,
        pos_sophistication=0.5,
        category_mix={"fmcg_packaged": 1.0},
        coverage_score=0.9,
        extraction_confidence=0.8,
    )
    base.update(kw)
    return VisionFeatures(**base)


def _geo(**kw):
    base = dict(
        tier="tier2",
        catchment_population_1km=10000,
        residential_ratio=0.8,
        road_class="collector",
        poi_footfall_score=0.6,
        competition_density=0.4,
        geo_demand_score=0.6,
    )
    base.update(kw)
    return GeoFeatures(**base)


# ---- Loan sizing tests ----

def test_eligible_loan_has_positive_amounts():
    loan = size_loan(_response(), _estimate())
    assert loan.eligible
    assert loan.recommended_loan > 0
    assert loan.max_eligible_loan >= loan.recommended_loan
    assert len(loan.emi_options) == 4


def test_emi_options_are_ordered():
    loan = size_loan(_response(), _estimate())
    emis = [o.emi_amount for o in loan.emi_options]
    # Shorter tenure -> higher EMI.
    assert emis[0] > emis[-1]


def test_longest_tenure_foir_within_ceiling():
    """At the longest tenure the EMI should be affordable (within FOIR)."""
    loan = size_loan(_response(), _estimate())
    longest = loan.emi_options[-1]  # 24 months
    assert longest.foir <= loan.foir_ceiling + 0.01


def test_rejected_shops_get_zero_loan():
    resp = _response(
        recommendation="reject_or_field_visit",
        risk_flags=["stock_photo_suspect"],
    )
    loan = size_loan(resp, _estimate())
    assert not loan.eligible
    assert loan.recommended_loan == 0
    assert loan.ineligible_reason is not None


def test_low_income_below_minimum():
    est = _estimate(monthly_income_range=[3000, 6000])
    loan = size_loan(_response(), est)
    assert not loan.eligible
    assert "minimum" in loan.ineligible_reason.lower()


def test_risk_flags_reduce_foir():
    clean = size_loan(_response(), _estimate())
    risky = size_loan(
        _response(
            risk_flags=["demand_supply_disagreement"],
            confidence_score=0.45,
        ),
        _estimate(),
    )
    assert risky.foir_ceiling < clean.foir_ceiling


# ---- Peer benchmark tests ----

def test_peer_percentile_in_range():
    peer = benchmark(_vision(), _geo(), _estimate())
    assert 0 <= peer.percentile <= 100
    assert peer.cohort_size > 0
    assert peer.cohort_median_daily > 0


def test_peer_is_deterministic():
    a = benchmark(_vision(), _geo(), _estimate())
    b = benchmark(_vision(), _geo(), _estimate())
    assert a.percentile == b.percentile
    assert a.cohort_median_daily == b.cohort_median_daily


def test_high_estimate_gets_high_percentile():
    low = benchmark(_vision(), _geo(), _estimate(structural_point_estimate=2000))
    high = benchmark(_vision(), _geo(), _estimate(structural_point_estimate=20000))
    assert high.percentile > low.percentile
