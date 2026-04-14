"""Final decision assembler.

Combines cash-flow estimate + risk flags + extraction confidence into the
response schema the API returns. Also computes the top feature drivers for
the explainability section — a credit officer must see *why*.
"""
from __future__ import annotations

from api.schemas import (
    CashFlowEstimate,
    GeoFeatures,
    RiskFlag,
    UnderwriteResponse,
    VisionFeatures,
)
from pipelines.lending.loan_sizer import size_loan
from pipelines.lending.peer_benchmark import benchmark


SEVERITY_PENALTY = {"low": 0.05, "medium": 0.15, "high": 0.40}


def _confidence(
    vision: VisionFeatures,
    estimate: CashFlowEstimate,
    flags: list[RiskFlag],
) -> float:
    base = 0.3 + 0.5 * vision.extraction_confidence + 0.2 * vision.coverage_score
    base -= 0.3 * estimate.prior_disagreement
    for f in flags:
        base -= SEVERITY_PENALTY[f.severity]
    return round(max(min(base, 0.99), 0.05), 3)


def _recommendation(confidence: float, flags: list[RiskFlag]) -> str:
    if any(f.severity == "high" for f in flags):
        return "reject_or_field_visit"
    if confidence >= 0.75 and not any(f.severity == "medium" for f in flags):
        return "approve_for_underwriting"
    if confidence < 0.4:
        return "reject_or_field_visit"
    return "needs_verification"


def _top_drivers(vision: VisionFeatures, geo: GeoFeatures) -> list[str]:
    drivers: list[tuple[str, float]] = [
        (f"geo_demand={geo.geo_demand_score}", geo.geo_demand_score),
        (f"shelf_density={vision.shelf_density_index}", vision.shelf_density_index),
        (f"sku_diversity={vision.sku_diversity}", vision.sku_diversity),
        (f"tier={geo.tier}", 1.0 if geo.tier in ("metro", "tier1") else 0.5),
        (f"road_class={geo.road_class}", 1.0 if geo.road_class == "arterial" else 0.4),
        (f"refill_signal={vision.refill_signal}", vision.refill_signal),
    ]
    drivers.sort(key=lambda x: x[1], reverse=True)
    return [d[0] for d in drivers[:4]]


def assemble_decision(
    vision: VisionFeatures,
    geo: GeoFeatures,
    estimate: CashFlowEstimate,
    flags: list[RiskFlag],
) -> UnderwriteResponse:
    conf = _confidence(vision, estimate, flags)
    rec = _recommendation(conf, flags)

    response = UnderwriteResponse(
        daily_sales_range=estimate.daily_sales_range,
        monthly_revenue_range=estimate.monthly_revenue_range,
        monthly_income_range=estimate.monthly_income_range,
        confidence_score=conf,
        risk_flags=[f.code for f in flags],
        recommendation=rec,  # type: ignore[arg-type]
        explainability={},
    )

    loan = size_loan(response, estimate)
    peer = benchmark(vision, geo, estimate)

    response.explainability = {
        "shop_type": vision.shop_type,
        "tier": geo.tier,
        "top_drivers": _top_drivers(vision, geo),
        "structural_point_estimate": estimate.structural_point_estimate,
        "model_point_estimate": estimate.model_point_estimate,
        "prior_disagreement": estimate.prior_disagreement,
        "flag_details": [
            {"code": f.code, "severity": f.severity, "message": f.message}
            for f in flags
        ],
        "vision_features": vision.model_dump(),
        "geo_features": geo.model_dump(),
        "loan_sizing": {
            "eligible": loan.eligible,
            "recommended_loan": loan.recommended_loan,
            "max_eligible_loan": loan.max_eligible_loan,
            "confidence_haircut": loan.confidence_haircut,
            "foir_ceiling": loan.foir_ceiling,
            "risk_adjustment": loan.risk_adjustment,
            "ineligible_reason": loan.ineligible_reason,
            "emi_options": [
                {
                    "tenure_months": e.tenure_months,
                    "emi_amount": e.emi_amount,
                    "total_repayment": e.total_repayment,
                    "foir": e.foir,
                }
                for e in loan.emi_options
            ],
        },
        "peer_benchmark": {
            "peer_cohort": peer.peer_cohort,
            "percentile": peer.percentile,
            "cohort_median_daily": peer.cohort_median_daily,
            "cohort_p25_daily": peer.cohort_p25_daily,
            "cohort_p75_daily": peer.cohort_p75_daily,
            "this_shop_daily": peer.this_shop_daily,
            "interpretation": peer.interpretation,
        },
    }

    return response
