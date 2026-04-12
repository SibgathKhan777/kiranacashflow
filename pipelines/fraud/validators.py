"""Cross-signal fraud & consistency validators.

Each validator returns zero or more RiskFlags. These run *after* estimation
so they can inspect both the extracted features and the prior disagreement.

Severity scale:
  low    -> informational, nudges confidence down slightly
  medium -> routes to needs_verification
  high   -> routes to reject_or_field_visit regardless of other signals
"""
from __future__ import annotations

from api.schemas import (
    CashFlowEstimate,
    GeoFeatures,
    RiskFlag,
    VisionFeatures,
)
from pipelines.ingest import ImageRecord


def _coverage_flags(records: list[ImageRecord]) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    usable = [r for r in records if r.report.quality_pass]
    if len(usable) < 3:
        flags.append(RiskFlag(
            code="insufficient_images",
            severity="medium",
            message=f"Only {len(usable)} usable image(s) provided; minimum 3 recommended.",
        ))

    roles = {r.report.role for r in usable}
    if "interior_shelf" not in roles and "counter" not in roles:
        flags.append(RiskFlag(
            code="interior_view_missing",
            severity="medium",
            message="No interior shot detected — inventory cannot be verified.",
        ))
    if "exterior" not in roles:
        flags.append(RiskFlag(
            code="exterior_view_missing",
            severity="low",
            message="No exterior shot — store size and street context unverified.",
        ))
    return flags


def _watermark_flags(records: list[ImageRecord]) -> list[RiskFlag]:
    hits = [r for r in records if r.report.watermark_suspect]
    if not hits:
        return []
    return [RiskFlag(
        code="stock_photo_suspect",
        severity="high",
        message=f"Watermark/overlay detected on {len(hits)} image(s) — possible stock photo submission.",
    )]


def _duplicate_flags(records: list[ImageRecord]) -> list[RiskFlag]:
    dupes = [r for r in records if r.report.duplicate_of]
    if not dupes:
        return []
    return [RiskFlag(
        code="duplicate_images",
        severity="medium",
        message=f"{len(dupes)} duplicate image(s) submitted — possible padding of evidence.",
    )]


def _inventory_footfall_mismatch(
    vision: VisionFeatures, geo: GeoFeatures
) -> list[RiskFlag]:
    # High inventory in low-footfall area, OR low inventory in high-footfall area.
    hi_inv = vision.inventory_value_est > 300_000
    lo_inv = vision.inventory_value_est < 80_000
    hi_geo = geo.geo_demand_score > 0.7
    lo_geo = geo.geo_demand_score < 0.3

    if hi_inv and lo_geo:
        return [RiskFlag(
            code="inventory_footfall_mismatch",
            severity="high",
            message=(
                f"Inventory ≈ ₹{vision.inventory_value_est:,.0f} is high for a "
                f"low-demand area (geo score {geo.geo_demand_score}). "
                "Possible staged or borrowed stock."
            ),
        )]
    if lo_inv and hi_geo:
        return [RiskFlag(
            code="under_stocked_for_location",
            severity="medium",
            message=(
                "Inventory appears thin for a high-demand location — "
                "revenue estimate may under-call actual cashflow."
            ),
        )]
    return []


def _staged_inventory(vision: VisionFeatures) -> list[RiskFlag]:
    # Very high SDI with very low refill_signal = suspiciously tidy.
    if vision.shelf_density_index > 0.85 and vision.refill_signal < 0.25:
        return [RiskFlag(
            code="staged_inventory_suspect",
            severity="medium",
            message=(
                "Shelves are unusually full with no signs of recent sales — "
                "possible inspection-day staging."
            ),
        )]
    return []


def _prior_disagreement(estimate: CashFlowEstimate) -> list[RiskFlag]:
    if estimate.prior_disagreement > 0.6:
        return [RiskFlag(
            code="demand_supply_disagreement",
            severity="medium",
            message=(
                f"Demand-side and supply-side estimates diverge by "
                f"{estimate.prior_disagreement:.0%}. Underwriting should verify."
            ),
        )]
    return []


def _shop_type_flags(vision: VisionFeatures) -> list[RiskFlag]:
    if vision.shop_type == "organized_retail":
        return [RiskFlag(
            code="non_kirana_shop_type",
            severity="high",
            message="Shop appears to be organized retail, not a kirana. Different product required.",
        )]
    if vision.shop_type == "exterior_only":
        return [RiskFlag(
            code="exterior_only_submission",
            severity="medium",
            message="Only exterior visible — interior inventory unverified.",
        )]
    if vision.shop_type == "unknown":
        return [RiskFlag(
            code="shop_type_unclassified",
            severity="medium",
            message="Could not confidently classify the shop type.",
        )]
    return []


def run_fraud_validators(
    records: list[ImageRecord],
    vision: VisionFeatures,
    geo: GeoFeatures,
    estimate: CashFlowEstimate,
) -> list[RiskFlag]:
    flags: list[RiskFlag] = []
    flags.extend(_coverage_flags(records))
    flags.extend(_watermark_flags(records))
    flags.extend(_duplicate_flags(records))
    flags.extend(_shop_type_flags(vision))
    flags.extend(_inventory_footfall_mismatch(vision, geo))
    flags.extend(_staged_inventory(vision))
    flags.extend(_prior_disagreement(estimate))
    return flags
