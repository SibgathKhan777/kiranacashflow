"""Pydantic schemas for the underwriting API.

All monetary values are in INR. All ranges are (low, high) tuples expressed as
two-element lists in JSON so they're easy to consume from any client.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, conlist


Range = conlist(float, min_length=2, max_length=2)


class GPS(BaseModel):
    lat: float = Field(..., ge=-90, le=90)
    lng: float = Field(..., ge=-180, le=180)


class OwnerDeclared(BaseModel):
    """Optional owner-provided metadata. Never trusted blindly."""
    shop_size_sqft: float | None = None
    monthly_rent: float | None = None
    years_in_operation: float | None = None


class UnderwriteRequest(BaseModel):
    gps: GPS
    owner_declared: OwnerDeclared = OwnerDeclared()
    # Images are uploaded separately via multipart; this model is for the
    # JSON metadata sidecar.


class ImageQualityReport(BaseModel):
    filename: str
    role: Literal[
        "interior_shelf", "counter", "exterior", "street",
        "street_stall", "specialty_loose", "irrelevant", "unknown",
    ]
    blur_score: float
    brightness: float
    resolution_ok: bool
    duplicate_of: str | None = None
    watermark_suspect: bool = False
    quality_pass: bool
    notes: list[str] = []


class VisionFeatures(BaseModel):
    shop_type: Literal[
        "packaged_kirana", "loose_goods", "street_stall",
        "organized_retail", "exterior_only", "unknown",
    ]
    shelf_density_index: float  # 0..1
    sku_diversity: float        # 0..1
    inventory_value_est: float  # INR
    refill_signal: float        # 0..1 (higher = more "real demand" evidence)
    store_size_sqft: float
    pos_sophistication: float   # 0..1
    category_mix: dict[str, float]
    coverage_score: float       # 0..1 (how much of the shop we actually saw)
    extraction_confidence: float  # 0..1


class GeoFeatures(BaseModel):
    tier: Literal["metro", "tier1", "tier2", "tier3", "rural"]
    catchment_population_1km: int
    residential_ratio: float     # 0..1
    road_class: Literal["arterial", "collector", "residential", "alley"]
    poi_footfall_score: float    # 0..1
    competition_density: float   # 0..1 (higher = more competing kiranas)
    geo_demand_score: float      # 0..1 composite


class CashFlowEstimate(BaseModel):
    daily_sales_range: list[float]
    monthly_revenue_range: list[float]
    monthly_income_range: list[float]
    structural_point_estimate: float  # pre-calibration best guess
    model_point_estimate: float       # post-calibration (stubbed in Phase 1)
    prior_disagreement: float          # 0..1, 0 = perfect agreement


class RiskFlag(BaseModel):
    code: str
    severity: Literal["low", "medium", "high"]
    message: str


class UnderwriteResponse(BaseModel):
    daily_sales_range: list[float]
    monthly_revenue_range: list[float]
    monthly_income_range: list[float]
    confidence_score: float
    risk_flags: list[str]
    recommendation: Literal[
        "approve_for_underwriting",
        "needs_verification",
        "reject_or_field_visit",
    ]
    explainability: dict
