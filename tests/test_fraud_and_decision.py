"""Tests for fraud validators and decision assembly."""
from __future__ import annotations

from api.schemas import (
    CashFlowEstimate,
    GeoFeatures,
    OwnerDeclared,
    VisionFeatures,
)
from pipelines.decision import assemble_decision
from pipelines.fraud import run_fraud_validators
from pipelines.fusion import estimate_cashflow
from pipelines.ingest import assess_images
from pipelines.vision import extract_vision_features
from pipelines.geo import extract_geo_features


def _vision(**kw):
    base = dict(
        shop_type="packaged_kirana",
        shelf_density_index=0.7,
        sku_diversity=0.6,
        inventory_value_est=180_000,
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
        catchment_population_1km=10_000,
        residential_ratio=0.8,
        road_class="collector",
        poi_footfall_score=0.6,
        competition_density=0.4,
        geo_demand_score=0.6,
    )
    base.update(kw)
    return GeoFeatures(**base)


def test_staged_inventory_is_flagged(clean_metro_files, mumbai_gps):
    records = assess_images(clean_metro_files)
    vf = _vision(shelf_density_index=0.95, refill_signal=0.1)
    gf = extract_geo_features(mumbai_gps)
    est = estimate_cashflow(vf, gf, OwnerDeclared())
    flags = run_fraud_validators(records, vf, gf, est)
    codes = [f.code for f in flags]
    assert "staged_inventory_suspect" in codes


def test_inventory_footfall_mismatch(clean_metro_files):
    records = assess_images(clean_metro_files)
    vf = _vision(inventory_value_est=600_000)
    gf = _geo(geo_demand_score=0.15)
    est = estimate_cashflow(vf, gf, OwnerDeclared())
    flags = run_fraud_validators(records, vf, gf, est)
    assert any(f.code == "inventory_footfall_mismatch" for f in flags)


def test_insufficient_images_flag(make_image):
    records = assess_images([("only.png", make_image(5, style="busy"))])
    vf = extract_vision_features(records)
    gf = _geo()
    est = estimate_cashflow(vf, gf, OwnerDeclared())
    flags = run_fraud_validators(records, vf, gf, est)
    assert any(f.code == "insufficient_images" for f in flags)


def test_high_severity_forces_reject(clean_metro_files):
    records = assess_images(clean_metro_files)
    vf = _vision(shop_type="organized_retail")
    gf = _geo()
    est = estimate_cashflow(vf, gf, OwnerDeclared())
    flags = run_fraud_validators(records, vf, gf, est)
    decision = assemble_decision(vf, gf, est, flags)
    assert decision.recommendation == "reject_or_field_visit"


def test_clean_high_confidence_approves(clean_metro_files):
    records = assess_images(clean_metro_files)
    vf = _vision(extraction_confidence=0.95, coverage_score=1.0)
    gf = _geo(geo_demand_score=0.65, competition_density=0.3)
    est = estimate_cashflow(vf, gf, OwnerDeclared())
    # Manually empty flag list to simulate a truly clean case.
    decision = assemble_decision(vf, gf, est, [])
    assert decision.recommendation == "approve_for_underwriting"
    assert decision.confidence_score >= 0.75
