"""Tests for pipelines.geo.extractor (stub path)."""
from __future__ import annotations

from api.schemas import GPS
from pipelines.geo import extract_geo_features


def test_mumbai_is_metro(mumbai_gps):
    gf = extract_geo_features(mumbai_gps)
    assert gf.tier == "metro"
    assert gf.catchment_population_1km > 0
    assert gf.road_class in {"arterial", "collector", "residential", "alley"}
    assert 0.0 <= gf.geo_demand_score <= 1.0


def test_stub_is_deterministic(mumbai_gps):
    a = extract_geo_features(mumbai_gps)
    b = extract_geo_features(mumbai_gps)
    assert a.model_dump() == b.model_dump()


def test_rural_tier_has_lower_population():
    metro = extract_geo_features(GPS(lat=19.07, lng=72.87))
    rural = extract_geo_features(GPS(lat=24.55, lng=84.12))
    assert rural.tier in {"tier2", "tier3", "rural"}
    assert rural.catchment_population_1km < metro.catchment_population_1km
