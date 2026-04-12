"""Geo-spatial feature extractor.

Preferred path: query OpenStreetMap via Overpass for real building density,
POI counts, road class, and competing-shop counts around the shop's GPS pin.
These raw counts are then normalized against tier-specific priors to produce
the `GeoFeatures` schema the fusion layer consumes.

Fallback path: if Overpass is unreachable (offline dev, rate limit, timeout),
we fall back to a deterministic stub so the pipeline remains usable. The stub
is *not* random — same lat/lng always produces the same features, so tests
remain stable.

Tier classification is still heuristic (metro bounding boxes + tier1 city
radii + India-ish default). Phase 3 will replace it with a pincode lookup.
"""
from __future__ import annotations

import hashlib
import math
import os

from api.schemas import GeoFeatures, GPS

from .overpass import OverpassSnapshot, OverpassUnavailable, fetch_overpass


# ---------------------------------------------------------------------------
# Tier classification (unchanged from Phase 1)
# ---------------------------------------------------------------------------

METRO_BOXES: list[tuple[str, float, float, float, float]] = [
    ("mumbai",    18.89, 19.27, 72.77, 72.99),
    ("delhi",     28.40, 28.88, 76.84, 77.35),
    ("bangalore", 12.83, 13.14, 77.46, 77.78),
    ("chennai",   12.83, 13.23, 80.12, 80.32),
    ("hyderabad", 17.25, 17.55, 78.30, 78.65),
    ("kolkata",   22.45, 22.68, 88.25, 88.45),
    ("pune",      18.43, 18.66, 73.73, 73.99),
    ("ahmedabad", 22.95, 23.13, 72.48, 72.70),
]

TIER1_CITIES: list[tuple[str, float, float, float]] = [
    ("jaipur",         26.92, 75.79, 0.25),
    ("lucknow",        26.85, 80.95, 0.25),
    ("surat",          21.17, 72.83, 0.20),
    ("kanpur",         26.45, 80.33, 0.20),
    ("nagpur",         21.15, 79.09, 0.22),
    ("indore",         22.72, 75.86, 0.22),
    ("bhopal",         23.26, 77.40, 0.22),
    ("coimbatore",     11.02, 76.96, 0.20),
    ("kochi",           9.93, 76.27, 0.18),
    ("visakhapatnam",  17.69, 83.22, 0.20),
]


def _classify_tier(lat: float, lng: float) -> str:
    for _, la_min, la_max, lg_min, lg_max in METRO_BOXES:
        if la_min <= lat <= la_max and lg_min <= lng <= lg_max:
            return "metro"
    for _, clat, clng, r in TIER1_CITIES:
        if math.hypot(lat - clat, lng - clng) <= r:
            return "tier1"
    if 6.0 <= lat <= 37.0 and 68.0 <= lng <= 98.0:
        key = int(hashlib.md5(f"{lat:.3f},{lng:.3f}".encode()).hexdigest(), 16)
        bucket = key % 10
        if bucket < 4:
            return "tier2"
        if bucket < 8:
            return "tier3"
        return "rural"
    return "rural"


# ---------------------------------------------------------------------------
# Real path: normalize Overpass snapshot into GeoFeatures
# ---------------------------------------------------------------------------

# Building-count thresholds for saturating the population proxy, per tier.
# Rationale: 1 km² of dense metro has ~600+ building footprints in OSM;
# rural ~20. We saturate at the tier-specific "very busy" level.
_BUILDING_SATURATION = {
    "metro":  (50, 800),
    "tier1":  (40, 500),
    "tier2":  (30, 300),
    "tier3":  (20, 200),
    "rural":  (5, 60),
}

# Rough residents-per-building by tier (family sizes + floors).
_RESIDENTS_PER_BUILDING = {
    "metro":  45,
    "tier1":  30,
    "tier2":  20,
    "tier3":  12,
    "rural":  7,
}

# POI weights for the footfall score. Schools/transit drive daily passers-by;
# banks/markets bring destination traffic; food amenities indicate walkability.
_POI_WEIGHTS = {
    "school":  0.20,
    "health":  0.10,
    "finance": 0.15,
    "transit": 0.20,
    "food":    0.20,
    "worship": 0.05,
    "market":  0.10,
}


def _normalize_from_overpass(gps: GPS, tier: str, snap: OverpassSnapshot) -> GeoFeatures:
    low, high = _BUILDING_SATURATION[tier]
    bld = snap.building_count_1km
    # Saturating piecewise-linear mapping for the population proxy.
    if bld <= low:
        density_norm = 0.1 + 0.2 * (bld / max(low, 1))
    elif bld >= high:
        density_norm = 1.0
    else:
        density_norm = 0.3 + 0.7 * ((bld - low) / (high - low))

    catchment_pop = int(snap.building_count_1km * _RESIDENTS_PER_BUILDING[tier])
    catchment_pop = max(catchment_pop, 500)

    # Residential ratio: share of tagged buildings that are residential. When
    # OSM tags are sparse, fall back to a tier prior.
    typed = snap.residential_building_count + snap.commercial_building_count
    if typed >= 5:
        residential_ratio = snap.residential_building_count / typed
    else:
        residential_ratio = {"metro": 0.65, "tier1": 0.70, "tier2": 0.78, "tier3": 0.82, "rural": 0.88}[tier]
    residential_ratio = round(min(max(residential_ratio, 0.0), 1.0), 3)

    # POI footfall score: weighted sum, squashed to 0..1. The divisor is the
    # POI count at which each category saturates (roughly 5 of each is "a lot").
    poi_score = 0.0
    for key, weight in _POI_WEIGHTS.items():
        count = snap.poi_counts.get(key, 0)
        poi_score += weight * min(count / 5.0, 1.0)
    poi_score = round(min(max(poi_score, 0.0), 1.0), 3)

    # Competition density: saturate at ~12 competing shops in a 500m radius.
    competition = min(snap.competing_kirana_count_500m / 12.0, 1.0)
    competition = round(competition, 3)

    geo_demand = round(
        0.5 * density_norm + 0.4 * poi_score - 0.2 * max(competition - 0.5, 0),
        3,
    )
    geo_demand = min(max(geo_demand, 0.0), 1.0)

    return GeoFeatures(
        tier=tier,  # type: ignore[arg-type]
        catchment_population_1km=catchment_pop,
        residential_ratio=residential_ratio,
        road_class=snap.nearest_road_class,  # type: ignore[arg-type]
        poi_footfall_score=poi_score,
        competition_density=competition,
        geo_demand_score=geo_demand,
    )


# ---------------------------------------------------------------------------
# Fallback path: deterministic stub (Phase 1 behaviour)
# ---------------------------------------------------------------------------

def _det_rand(lat: float, lng: float, salt: str) -> float:
    key = f"{lat:.4f},{lng:.4f},{salt}"
    h = int(hashlib.md5(key.encode()).hexdigest(), 16)
    return (h % 10_000) / 10_000.0


def _stub_features(gps: GPS, tier: str) -> GeoFeatures:
    tier_pop_base = {
        "metro":  (18_000, 45_000),
        "tier1":  (10_000, 25_000),
        "tier2":  ( 5_000, 14_000),
        "tier3":  ( 2_500,  7_500),
        "rural":  (   500,  3_000),
    }[tier]
    pop_rand = _det_rand(gps.lat, gps.lng, "pop")
    catchment_pop = int(tier_pop_base[0] + pop_rand * (tier_pop_base[1] - tier_pop_base[0]))

    residential_ratio = round(0.4 + 0.5 * _det_rand(gps.lat, gps.lng, "res"), 3)

    road_rand = _det_rand(gps.lat, gps.lng, "road")
    if road_rand < 0.15:
        road_class = "arterial"
    elif road_rand < 0.45:
        road_class = "collector"
    elif road_rand < 0.85:
        road_class = "residential"
    else:
        road_class = "alley"

    poi_footfall = round(
        0.2 + 0.6 * _det_rand(gps.lat, gps.lng, "poi")
        + (0.2 if tier == "metro" else 0.05 if tier == "tier1" else 0.0),
        3,
    )
    poi_footfall = min(max(poi_footfall, 0.0), 1.0)

    competition = round(0.2 + 0.7 * _det_rand(gps.lat, gps.lng, "comp"), 3)
    competition = min(max(competition, 0.0), 1.0)

    pop_norm = min(catchment_pop / 30_000, 1.0)
    geo_demand = round(
        0.5 * pop_norm + 0.4 * poi_footfall - 0.2 * max(competition - 0.5, 0),
        3,
    )
    geo_demand = min(max(geo_demand, 0.0), 1.0)

    return GeoFeatures(
        tier=tier,  # type: ignore[arg-type]
        catchment_population_1km=catchment_pop,
        residential_ratio=residential_ratio,
        road_class=road_class,  # type: ignore[arg-type]
        poi_footfall_score=poi_footfall,
        competition_density=competition,
        geo_demand_score=geo_demand,
    )


# ---------------------------------------------------------------------------
# Public entrypoint
# ---------------------------------------------------------------------------

def extract_geo_features(gps: GPS) -> GeoFeatures:
    """Extract geo features, preferring Overpass and falling back to the stub.

    Set `KIRANA_GEO_STUB=1` in the environment to force the deterministic
    stub path — useful for tests and demos on flaky networks.
    """
    tier = _classify_tier(gps.lat, gps.lng)

    if os.environ.get("KIRANA_GEO_STUB") == "1":
        return _stub_features(gps, tier)

    try:
        snap = fetch_overpass(gps.lat, gps.lng)
    except OverpassUnavailable:
        return _stub_features(gps, tier)

    return _normalize_from_overpass(gps, tier, snap)
