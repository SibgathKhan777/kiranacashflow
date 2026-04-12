"""Thin Overpass API client with on-disk caching and graceful offline fallback.

We deliberately keep this lightweight — no osmnx/geopandas dependencies. One
HTTP POST, one JSON parse. Timeouts are short so a dead network doesn't hang
the pipeline; callers should catch `OverpassUnavailable` and fall back.

Reference: https://wiki.openstreetmap.org/wiki/Overpass_API
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import requests_cache
    _SESSION: Any = requests_cache.CachedSession(
        cache_name=str(Path(__file__).resolve().parents[2] / ".cache" / "overpass"),
        backend="sqlite",
        expire_after=60 * 60 * 24 * 7,  # 7 days
        allowable_methods=("GET", "POST"),
    )
except ImportError:  # pragma: no cover - optional dep
    import requests
    _SESSION = requests.Session()


OVERPASS_ENDPOINT = "https://overpass-api.de/api/interpreter"
REQUEST_TIMEOUT = 8.0


class OverpassUnavailable(RuntimeError):
    """Raised when Overpass cannot be reached or returns an error."""


@dataclass
class OverpassSnapshot:
    """Aggregated counts/tags from a single Overpass round-trip."""
    building_count_1km: int
    residential_building_count: int
    commercial_building_count: int
    poi_counts: dict[str, int]       # schools, offices, transit, hospital, atm
    nearest_road_class: str           # arterial|collector|residential|alley
    competing_kirana_count_500m: int


def _meters_to_deg_lat(m: float) -> float:
    return m / 111_320.0


def _meters_to_deg_lng(m: float, at_lat: float) -> float:
    return m / (111_320.0 * max(math.cos(math.radians(at_lat)), 0.01))


def _build_query(lat: float, lng: float, radius_m: int = 1000) -> str:
    return f"""
[out:json][timeout:6];
(
  // 1km: building footprints for population proxy
  way(around:{radius_m},{lat},{lng})["building"];
  // 500m: POIs that drive footfall
  node(around:500,{lat},{lng})["amenity"~"school|college|university|hospital|clinic|bank|atm|bus_station|place_of_worship|marketplace|restaurant|cafe"];
  node(around:500,{lat},{lng})["shop"~"convenience|supermarket|grocery|general"];
  node(around:500,{lat},{lng})["public_transport"];
  // 200m: roads to classify nearest edge
  way(around:200,{lat},{lng})["highway"];
);
out tags;
"""


_ROAD_PRIORITY = {
    "motorway": "arterial",
    "trunk": "arterial",
    "primary": "arterial",
    "secondary": "collector",
    "tertiary": "collector",
    "unclassified": "residential",
    "residential": "residential",
    "living_street": "residential",
    "service": "alley",
    "track": "alley",
    "footway": "alley",
    "path": "alley",
}
_ROAD_RANK = {"arterial": 3, "collector": 2, "residential": 1, "alley": 0}


def fetch_overpass(lat: float, lng: float, radius_m: int = 1000) -> OverpassSnapshot:
    """Fetch and aggregate Overpass results. Raises OverpassUnavailable on failure."""
    query = _build_query(lat, lng, radius_m)
    try:
        r = _SESSION.post(OVERPASS_ENDPOINT, data={"data": query}, timeout=REQUEST_TIMEOUT)
    except Exception as exc:
        raise OverpassUnavailable(f"network error: {exc}") from exc
    if r.status_code != 200:
        raise OverpassUnavailable(f"status {r.status_code}: {r.text[:200]}")
    try:
        payload = r.json()
    except Exception as exc:
        raise OverpassUnavailable(f"bad json: {exc}") from exc

    buildings = 0
    residential = 0
    commercial = 0
    poi_counts: dict[str, int] = {
        "school": 0,
        "health": 0,
        "finance": 0,
        "transit": 0,
        "food": 0,
        "worship": 0,
        "market": 0,
    }
    competing_kiranas = 0
    best_road = "residential"
    best_rank = _ROAD_RANK[best_road]

    for el in payload.get("elements", []):
        tags = el.get("tags") or {}

        if "building" in tags:
            buildings += 1
            btype = tags.get("building", "")
            if btype in ("residential", "apartments", "house", "detached", "terrace"):
                residential += 1
            elif btype in ("commercial", "retail", "office", "industrial"):
                commercial += 1

        amen = tags.get("amenity")
        if amen:
            if amen in ("school", "college", "university"):
                poi_counts["school"] += 1
            elif amen in ("hospital", "clinic", "pharmacy", "doctors"):
                poi_counts["health"] += 1
            elif amen in ("bank", "atm"):
                poi_counts["finance"] += 1
            elif amen in ("bus_station",):
                poi_counts["transit"] += 1
            elif amen in ("restaurant", "cafe", "fast_food", "food_court"):
                poi_counts["food"] += 1
            elif amen == "place_of_worship":
                poi_counts["worship"] += 1
            elif amen == "marketplace":
                poi_counts["market"] += 1

        if "public_transport" in tags:
            poi_counts["transit"] += 1

        shop = tags.get("shop")
        if shop in ("convenience", "supermarket", "grocery", "general"):
            competing_kiranas += 1

        highway = tags.get("highway")
        if highway:
            mapped = _ROAD_PRIORITY.get(highway, "residential")
            rank = _ROAD_RANK[mapped]
            if rank > best_rank:
                best_rank = rank
                best_road = mapped

    return OverpassSnapshot(
        building_count_1km=buildings,
        residential_building_count=residential,
        commercial_building_count=commercial,
        poi_counts=poi_counts,
        nearest_road_class=best_road,
        competing_kirana_count_500m=competing_kiranas,
    )
