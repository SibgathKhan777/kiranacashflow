"""Visual feature extractor — Phase 3.

Combines real image analysis with the new inventory estimation module:
  - Grid-based product region detection
  - HSV color → category matching → price band lookup
  - Visibility-corrected total inventory value
  - Fast-moving vs slow-moving turnover mix

No GPU dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from api.schemas import VisionFeatures
from pipelines.ingest import ImageRecord
from pipelines.vision.classifiers import classify_shop_type, detect_pos_signals
from pipelines.vision.inventory import estimate_inventory


DEFAULT_CATEGORY_MIX = {
    "staples": 0.25,
    "fmcg_packaged": 0.35,
    "personal_care": 0.15,
    "tobacco_paan": 0.05,
    "beverages": 0.10,
    "dairy": 0.05,
    "loose_goods": 0.05,
}


@dataclass
class _PerImageStats:
    color_variance: float
    edge_density: float
    saturation: float
    brightness: float
    aspect: float
    horiz_line_density: float
    vert_line_density: float


def _stats(arr: np.ndarray) -> _PerImageStats:
    gray = arr.mean(axis=2)
    gx = np.abs(np.diff(gray, axis=1))
    gy = np.abs(np.diff(gray, axis=0))
    gx = np.pad(gx, ((0, 0), (0, 1)))
    gy = np.pad(gy, ((0, 1), (0, 0)))
    grad = np.hypot(gx, gy)
    edge_density = float((grad > 20).mean())
    color_variance = float(arr.reshape(-1, 3).astype(np.float32).var(axis=0).mean())

    mx = arr.max(axis=2).astype(np.float32)
    mn = arr.min(axis=2).astype(np.float32)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1), 0.0)
    saturation = float(sat.mean())
    brightness = float(gray.mean())
    h, w = gray.shape
    aspect = w / max(h, 1)

    horiz_line_density = float((gy > 15).mean(axis=1).mean())
    vert_line_density = float((gx > 15).mean(axis=0).mean())

    return _PerImageStats(
        color_variance=color_variance,
        edge_density=edge_density,
        saturation=saturation,
        brightness=brightness,
        aspect=aspect,
        horiz_line_density=horiz_line_density,
        vert_line_density=vert_line_density,
    )


def _estimate_sdi(interior_stats: list[_PerImageStats]) -> float:
    """Shelf Density Index from interior image edge profiles.

    Higher edge density + higher horizontal line density = more shelved
    products. We combine both because edge density alone can't distinguish
    "busy but messy" from "neat shelves full of products."
    """
    if not interior_stats:
        return 0.3
    combined = [
        0.55 * s.edge_density + 0.45 * s.horiz_line_density
        for s in interior_stats
    ]
    return float(np.clip(np.mean(combined) * 3.0, 0.05, 0.98))


def _estimate_sku_diversity(
    stats: list[_PerImageStats], shop_type: str
) -> tuple[float, dict[str, float]]:
    """SKU diversity from color variance + saturation.

    Different-colored packaging = more categories. Low saturation + low
    color variance = loose goods (earthy tones). High everything = packaged.
    """
    avg_sat = float(np.mean([s.saturation for s in stats]))
    avg_cv = float(np.mean([s.color_variance for s in stats]))

    diversity = float(np.clip(avg_sat * 0.8 + avg_cv / 5000, 0.05, 0.98))

    # Shift category mix based on shop type.
    if shop_type == "loose_goods":
        mix = {
            "staples": 0.15, "fmcg_packaged": 0.10, "personal_care": 0.05,
            "tobacco_paan": 0.05, "beverages": 0.05, "dairy": 0.05,
            "loose_goods": 0.55,
        }
    elif shop_type == "street_stall":
        mix = {
            "staples": 0.10, "fmcg_packaged": 0.15, "personal_care": 0.10,
            "tobacco_paan": 0.20, "beverages": 0.15, "dairy": 0.05,
            "loose_goods": 0.25,
        }
    else:
        mix = dict(DEFAULT_CATEGORY_MIX)

    return diversity, mix


def _estimate_inventory_value(
    sdi: float, stats: list[_PerImageStats], shop_type: str
) -> float:
    """Inventory value from SDI + brightness + shop type.

    Loose goods stores carry higher-value inventory per visible sqft
    (dry fruits cost more than biscuits). Street stalls carry less.
    """
    avg_bright = float(np.mean([s.brightness for s in stats]))

    base = 80_000 + sdi * 400_000 + avg_bright * 200
    if shop_type == "loose_goods":
        base *= 1.5  # dry fruits/spices are higher ₹/kg than packaged FMCG
    elif shop_type == "street_stall":
        base *= 0.5
    elif shop_type == "organized_retail":
        base *= 3.0
    return float(base)


def _estimate_refill_signal(interior_stats: list[_PerImageStats]) -> float:
    """Refill signal from shelf-level edge variance.

    Real demand leaves uneven shelves: some areas depleted, others full.
    Perfectly uniform high-density is suspicious (staged). Very low density
    everywhere is also weak.
    """
    if not interior_stats:
        return 0.5
    edge_values = [s.edge_density for s in interior_stats]
    var = float(np.var(edge_values))
    mean = float(np.mean(edge_values))
    # High variance + moderate mean = natural demand pattern.
    return float(np.clip(var * 30 + mean * 0.5, 0.05, 0.95))


def _estimate_store_size(
    records: list[ImageRecord], sdi: float
) -> float:
    """Store size from exterior dimensions + interior visible area.

    Phase 3 will use MiDaS monocular depth for real metric estimation.
    For now: distinct view angles + SDI as proxy.
    """
    distinct_roles = len({r.report.role for r in records if r.report.quality_pass})
    # More roles = we see more of the shop = probably larger.
    return float(90 + distinct_roles * 40 + sdi * 60)


def extract_vision_features(records: list[ImageRecord]) -> VisionFeatures:
    usable = [r for r in records if r.report.quality_pass]
    if not usable:
        return VisionFeatures(
            shop_type="unknown",
            shelf_density_index=0.0,
            sku_diversity=0.0,
            inventory_value_est=0.0,
            refill_signal=0.0,
            store_size_sqft=0.0,
            pos_sophistication=0.0,
            category_mix=DEFAULT_CATEGORY_MIX,
            coverage_score=0.0,
            extraction_confidence=0.05,
        )

    stats = [_stats(r.array) for r in usable]

    # Shop type from the enhanced classifier.
    shop_type, _shop_evidence = classify_shop_type(records)

    # Interior-specific stats for shelf analysis.
    interior = [
        s for s, r in zip(stats, usable)
        if r.report.role in ("interior_shelf", "counter")
    ]

    sdi = _estimate_sdi(interior)
    refill = _estimate_refill_signal(interior)
    store_size = _estimate_store_size(usable, sdi)

    # Real inventory estimation: product detection + price band mapping.
    inv_est = estimate_inventory(
        arrays=[r.array for r in usable],
        roles=[r.report.role for r in usable],
        store_size_sqft=store_size,
    )
    inventory = inv_est.estimated_total_value

    # Derive category mix from detected product breakdown.
    if inv_est.category_breakdown and sum(inv_est.category_breakdown.values()) > 0:
        total_detected = sum(inv_est.category_breakdown.values())
        category_mix = {
            cat: round(val / total_detected, 4)
            for cat, val in inv_est.category_breakdown.items()
        }
        # Ensure all categories present (fill missing with small weight).
        for cat in DEFAULT_CATEGORY_MIX:
            if cat not in category_mix:
                category_mix[cat] = 0.01
        # Renormalize.
        total = sum(category_mix.values())
        category_mix = {k: round(v / total, 4) for k, v in category_mix.items()}
        sku_div = min(len(inv_est.category_breakdown) / 7.0 + 0.1, 0.98)
    else:
        sku_div, category_mix = _estimate_sku_diversity(stats, shop_type)

    # POS sophistication from brand/QR detection across all images.
    pos_detections = [detect_pos_signals(r.array) for r in usable]
    pos_soph = max(d["pos_score"] for d in pos_detections) if pos_detections else 0.0
    # Boost slightly by brightness (well-lit shops tend to be more sophisticated).
    avg_bright = float(np.mean([s.brightness for s in stats]))
    pos_soph = float(np.clip(pos_soph + avg_bright / 600, 0.05, 0.95))

    # Coverage: what fraction of the ideal role set did we receive?
    ideal_roles = {"interior_shelf", "counter", "exterior"}
    actual_roles = {r.report.role for r in usable}
    coverage = len(ideal_roles & actual_roles) / len(ideal_roles)

    # Extraction confidence.
    qual_pass_ratio = len(usable) / max(len(records), 1)
    extraction_conf = float(np.clip(
        0.3 + 0.4 * coverage + 0.3 * qual_pass_ratio,
        0.1, 0.95,
    ))

    return VisionFeatures(
        shop_type=shop_type,  # type: ignore[arg-type]
        shelf_density_index=round(sdi, 3),
        sku_diversity=round(sku_div, 3),
        inventory_value_est=round(inventory, 0),
        inventory_visible_value=round(inv_est.visible_value, 0),
        inventory_cells_detected=inv_est.cell_count,
        inventory_detection_confidence=round(inv_est.detection_confidence, 3),
        fast_moving_fraction=round(inv_est.fast_moving_fraction, 3),
        slow_moving_fraction=round(inv_est.slow_moving_fraction, 3),
        avg_inventory_turnover_days=round(inv_est.avg_turnover_days, 1),
        refill_signal=round(refill, 3),
        store_size_sqft=round(store_size, 1),
        pos_sophistication=round(pos_soph, 3),
        category_mix=category_mix,
        coverage_score=round(coverage, 3),
        extraction_confidence=round(extraction_conf, 3),
    )
