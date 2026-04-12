"""Real Inventory Value Approximation.

What the problem statement asks for:
  - Detect products and map them to price bands
  - Estimate total inventory value
  - Fast-moving vs slow-moving mix

How we do it without a GPU/neural net:
  1. PRODUCT REGION DETECTION: segment the image into candidate product
     "cells" using a grid + color clustering approach. Each cell is a
     candidate shelf face (one product facing).
  2. CATEGORY CLASSIFICATION: classify each cell into a category
     (staples/fmcg/beverages/etc.) using its dominant HSV color signature
     matched against the price_bands.json color fingerprints.
  3. PRICE BAND MAPPING: for each detected cell, look up the avg_price
     and avg_units_per_face for that category to estimate cell value.
  4. VISIBILITY CORRECTION: only 30-50% of actual stock is visible on
     shelves. Apply a depth multiplier based on shelf depth signals.
  5. TURNOVER MIX: derive fast-moving (turnover ≤ 7 days) vs slow-moving
     (turnover > 21 days) fractions from the category mix.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

LOOKUPS = Path(__file__).resolve().parents[2] / "data" / "lookups"
PRICE_BANDS = json.loads((LOOKUPS / "price_bands.json").read_text())
CATEGORY_MARGINS = json.loads((LOOKUPS / "category_margins.json").read_text())

CATEGORIES = list(PRICE_BANDS["categories"].keys())

# HSV color fingerprints for each category.
# Format: list of (hue_center, hue_range, min_sat, min_val) tuples.
# These capture the dominant packaging colors of Indian FMCG products.
CATEGORY_COLOR_SIGNATURES: dict[str, list[tuple]] = {
    "staples": [
        (30, 25, 0.05, 0.7),    # white/cream (atta, rice bags)
        (40, 20, 0.3, 0.6),     # brown/yellow (dal, grain)
    ],
    "fmcg_packaged": [
        (10, 20, 0.7, 0.7),     # red/orange (chips, noodles)
        (55, 15, 0.6, 0.7),     # yellow (biscuits, tea)
        (220, 30, 0.5, 0.6),    # blue (detergent)
        (120, 40, 0.4, 0.6),    # green (various FMCG)
    ],
    "personal_care": [
        (330, 20, 0.3, 0.8),    # pink/white (creams, lotions)
        (210, 30, 0.4, 0.8),    # blue-white (toothpaste)
        (85, 30, 0.4, 0.6),     # yellow-green (hair oil)
    ],
    "tobacco_paan": [
        (340, 15, 0.5, 0.9),    # bright red-pink (gutka)
        (150, 30, 0.5, 0.5),    # green (paan masala)
        (0, 10, 0.2, 0.3),      # near-black (beedi)
    ],
    "beverages": [
        (0, 15, 0.8, 0.5),      # deep red (Cola)
        (200, 25, 0.5, 0.7),    # blue-clear (water)
        (30, 20, 0.8, 0.9),     # orange (juices)
    ],
    "dairy": [
        (50, 30, 0.1, 0.9),     # white/off-white (milk, curd)
    ],
    "loose_goods": [
        (25, 20, 0.4, 0.4),     # brown/tan (peanuts, namkeen)
        (15, 15, 0.5, 0.6),     # earthy orange (spices)
    ],
}

# Visibility correction: only a fraction of inventory is visible.
# More shelf bays visible = better coverage.
VISIBILITY_MULTIPLIER_BASE = 3.0   # assume we see ~33% of total stock
VISIBILITY_MULTIPLIER_VARIANCE = 0.5

# Minimum detectable cells per image.
GRID_ROWS = 5
GRID_COLS = 8


@dataclass
class DetectedRegion:
    row: int
    col: int
    dominant_hue: float
    dominant_sat: float
    dominant_val: float
    category: str
    category_confidence: float
    estimated_cell_value: float
    turnover_days: float


@dataclass
class InventoryEstimate:
    total_value: float
    visible_value: float          # what we can see
    estimated_total_value: float  # after visibility correction
    category_breakdown: dict[str, float]   # category -> ₹ value
    category_unit_counts: dict[str, int]   # category -> detected facings
    fast_moving_fraction: float   # fraction of value with turnover ≤ 7 days
    slow_moving_fraction: float   # fraction of value with turnover > 21 days
    mid_moving_fraction: float
    avg_turnover_days: float
    cell_count: int
    detection_confidence: float
    notes: list[str] = field(default_factory=list)


def _rgb_to_hsv(r: float, g: float, b: float) -> tuple[float, float, float]:
    """Convert 0-255 RGB to HSV (H: 0-360, S: 0-1, V: 0-1)."""
    r, g, b = r / 255, g / 255, b / 255
    mx = max(r, g, b)
    mn = min(r, g, b)
    diff = mx - mn

    if diff == 0:
        h = 0.0
    elif mx == r:
        h = (60 * ((g - b) / diff) + 360) % 360
    elif mx == g:
        h = (60 * ((b - r) / diff) + 120) % 360
    else:
        h = (60 * ((r - g) / diff) + 240) % 360

    s = 0.0 if mx == 0 else diff / mx
    v = mx
    return h, s, v


def _hue_distance(h1: float, h2: float) -> float:
    """Circular hue distance (0-180)."""
    d = abs(h1 - h2)
    return min(d, 360 - d)


def _match_category(hue: float, sat: float, val: float) -> tuple[str, float]:
    """Match HSV color to the most likely product category.

    Returns (category, confidence 0..1).
    """
    best_cat = "fmcg_packaged"
    best_score = -1.0

    for cat, sigs in CATEGORY_COLOR_SIGNATURES.items():
        for h_center, h_range, min_sat, min_val in sigs:
            h_dist = _hue_distance(hue, h_center)
            if h_dist > h_range * 2:
                continue
            if sat < min_sat * 0.5:
                continue

            # Score: closer to center + higher saturation = better match.
            h_score = max(0, 1 - h_dist / (h_range + 1))
            s_score = min(sat / max(min_sat, 0.1), 1.0)
            v_score = min(val / max(min_val, 0.1), 1.0)
            score = h_score * 0.5 + s_score * 0.3 + v_score * 0.2

            if score > best_score:
                best_score = score
                best_cat = cat

    # Low-saturation regions are likely staples (white/plain packaging).
    if sat < 0.15 and val > 0.7:
        return "staples", max(best_score, 0.6)

    return best_cat, max(best_score, 0.0)


def _estimate_cell_value(category: str) -> tuple[float, float]:
    """Estimate ₹ value and turnover days for one shelf face (cell)."""
    cat_data = PRICE_BANDS["categories"].get(category, {})
    items = cat_data.get("items", {})
    if not items:
        return 150.0, 20.0

    # Average across items in category.
    values = []
    turnovers = []
    for item in items.values():
        unit_val = item["avg_price"] * item["avg_units_per_face"]
        values.append(unit_val)
        turnovers.append(item["turnover_days"])

    return float(np.mean(values)), float(np.mean(turnovers))


def _extract_shelf_cells(
    array: np.ndarray,
    role: str,
    grid_rows: int = GRID_ROWS,
    grid_cols: int = GRID_COLS,
) -> list[DetectedRegion]:
    """Divide the image into a grid of candidate shelf-face cells.

    Only interior_shelf and counter images are used for product detection.
    We skip the top 15% (often ceiling/header) and bottom 10% (floor/base).
    """
    if role not in ("interior_shelf", "counter", "street_stall"):
        return []

    h, w, _ = array.shape

    # Skip ceiling and floor zones.
    top_skip = int(h * 0.15)
    bot_skip = int(h * 0.10)
    usable = array[top_skip : h - bot_skip, :, :]

    uh, uw, _ = usable.shape
    cell_h = uh // grid_rows
    cell_w = uw // grid_cols

    regions: list[DetectedRegion] = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            cell = usable[r * cell_h : (r + 1) * cell_h, c * cell_w : (c + 1) * cell_w]
            if cell.size == 0:
                continue

            # Dominant color = median RGB of the cell (robust to outliers).
            median_rgb = np.median(cell.reshape(-1, 3), axis=0)
            hue, sat, val = _rgb_to_hsv(*median_rgb)

            # Skip cells that are mostly background (very dark or very uniform).
            edge_var = float(cell.astype(np.float32).var())
            if edge_var < 50 or val < 0.1:
                continue

            cat, conf = _match_category(hue, sat, val)
            cell_value, turnover = _estimate_cell_value(cat)

            regions.append(DetectedRegion(
                row=r, col=c,
                dominant_hue=round(hue, 1),
                dominant_sat=round(sat, 3),
                dominant_val=round(val, 3),
                category=cat,
                category_confidence=round(conf, 3),
                estimated_cell_value=round(cell_value, 1),
                turnover_days=turnover,
            ))

    return regions


def estimate_inventory(
    arrays: list[np.ndarray],
    roles: list[str],
    store_size_sqft: float = 120.0,
) -> InventoryEstimate:
    """Estimate total inventory value from interior images.

    Steps:
    1. Detect product cells via grid segmentation on each interior image.
    2. Map each cell to a category via HSV color matching.
    3. Compute visible value = sum(cell_value).
    4. Apply visibility correction (we see ~33% of actual stock).
    5. Derive fast/slow turnover mix.
    """
    all_regions: list[DetectedRegion] = []
    for arr, role in zip(arrays, roles):
        cells = _extract_shelf_cells(arr, role)
        all_regions.extend(cells)

    notes: list[str] = []

    if not all_regions:
        notes.append("no_interior_images_for_product_detection")
        # Fall back to store-size-based prior.
        fallback = store_size_sqft * 900  # ₹900/sqft average working capital
        return InventoryEstimate(
            total_value=round(fallback, 0),
            visible_value=0,
            estimated_total_value=round(fallback, 0),
            category_breakdown={},
            category_unit_counts={},
            fast_moving_fraction=0.25,
            slow_moving_fraction=0.40,
            mid_moving_fraction=0.35,
            avg_turnover_days=18.0,
            cell_count=0,
            detection_confidence=0.20,
            notes=notes,
        )

    # Aggregate by category.
    cat_value: dict[str, float] = {}
    cat_count: dict[str, int] = {}
    cat_turnover: dict[str, list[float]] = {}

    for reg in all_regions:
        cat = reg.category
        cat_value[cat] = cat_value.get(cat, 0) + reg.estimated_cell_value
        cat_count[cat] = cat_count.get(cat, 0) + 1
        cat_turnover.setdefault(cat, []).append(reg.turnover_days)

    visible_value = sum(cat_value.values())

    # Visibility correction.
    # More images = better coverage = lower multiplier needed.
    n_interior = sum(1 for r in roles if r in ("interior_shelf", "counter", "street_stall"))
    visibility_mult = max(VISIBILITY_MULTIPLIER_BASE - 0.3 * (n_interior - 1), 1.5)
    total_value = visible_value * visibility_mult

    # Store-size sanity cap (₹1,500/sqft max working capital for kirana).
    cap = store_size_sqft * 1500
    if total_value > cap:
        notes.append(f"inventory_capped_at_store_size_prior (cap=₹{cap:,.0f})")
        total_value = cap

    # Turnover mix.
    total_val = max(sum(cat_value.values()), 1)
    fast_val = sum(
        v for cat, v in cat_value.items()
        if np.mean(cat_turnover.get(cat, [20])) <= 7
    )
    slow_val = sum(
        v for cat, v in cat_value.items()
        if np.mean(cat_turnover.get(cat, [20])) > 21
    )
    mid_val = total_val - fast_val - slow_val

    avg_turnover = float(np.mean([r.turnover_days for r in all_regions]))

    # Detection confidence: based on number of cells detected and
    # average per-cell confidence.
    avg_conf = float(np.mean([r.category_confidence for r in all_regions]))
    cell_coverage = min(len(all_regions) / (GRID_ROWS * GRID_COLS * max(n_interior, 1)), 1.0)
    detection_conf = round(0.4 * avg_conf + 0.6 * cell_coverage, 3)

    return InventoryEstimate(
        total_value=round(total_value, 0),
        visible_value=round(visible_value, 0),
        estimated_total_value=round(total_value, 0),
        category_breakdown={k: round(v, 0) for k, v in cat_value.items()},
        category_unit_counts=cat_count,
        fast_moving_fraction=round(fast_val / total_val, 3),
        slow_moving_fraction=round(slow_val / total_val, 3),
        mid_moving_fraction=round(mid_val / total_val, 3),
        avg_turnover_days=round(avg_turnover, 1),
        cell_count=len(all_regions),
        detection_confidence=detection_conf,
        notes=notes,
    )
