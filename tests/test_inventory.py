"""Unit tests for pipelines.vision.inventory — product detection + price bands."""
from __future__ import annotations

import numpy as np

from pipelines.vision.inventory import (
    _estimate_cell_value,
    _extract_shelf_cells,
    _match_category,
    _rgb_to_hsv,
    estimate_inventory,
)


# ---------------------------------------------------------------------------
# _rgb_to_hsv
# ---------------------------------------------------------------------------

def test_rgb_to_hsv_red():
    h, s, v = _rgb_to_hsv(255, 0, 0)
    assert abs(h - 0) < 5 or abs(h - 360) < 5
    assert s > 0.9
    assert v > 0.9


def test_rgb_to_hsv_blue():
    h, s, v = _rgb_to_hsv(0, 0, 255)
    assert abs(h - 240) < 5
    assert s > 0.9


def test_rgb_to_hsv_white():
    h, s, v = _rgb_to_hsv(255, 255, 255)
    assert s < 0.05
    assert v > 0.95


def test_rgb_to_hsv_black():
    h, s, v = _rgb_to_hsv(0, 0, 0)
    assert v < 0.05


# ---------------------------------------------------------------------------
# _match_category
# ---------------------------------------------------------------------------

def test_match_white_cream_is_staples():
    # White/off-white = atta/rice bags
    h, s, v = _rgb_to_hsv(240, 235, 220)
    cat, conf = _match_category(h, s, v)
    assert cat == "staples"


def test_match_deep_red_is_beverages_or_fmcg():
    # Deep red = Cola can or chips packet
    h, s, v = _rgb_to_hsv(220, 20, 20)
    cat, conf = _match_category(h, s, v)
    assert cat in ("beverages", "fmcg_packaged", "tobacco_paan")


def test_match_paytm_blue_is_beverages_or_personal():
    # Paytm-style blue
    h, s, v = _rgb_to_hsv(30, 186, 242)
    cat, conf = _match_category(h, s, v)
    # Blue hue maps to beverages (water) or personal care (toothpaste)
    assert cat in ("beverages", "personal_care", "fmcg_packaged")


def test_match_returns_string_and_float():
    h, s, v = _rgb_to_hsv(180, 120, 60)
    cat, conf = _match_category(h, s, v)
    assert isinstance(cat, str)
    assert 0.0 <= conf <= 1.0


# ---------------------------------------------------------------------------
# _estimate_cell_value
# ---------------------------------------------------------------------------

def test_cell_value_all_categories_positive():
    categories = ["staples", "fmcg_packaged", "personal_care",
                  "tobacco_paan", "beverages", "dairy", "loose_goods"]
    for cat in categories:
        value, turnover = _estimate_cell_value(cat)
        assert value > 0, f"Zero value for {cat}"
        assert turnover > 0, f"Zero turnover for {cat}"


def test_loose_goods_and_tobacco_both_positive():
    # Both categories must have positive cell values
    loose_val, loose_t = _estimate_cell_value("loose_goods")
    tobacco_val, tobacco_t = _estimate_cell_value("tobacco_paan")
    assert loose_val > 0
    assert tobacco_val > 0
    assert loose_t > 0
    assert tobacco_t > 0


def test_dairy_fastest_turnover():
    _, dairy_t = _estimate_cell_value("dairy")
    _, staples_t = _estimate_cell_value("staples")
    assert dairy_t < staples_t


# ---------------------------------------------------------------------------
# _extract_shelf_cells
# ---------------------------------------------------------------------------

def test_shelf_cells_from_interior_image():
    rng = np.random.default_rng(1)
    arr = rng.integers(80, 220, (600, 800, 3), dtype=np.uint8)
    cells = _extract_shelf_cells(arr, "interior_shelf")
    assert len(cells) > 0
    assert all(0 <= c.dominant_hue <= 360 for c in cells)
    assert all(0 <= c.dominant_sat <= 1 for c in cells)


def test_no_cells_from_exterior_image():
    rng = np.random.default_rng(2)
    arr = rng.integers(80, 220, (600, 800, 3), dtype=np.uint8)
    cells = _extract_shelf_cells(arr, "exterior")
    assert cells == []


def test_no_cells_from_street_view():
    rng = np.random.default_rng(3)
    arr = rng.integers(80, 220, (600, 800, 3), dtype=np.uint8)
    cells = _extract_shelf_cells(arr, "street")
    assert cells == []


def test_cells_have_valid_estimated_values():
    rng = np.random.default_rng(4)
    arr = rng.integers(80, 220, (600, 800, 3), dtype=np.uint8)
    cells = _extract_shelf_cells(arr, "interior_shelf")
    for c in cells:
        assert c.estimated_cell_value > 0
        assert c.turnover_days > 0
        assert c.category in [
            "staples", "fmcg_packaged", "personal_care",
            "tobacco_paan", "beverages", "dairy", "loose_goods"
        ]


# ---------------------------------------------------------------------------
# estimate_inventory (integration)
# ---------------------------------------------------------------------------

def test_inventory_from_interior_image():
    rng = np.random.default_rng(10)
    arr = rng.integers(80, 220, (600, 800, 3), dtype=np.uint8)
    inv = estimate_inventory([arr], ["interior_shelf"], store_size_sqft=150)

    assert inv.cell_count > 0
    assert inv.visible_value > 0
    assert inv.estimated_total_value >= inv.visible_value
    assert 0.0 <= inv.fast_moving_fraction <= 1.0
    assert 0.0 <= inv.slow_moving_fraction <= 1.0
    assert abs(inv.fast_moving_fraction + inv.slow_moving_fraction + inv.mid_moving_fraction - 1.0) < 0.01
    assert inv.avg_turnover_days > 0
    assert inv.detection_confidence > 0


def test_inventory_fallback_with_no_interior():
    rng = np.random.default_rng(11)
    arr = rng.integers(80, 220, (600, 800, 3), dtype=np.uint8)
    inv = estimate_inventory([arr], ["exterior"], store_size_sqft=120)

    assert inv.cell_count == 0
    assert "no_interior_images_for_product_detection" in inv.notes
    assert inv.estimated_total_value == 120 * 900  # store-size-based fallback


def test_inventory_more_images_lower_multiplier():
    rng = np.random.default_rng(12)
    arr = rng.integers(80, 220, (600, 800, 3), dtype=np.uint8)

    single = estimate_inventory([arr], ["interior_shelf"], store_size_sqft=150)
    triple = estimate_inventory([arr, arr, arr], ["interior_shelf"] * 3, store_size_sqft=150)

    # More images → lower visibility multiplier → total/visible ratio is smaller
    if single.visible_value > 0 and triple.visible_value > 0:
        single_mult = single.estimated_total_value / single.visible_value
        triple_mult = triple.estimated_total_value / triple.visible_value
        assert triple_mult <= single_mult


def test_store_size_cap_applies():
    rng = np.random.default_rng(13)
    # Very small shop: 20 sqft → cap = 20 * 1500 = ₹30,000
    arr = rng.integers(80, 220, (600, 800, 3), dtype=np.uint8)
    inv = estimate_inventory([arr], ["interior_shelf"], store_size_sqft=20)
    assert inv.estimated_total_value <= 20 * 1500 + 1  # allow rounding
