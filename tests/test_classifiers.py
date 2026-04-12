"""Tests for pipelines.vision.classifiers."""
from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from pipelines.vision.classifiers import classify_role, classify_shop_type, detect_pos_signals


def _make_arr(h=600, w=800, style="busy", seed=42):
    rng = np.random.default_rng(seed)
    if style == "sky_top":
        arr = np.zeros((h, w, 3), dtype=np.uint8)
        arr[: h // 3] = [170, 200, 230]  # sky: bright, blue-ish
        arr[h // 3 :] = rng.integers(60, 180, size=(h - h // 3, w, 3))
        return arr
    if style == "dim_cluttered":
        return rng.integers(10, 70, size=(h, w, 3), dtype=np.uint8)
    if style == "bright_shelves":
        arr = rng.integers(100, 230, size=(h, w, 3), dtype=np.uint8)
        # Add horizontal shelf lines.
        for y in range(0, h, 60):
            arr[y : y + 3] = [50, 50, 50]
        return arr
    return rng.integers(40, 220, size=(h, w, 3), dtype=np.uint8)


def test_sky_top_classifies_as_exterior():
    arr = _make_arr(style="sky_top")
    pil = Image.fromarray(arr)
    role, evidence = classify_role(pil, arr)
    assert role == "exterior", f"expected exterior, got {role}: {evidence}"


def test_bright_shelves_classifies_as_interior():
    arr = _make_arr(style="bright_shelves")
    pil = Image.fromarray(arr)
    role, evidence = classify_role(pil, arr)
    assert role in ("interior_shelf", "counter"), f"got {role}: {evidence}"


def test_dim_cluttered_classifies_as_stall_or_interior():
    arr = _make_arr(style="dim_cluttered")
    pil = Image.fromarray(arr)
    role, evidence = classify_role(pil, arr)
    assert role in ("street_stall", "interior_shelf", "counter"), f"got {role}"


def test_paytm_blue_detection():
    # Paint a small Paytm-blue rectangle.
    arr = np.full((600, 800, 3), 120, dtype=np.uint8)
    arr[100:200, 200:350] = [30, 180, 240]  # Paytm blue
    result = detect_pos_signals(arr)
    assert result["detections"]["paytm_blue"] is True
    assert result["pos_score"] > 0.2


def test_no_brand_detection_on_plain_gray():
    arr = np.full((600, 800, 3), 128, dtype=np.uint8)
    result = detect_pos_signals(arr)
    assert result["pos_score"] == 0.0
    assert not any(result["detections"].values())


def test_cocacola_red_detection():
    arr = np.full((600, 800, 3), 100, dtype=np.uint8)
    arr[50:200, 50:100] = [220, 30, 20]  # Coca-Cola red
    result = detect_pos_signals(arr)
    assert result["detections"]["cocacola_red"] is True
