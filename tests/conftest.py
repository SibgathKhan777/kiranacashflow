"""Shared pytest fixtures and helpers.

All tests force the geo extractor into stub mode so they are hermetic — no
Overpass calls, no network flake. Tests that exercise the real Overpass
client path should live in a separate `tests/integration/` tree and be
skipped by default.
"""
from __future__ import annotations

import io
import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# Make `api`, `pipelines`, etc. importable when pytest is run from any cwd.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Force deterministic geo stub for all tests in this suite.
os.environ["KIRANA_GEO_STUB"] = "1"


def _make_image(seed: int, size=(800, 600), style: str = "busy") -> bytes:
    rng = np.random.default_rng(seed)
    if style == "busy":
        arr = rng.integers(40, 220, size=(size[1], size[0], 3), dtype=np.uint8)
        for _ in range(80):
            x, y = rng.integers(0, size[0] - 60), rng.integers(0, size[1] - 60)
            w, h = rng.integers(20, 80), rng.integers(30, 100)
            color = rng.integers(30, 255, size=3)
            arr[y : y + h, x : x + w] = color
    elif style == "dim":
        arr = rng.integers(10, 70, size=(size[1], size[0], 3), dtype=np.uint8)
    elif style == "bright_exterior":
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[: size[1] // 3] = [180, 200, 230]
        arr[size[1] // 3 :] = rng.integers(80, 200, size=(size[1] - size[1] // 3, size[0], 3))
    elif style == "blurry":
        arr = np.full((size[1], size[0], 3), 128, dtype=np.uint8)
    else:
        arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def make_image():
    """Factory fixture returning synthetic PNG bytes."""
    return _make_image


@pytest.fixture
def clean_metro_files(make_image):
    return [
        ("interior.png", make_image(1, style="busy")),
        ("counter.png", make_image(2, style="busy")),
        ("exterior.png", make_image(3, style="bright_exterior")),
    ]


@pytest.fixture
def poor_input_files(make_image):
    # Two copies of the same dim image -> duplicate + too_dark.
    dim = make_image(10, style="dim")
    return [("dim1.png", dim), ("dim2.png", dim)]


@pytest.fixture
def mumbai_gps():
    from api.schemas import GPS
    return GPS(lat=19.07, lng=72.87)


@pytest.fixture
def rural_gps():
    from api.schemas import GPS
    return GPS(lat=15.50, lng=75.90)
