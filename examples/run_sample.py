"""End-to-end smoke test.

Generates synthetic images in-memory (different "shops") and runs the full
pipeline for each. No network, no files on disk. Useful both as a CI smoke
test and as a quick sanity check after changes.
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.schemas import GPS, OwnerDeclared  # noqa: E402
from pipelines.decision import assemble_decision  # noqa: E402
from pipelines.fraud import run_fraud_validators  # noqa: E402
from pipelines.fusion import estimate_cashflow  # noqa: E402
from pipelines.geo import extract_geo_features  # noqa: E402
from pipelines.ingest import assess_images  # noqa: E402
from pipelines.vision import extract_vision_features  # noqa: E402


def _synthetic_image(seed: int, size=(800, 600), style: str = "busy") -> bytes:
    rng = np.random.default_rng(seed)
    if style == "busy":
        # Lots of colored rectangles = dense packaged shelves.
        arr = rng.integers(40, 220, size=(size[1], size[0], 3), dtype=np.uint8)
        for _ in range(80):
            x, y = rng.integers(0, size[0] - 60), rng.integers(0, size[1] - 60)
            w, h = rng.integers(20, 80), rng.integers(30, 100)
            color = rng.integers(30, 255, size=3)
            arr[y : y + h, x : x + w] = color
    elif style == "dim":
        arr = rng.integers(10, 70, size=(size[1], size[0], 3), dtype=np.uint8)
    elif style == "bright_exterior":
        # Top third sky-like (bright + blueish), bottom two-thirds colorful.
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[: size[1] // 3] = [180, 200, 230]
        arr[size[1] // 3 :] = rng.integers(80, 200, size=(size[1] - size[1] // 3, size[0], 3))
    else:
        arr = rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def _scenario_clean_metro():
    files = [
        ("interior.png", _synthetic_image(1, style="busy")),
        ("counter.png", _synthetic_image(2, style="busy")),
        ("exterior.png", _synthetic_image(3, style="bright_exterior")),
    ]
    gps = GPS(lat=19.07, lng=72.87)  # Mumbai
    return "clean_metro_shop", files, gps, OwnerDeclared(shop_size_sqft=180)


def _scenario_poor_inputs():
    files = [
        ("dim1.png", _synthetic_image(10, style="dim")),
        ("dim2.png", _synthetic_image(10, style="dim")),  # duplicate on purpose
    ]
    gps = GPS(lat=21.20, lng=79.10)  # Nagpur (tier1)
    return "poor_inputs", files, gps, OwnerDeclared()


def _scenario_rural():
    files = [
        ("interior.png", _synthetic_image(5, style="busy")),
        ("exterior.png", _synthetic_image(6, style="bright_exterior")),
    ]
    gps = GPS(lat=15.50, lng=75.90)  # rural Karnataka
    return "rural_shop", files, gps, OwnerDeclared(shop_size_sqft=90, years_in_operation=8)


def _run(name, files, gps, owner):
    records = assess_images(files)
    vision = extract_vision_features(records)
    geo = extract_geo_features(gps)
    estimate = estimate_cashflow(vision, geo, owner)
    flags = run_fraud_validators(records, vision, geo, estimate)
    decision = assemble_decision(vision, geo, estimate, flags)

    print(f"\n=== Scenario: {name} ===")
    print(f"tier={geo.tier}  shop_type={vision.shop_type}  "
          f"confidence={decision.confidence_score}  -> {decision.recommendation}")
    print(f"daily_sales:     {decision.daily_sales_range}")
    print(f"monthly_revenue: {decision.monthly_revenue_range}")
    print(f"monthly_income:  {decision.monthly_income_range}")
    if decision.risk_flags:
        print(f"flags: {decision.risk_flags}")
    else:
        print("flags: none")


def main() -> int:
    for scenario in (_scenario_clean_metro, _scenario_poor_inputs, _scenario_rural):
        _run(*scenario())
    print("\nSmoke test completed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
