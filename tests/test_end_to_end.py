"""End-to-end pipeline test via the FastAPI TestClient."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.main import app


client = TestClient(app)


def _png_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 220, size=(600, 800, 3), dtype=np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, 740), rng.integers(0, 540)
        arr[y : y + 60, x : x + 60] = rng.integers(30, 255, size=3)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_underwrite_roundtrip():
    files = [
        ("images", ("a.png", _png_bytes(1), "image/png")),
        ("images", ("b.png", _png_bytes(2), "image/png")),
        ("images", ("c.png", _png_bytes(3), "image/png")),
    ]
    data = {"lat": "19.07", "lng": "72.87", "shop_size_sqft": "150"}
    r = client.post("/underwrite", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
    # Required fields present.
    for key in (
        "daily_sales_range",
        "monthly_revenue_range",
        "monthly_income_range",
        "confidence_score",
        "risk_flags",
        "recommendation",
        "explainability",
    ):
        assert key in body
    # Ranges ordered.
    assert body["daily_sales_range"][0] <= body["daily_sales_range"][1]
    assert body["recommendation"] in {
        "approve_for_underwriting",
        "needs_verification",
        "reject_or_field_visit",
    }


def test_underwrite_rejects_no_images():
    r = client.post("/underwrite", data={"lat": "19.07", "lng": "72.87"})
    # FastAPI returns 422 when the required File field is missing.
    assert r.status_code in (400, 422)
