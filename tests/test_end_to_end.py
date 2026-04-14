"""End-to-end pipeline test via the FastAPI TestClient."""
from __future__ import annotations

import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from api.database import init_db
from api.main import app

# Use context manager so startup event fires and DB tables are created.
@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def _png_bytes(seed: int) -> bytes:
    rng = np.random.default_rng(seed)
    arr = rng.integers(40, 220, size=(600, 800, 3), dtype=np.uint8)
    for _ in range(60):
        x, y = rng.integers(0, 740), rng.integers(0, 540)
        arr[y : y + 60, x : x + 60] = rng.integers(30, 255, size=3)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_underwrite_roundtrip(client):
    files = [
        ("images", ("a.png", _png_bytes(1), "image/png")),
        ("images", ("b.png", _png_bytes(2), "image/png")),
        ("images", ("c.png", _png_bytes(3), "image/png")),
    ]
    data = {"lat": "19.07", "lng": "72.87", "shop_size_sqft": "150"}
    r = client.post("/underwrite", files=files, data=data)
    assert r.status_code == 200, r.text
    body = r.json()
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
    assert body["daily_sales_range"][0] <= body["daily_sales_range"][1]
    assert body["recommendation"] in {
        "approve_for_underwriting",
        "needs_verification",
        "reject_or_field_visit",
    }
    # DB was written — history endpoint should return ≥1 record
    h = client.get("/history")
    assert h.status_code == 200
    assert len(h.json()) >= 1


def test_stats(client):
    s = client.get("/stats")
    assert s.status_code == 200
    body = s.json()
    assert "total" in body and "approved" in body


def test_underwrite_rejects_no_images(client):
    r = client.post("/underwrite", data={"lat": "19.07", "lng": "72.87"})
    assert r.status_code in (400, 422)
