"""FastAPI app — Kirana underwriting pipeline.

Endpoints
---------
GET  /health                  → liveness check
POST /underwrite              → run full pipeline, store result, return JSON
GET  /history                 → last N underwriting decisions
GET  /history/{id}            → single record (full response JSON)
GET  /stats                   → aggregate counts (approved / verify / rejected)

Multipart form for /underwrite
  images              File[]  1–8 JPEG/PNG photos
  lat                 float   decimal degrees
  lng                 float   decimal degrees
  shop_size_sqft      float?
  monthly_rent        float?
  years_in_operation  float?
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy.orm import Session

from api.database import get_db, get_stats, init_db, list_records, get_record, save_record
from api.schemas import GPS, OwnerDeclared, UnderwriteResponse
from pipelines.decision import assemble_decision
from pipelines.fraud import run_fraud_validators
from pipelines.fusion import estimate_cashflow
from pipelines.geo import extract_geo_features
from pipelines.ingest import assess_images
from pipelines.vision import extract_vision_features

# ── Lifespan (replaces deprecated on_event) ───────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    yield


# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="KiranaLens Underwriting API",
    version="2.0.0",
    description=(
        "Remote cash-flow underwriting for Indian kirana stores. "
        "GPS + photos → loan decision. No GST, no transactions needed."
    ),
    lifespan=lifespan,
)

# ── CORS — allow browser fetch from any origin (demo / localhost) ─────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Static files — serve demo/index.html at / ────────────────────────────────
_DEMO_DIR = Path(__file__).resolve().parents[1] / "demo"

@app.get("/", include_in_schema=False)
def serve_ui():
    return FileResponse(_DEMO_DIR / "index.html")


# Mount any other static assets (css, js, images if added later)
if _DEMO_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(_DEMO_DIR)), name="static")


# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok", "version": "2.0.0"}


# ── Stats ─────────────────────────────────────────────────────────────────────
@app.get("/stats", tags=["ops"])
def stats(db: Session = Depends(get_db)) -> dict:
    return get_stats(db)


# ── History ───────────────────────────────────────────────────────────────────
@app.get("/history", tags=["history"])
def history(
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
) -> list[dict]:
    records = list_records(db, limit=limit, offset=offset)
    return [
        {
            "id": r.id,
            "created_at": r.created_at.isoformat() if r.created_at else None,
            "lat": r.lat,
            "lng": r.lng,
            "image_count": r.image_count,
            "shop_type": r.shop_type,
            "tier": r.tier,
            "recommendation": r.recommendation,
            "confidence_score": r.confidence_score,
            "daily_sales_low": r.daily_sales_low,
            "daily_sales_high": r.daily_sales_high,
            "monthly_income_low": r.monthly_income_low,
            "monthly_income_high": r.monthly_income_high,
            "risk_flags": r.risk_flags,   # JSON string
            "processing_ms": r.processing_ms,
        }
        for r in records
    ]


@app.get("/history/{record_id}", tags=["history"])
def history_detail(record_id: int, db: Session = Depends(get_db)) -> dict:
    r = get_record(db, record_id)
    if not r:
        raise HTTPException(404, f"Record {record_id} not found")
    import json
    return {
        "id": r.id,
        "created_at": r.created_at.isoformat() if r.created_at else None,
        "lat": r.lat,
        "lng": r.lng,
        "shop_size_sqft": r.shop_size_sqft,
        "monthly_rent": r.monthly_rent,
        "years_in_operation": r.years_in_operation,
        "image_count": r.image_count,
        "image_names": json.loads(r.image_names or "[]"),
        "recommendation": r.recommendation,
        "confidence_score": r.confidence_score,
        "risk_flags": json.loads(r.risk_flags or "[]"),
        "processing_ms": r.processing_ms,
        "response": json.loads(r.response_json),
    }


# ── Core underwriting endpoint ────────────────────────────────────────────────
@app.post("/underwrite", response_model=UnderwriteResponse, tags=["underwriting"])
async def underwrite(
    lat: float = Form(...),
    lng: float = Form(...),
    shop_size_sqft: float | None = Form(None),
    monthly_rent: float | None = Form(None),
    years_in_operation: float | None = Form(None),
    images: list[UploadFile] = File(...),
    db: Session = Depends(get_db),
) -> UnderwriteResponse:
    if not 1 <= len(images) <= 8:
        raise HTTPException(400, "Provide between 1 and 8 images.")

    t0 = time.monotonic()

    files: list[tuple[str, bytes]] = []
    for f in images:
        content = await f.read()
        if not content:
            raise HTTPException(400, f"Empty upload: {f.filename}")
        files.append((f.filename or "unnamed", content))

    gps = GPS(lat=lat, lng=lng)
    owner = OwnerDeclared(
        shop_size_sqft=shop_size_sqft,
        monthly_rent=monthly_rent,
        years_in_operation=years_in_operation,
    )

    records = assess_images(files)
    vision = extract_vision_features(records)
    geo = extract_geo_features(gps)
    estimate = estimate_cashflow(vision, geo, owner)
    flags = run_fraud_validators(records, vision, geo, estimate)
    response = assemble_decision(vision, geo, estimate, flags)

    processing_ms = int((time.monotonic() - t0) * 1000)

    # Persist to DB
    save_record(
        db,
        lat=lat,
        lng=lng,
        shop_size_sqft=shop_size_sqft,
        monthly_rent=monthly_rent,
        years_in_operation=years_in_operation,
        image_names=[name for name, _ in files],
        response=response,
        processing_ms=processing_ms,
    )

    return response
