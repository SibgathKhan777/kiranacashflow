"""FastAPI app exposing the underwriting pipeline.

POST /underwrite
  multipart/form-data:
    images: 3-5 image files
    lat:    float
    lng:    float
    shop_size_sqft?: float
    monthly_rent?:   float
    years_in_operation?: float

Returns UnderwriteResponse JSON.
"""
from __future__ import annotations

from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from api.schemas import GPS, OwnerDeclared, UnderwriteResponse
from pipelines.decision import assemble_decision
from pipelines.fraud import run_fraud_validators
from pipelines.fusion import estimate_cashflow
from pipelines.geo import extract_geo_features
from pipelines.ingest import assess_images
from pipelines.vision import extract_vision_features


app = FastAPI(
    title="Kirana Underwriting API",
    version="0.1.0",
    description=(
        "Remote cash-flow underwriting for Indian kirana stores. "
        "Phase 1 skeleton with real ingestion, structural fusion, and fraud checks."
    ),
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "phase": 1}


@app.post("/underwrite", response_model=UnderwriteResponse)
async def underwrite(
    lat: float = Form(...),
    lng: float = Form(...),
    shop_size_sqft: float | None = Form(None),
    monthly_rent: float | None = Form(None),
    years_in_operation: float | None = Form(None),
    images: list[UploadFile] = File(...),
) -> UnderwriteResponse:
    if not 1 <= len(images) <= 8:
        raise HTTPException(400, "Provide between 1 and 8 images.")

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
    return assemble_decision(vision, geo, estimate, flags)
