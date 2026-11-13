# Kirana Underwriting (KiranaLens) — Design

This document covers the HTTP API, the underwriting pipeline behind it, how the decision is made,
what gets stored, and the design decisions and known limits. For setup and sample output, see
[README.md](README.md).

## 1. System overview

The service estimates a kirana store's cash flow from **1–8 photos and a GPS location** only (no
transaction data, GST records or field survey) and returns a loan-ready verdict.

```
POST /underwrite (photos + GPS + optional owner-declared facts)
        │
        ▼
 ingest/quality   →  image QA + role tagging (shelf, counter, exterior, street ...)
        │
        ├──▶ vision/   shop type, shelf density, SKU mix, inventory value, POS/brand signals
        └──▶ geo/      OpenStreetMap (Overpass): buildings, POIs, road class, competition, tier
        │
        ▼
 fusion/estimator →  demand-side × supply-side estimate, calibrated by LightGBM q10/q50/q90
        │
        ▼
 fraud/validators →  cross-signal checks → risk flags
        │
        ▼
 decision/assembler → confidence, recommendation, explanation, loan sizing, peer benchmark
        │
        ▼
 SQLite (underwriting_records)  +  JSON response
```

| Layer | Module | Notes |
|---|---|---|
| API | `api/main.py`, `api/schemas.py` | FastAPI; OpenAPI docs at `/docs`; serves `demo/index.html` at `/` |
| Persistence | `api/database.py` | SQLAlchemy over SQLite at `data/underwriting.db` |
| Image QA | `pipelines/ingest/quality.py` | Blur (Laplacian), brightness, perceptual-hash dedup, watermark, resolution |
| Vision | `pipelines/vision/` | 5×8 shelf grid, HSV colour → category, 40 SKU price bands |
| Geo | `pipelines/geo/` | Overpass client with a 7-day SQLite cache; `KIRANA_GEO_STUB=1` for offline runs |
| Fusion | `pipelines/fusion/` | Structural estimate + quantile LightGBM models in `models/quantile_lgbm/` |
| Fraud | `pipelines/fraud/validators.py` | Cross-signal risk flags |
| Decision | `pipelines/decision/assembler.py` | Confidence, recommendation, explanation |
| Lending | `pipelines/lending/` | FOIR-based loan sizing, EMI table, peer percentile |

## 2. HTTP API

| Method | Path | Purpose | Errors |
|---|---|---|---|
| `GET` | `/` | Browser demo (`demo/index.html`) | — |
| `GET` | `/health` | `{status: "ok", version: "2.0.0"}` | — |
| `POST` | `/underwrite` | Run the full pipeline, store the result, return the verdict | `400` image count outside 1–8, or an empty file; `422` invalid form fields |
| `GET` | `/history?limit=20&offset=0` | Summaries of recent decisions, newest first | — |
| `GET` | `/history/{id}` | One stored record, including the full original response | `404` not found |
| `GET` | `/stats` | `{total, approved, verify, rejected}` | — |

### 2.1 `POST /underwrite`

`multipart/form-data`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `images` | file[] | yes | 1–8 JPEG/PNG photos. Best results: interior shelves, counter, exterior |
| `lat` | float | yes | −90 to 90 |
| `lng` | float | yes | −180 to 180 |
| `shop_size_sqft` | float | no | Owner-declared; used as a hint, never trusted blindly |
| `monthly_rent` | float | no | Owner-declared |
| `years_in_operation` | float | no | Owner-declared |

```bash
curl -X POST http://localhost:8000/underwrite \
  -F lat=12.9716 -F lng=77.5946 \
  -F images=@shelf1.jpg -F images=@counter.jpg -F images=@front.jpg
```

### 2.2 Response (`UnderwriteResponse`)

All amounts are in INR. Ranges are `[low, high]` (the q10 and q90 bands).

| Field | Type | Notes |
|---|---|---|
| `daily_sales_range` | `[float, float]` | |
| `monthly_revenue_range` | `[float, float]` | |
| `monthly_income_range` | `[float, float]` | |
| `confidence_score` | float | 0.05–0.99 |
| `risk_flags` | string[] | Flag codes (section 4) |
| `recommendation` | enum | `approve_for_underwriting`, `needs_verification`, `reject_or_field_visit` |
| `explainability` | object | See below |

`explainability` contains:

- `shop_type`, `tier`, `top_drivers` (the 4 strongest signals, e.g. `geo_demand=0.72`)
- `structural_point_estimate`, `model_point_estimate`, `prior_disagreement`
- `flag_details[]`: `{code, severity, message}`
- `vision_features`, `geo_features`: the full feature sets used
- `loan_sizing`: `eligible`, `recommended_loan`, `max_eligible_loan`, `confidence_haircut`,
  `foir_ceiling`, `risk_adjustment`, `ineligible_reason`, and `emi_options[]` with
  `{tenure_months, emi_amount, total_repayment, foir}` for 6/12/18/24 months
- `peer_benchmark`: `peer_cohort`, `percentile`, cohort p25/median/p75 daily sales,
  `this_shop_daily`, `interpretation`

## 3. Estimation method

Two independent estimates are computed and blended:

- **Demand side:** `footfall × conversion_rate × ticket_size` (from geo features, tier priors and
  SKU diversity).
- **Supply side:** `inventory_value / turnover_days × refill_signal`. Inventory value is the visible
  shelf value (grid cells × category price band) multiplied by about 3 for unseen stock, and capped
  by store size.

The blend is fed to three LightGBM quantile models (q10 / q50 / q90) trained on 2,000 synthetic
samples, which produce the calibrated range. `prior_disagreement` (0–1) measures how far the two
sides disagree, and it lowers confidence.

## 4. Risk flags and decision

### 4.1 Risk flags

| Group | Codes |
|---|---|
| Coverage | `insufficient_images`, `interior_view_missing`, `exterior_view_missing`, `exterior_only_submission` |
| Image integrity | `stock_photo_suspect`, `duplicate_images` |
| Shop type | `non_kirana_shop_type`, `shop_type_unclassified` |
| Cross-signal | `inventory_footfall_mismatch`, `under_stocked_for_location`, `staged_inventory_suspect`, `demand_supply_disagreement` |

Each flag has a severity of `low`, `medium` or `high`.

### 4.2 Confidence

```
confidence = 0.3 + 0.5 × extraction_confidence + 0.2 × coverage_score
           − 0.3 × prior_disagreement
           − Σ penalty(flag)            low 0.05 · medium 0.15 · high 0.40
clamped to [0.05, 0.99]
```

### 4.3 Recommendation

Checked in order:

1. Any `high` flag → `reject_or_field_visit`
2. Confidence ≥ 0.75 and no `medium` flag → `approve_for_underwriting`
3. Confidence < 0.4 → `reject_or_field_visit`
4. Otherwise → `needs_verification`

### 4.4 Loan sizing

- FOIR (fixed obligations to income) ceiling of **50%** of net income for a clean profile, and
  **35%** when confidence is low or flags are present.
- The loan is sized on the **low end** of the income range, then reduced by a confidence haircut.
- Rates range from 18% (good profile) to 26% (high risk), with EMIs shown for 6, 12, 18 and 24
  months.

## 5. Data model

SQLite table `underwriting_records` (one row per `/underwrite` call):

| Column | Notes |
|---|---|
| `id`, `created_at` | Primary key; UTC timestamp |
| `lat`, `lng`, `shop_size_sqft`, `monthly_rent`, `years_in_operation` | Inputs |
| `image_count`, `image_names` | Image names stored as a JSON list. **The images themselves are not stored** |
| `recommendation`, `confidence_score`, `risk_flags` | Decision (flags as a JSON array) |
| `daily_sales_low/high`, `monthly_revenue_low/high`, `monthly_income_low/high` | Ranges flattened for querying |
| `shop_type`, `tier` | For filtering and drift tracking |
| `response_json` | Full response, for audit and replay |
| `processing_ms` | Pipeline latency |

## 6. Design decisions

- **Ranges, not single numbers.** Photos cannot pin down a shop's sales exactly. A calibrated range
  is more honest and lets the lender size the loan on the low end.
- **Two independent estimates.** Demand (location) and supply (stock) are estimated separately, so
  when they disagree, confidence drops instead of the error being hidden.
- **Explain every decision.** The response includes the drivers, features and flag messages that
  produced it, so a credit officer can see why.
- **Owner-declared facts are hints only.** They can inform the estimate but never override the
  observed signals.
- **Built to be skeptical of the photos.** Duplicate, stock, watermarked or staged photos, and stock
  levels that don't fit the location, are flagged instead of accepted.
- **Runs offline.** `KIRANA_GEO_STUB=1` replaces Overpass with deterministic geo data, so the full
  pipeline and tests (`pytest`) run without network access.
- **Store the full response.** Saving `response_json` lets any decision be audited or replayed
  later and supports tracking model drift.

## 7. Known limits

- **Trained on synthetic data.** The quantile models and peer cohorts come from 2,000 synthetic
  samples. They need calibrating against real loan outcomes before any real lending decision.
- **Rule-based vision.** Product detection uses colour and grid heuristics, not a trained object
  detector. Unusual lighting or shelf layouts reduce accuracy.
- **No authentication, and CORS allows any origin.** This is fine for a demo, but a deployment
  needs auth and a restricted origin list.
- **SQLite.** Suitable for a single instance only. Concurrent writers or multiple replicas would
  need Postgres or similar.
- **Overpass dependency.** The public Overpass API is rate-limited. The 7-day cache helps, but
  heavy use needs a self-hosted instance.
