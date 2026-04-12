# Kirana Underwriting

Remote cash-flow underwriting for Indian kirana stores using **only photos + GPS** — no transaction data, no GST records, no manual surveys.

Estimates daily sales, monthly revenue, and monthly income as **calibrated ranges** (not single-point predictions), with fraud detection, loan sizing, and peer benchmarking.

---

## Architecture

```
[3-5 images + GPS coordinates]
            │
            ▼
┌─────────────────────────────┐
│  Ingestion & QA             │  blur, brightness, dedup, watermark,
│                             │  role tagging, resolution checks
└─────────────────────────────┘
            │
   ┌────────┴────────┐
   ▼                 ▼
┌──────────────┐  ┌──────────────────┐
│ Vision       │  │ Geo (Overpass)   │
│ Pipeline     │  │ Pipeline         │
│              │  │                  │
│ • Grid-based │  │ • OSM buildings  │
│   product    │  │ • POI footfall   │
│   detection  │  │ • Road class     │
│ • HSV color  │  │ • Competition    │
│   → category │  │ • Tier classify  │
│ • Price band │  │ • Catchment pop  │
│   lookup     │  │                  │
│ • POS/brand  │  │ Offline fallback │
│   detection  │  │ to deterministic │
│ • Fast/slow  │  │ stub             │
│   mix        │  └──────────────────┘
└──────────────┘
        │                 │
        └────────┬─────────┘
                 ▼
┌────────────────────────────────┐
│  Structural Fusion             │
│                                │
│  DEMAND-SIDE:                  │
│  footfall × conversion × ticket│
│                                │
│  SUPPLY-SIDE:                  │
│  inventory / turnover_days     │
│                                │
│  + LightGBM q10 / q50 / q90   │
│    calibration head            │
└────────────────────────────────┘
                 │
                 ▼
┌────────────────────────────────┐
│  Fraud Validators (7 checks)   │
│  staged inventory, mismatch,   │
│  stock photos, duplicates ...  │
└────────────────────────────────┘
                 │
        ┌────────┼─────────┐
        ▼        ▼         ▼
┌──────────┐ ┌────────┐ ┌──────────┐
│ Decision │ │ Loan   │ │ Peer     │
│ + conf   │ │ Sizing │ │ Benchmark│
│ + flags  │ │ + EMI  │ │ + rank   │
└──────────┘ └────────┘ └──────────┘
                 │
                 ▼
         JSON Verdict
```

---

## What's inside

| Stage | Status | Detail |
|---|---|---|
| Image QA | ✅ Real | Blur (Laplacian), brightness, dedup (perceptual hash), watermark detection, resolution check |
| Role classification | ✅ Real | Spatial zone analysis, edge profiles, color histograms → interior/counter/exterior/street |
| Shop-type routing | ✅ Real | packaged_kirana / loose_goods / street_stall / organized_retail / exterior_only |
| Product detection | ✅ Real | 5×8 grid segmentation, HSV color → category matching, 40 SKU price bands |
| Inventory value | ✅ Real | Visible value → visibility-corrected total (×3.0 base), store-size cap |
| Fast/slow mix | ✅ Real | Turnover days per category → fast (≤7d) / mid / slow (>21d) fractions |
| POS/brand detection | ✅ Real | Paytm blue, PhonePe purple, Coca-Cola red, QR code regions |
| Geo features | ✅ Real | Overpass API: buildings, POIs, road class, competing shops (sqlite cached, 7-day TTL) |
| Tier classification | ✅ Real | Metro bounding boxes + tier-1 city radii + India-wide fallback |
| Structural fusion | ✅ Real | Demand-side × supply-side economic equations with blending weights |
| Quantile ML head | ✅ Real | LightGBM q10/q50/q90 trained on 2,000 synthetic samples |
| Fraud validators | ✅ Real | 7 cross-signal checks (staged inventory, footfall mismatch, stock photos, duplicates…) |
| Decision layer | ✅ Real | Confidence scoring, recommendation routing, full explainability |
| Loan sizing | ✅ Real | FOIR check, confidence haircut, EMI table (6/12/18/24 months, 18-26% rate) |
| Peer benchmarking | ✅ Real | Percentile rank vs simulated cohort by tier + shop type |
| API | ✅ Real | FastAPI, multipart upload, OpenAPI docs |
| Streamlit demo | ✅ Real | 4 pre-loaded scenarios, image thumbnails, charts, EMI table |

---

## Quickstart

```bash
cd kirana-underwriting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Run the Streamlit demo (recommended)
streamlit run demo/streamlit_app.py

# Or run the API
uvicorn api.main:app --reload --port 8000
# Open http://localhost:8000/docs for Swagger UI

# Run the smoke test (3 scenarios, no network needed)
KIRANA_GEO_STUB=1 python examples/run_sample.py

# Run tests (39 tests, ~12 seconds)
KIRANA_GEO_STUB=1 python -m pytest -q
```

---

## Sample output

```json
{
  "daily_sales_range": [7273, 21057],
  "monthly_revenue_range": [203636, 589596],
  "monthly_income_range": [17105, 88439],
  "confidence_score": 0.71,
  "risk_flags": [],
  "recommendation": "needs_verification",
  "explainability": {
    "shop_type": "packaged_kirana",
    "tier": "metro",
    "top_drivers": ["geo_demand=0.72", "shelf_density=0.65", "sku_diversity=0.61", "tier=metro"],
    "structural_point_estimate": 13886,
    "model_point_estimate": 14165,
    "prior_disagreement": 0.23
  }
}
```

---

## How values are calculated

### Inventory value (real product detection)
1. Each interior image is divided into a **5×8 grid** of shelf cells
2. Each cell's dominant HSV color is matched to a product category (staples / fmcg / beverages / personal care / tobacco / dairy / loose goods)
3. The matched category's **price band** is looked up (e.g. fmcg cell = ₹336/facing avg)
4. Visible value = Σ(cell values); Total = visible × visibility multiplier (~3×)

### Cash flow (structural prior + ML)
Two independent estimates are computed and blended:

- **Demand-side:** `footfall × conversion_rate × ticket_size` (from geo + SKU diversity)
- **Supply-side:** `inventory_value / turnover_days × refill_signal`

The blend is passed to a **LightGBM quantile model** (q10/q50/q90) for calibrated uncertainty bands.

### Confidence score
```
confidence = 0.3 + 0.5 × extraction_confidence + 0.2 × coverage_score
           - 0.3 × prior_disagreement - Σ(flag_penalties)
```

---

## Repository layout

```
kirana-underwriting/
├── api/
│   ├── main.py              FastAPI app, /health + /underwrite
│   └── schemas.py           Pydantic request/response models
├── pipelines/
│   ├── ingest/quality.py    Image QA, role classification
│   ├── vision/
│   │   ├── classifiers.py   Shop-type router, brand/POS detection
│   │   ├── extractor.py     Main vision feature extractor
│   │   └── inventory.py     Product detection + price band mapping
│   ├── geo/
│   │   ├── extractor.py     Geo features (Overpass + stub fallback)
│   │   └── overpass.py      Cached Overpass API client
│   ├── fusion/
│   │   ├── estimator.py     Structural prior + quantile head
│   │   └── ml/              LightGBM training + inference
│   ├── fraud/validators.py  7 cross-signal fraud checks
│   ├── decision/assembler.py Final verdict assembler
│   └── lending/
│       ├── loan_sizer.py    EMI affordability calculator
│       └── peer_benchmark.py Percentile ranking vs cohort
├── data/lookups/
│   ├── price_bands.json     40 SKUs with avg price + turnover
│   ├── category_margins.json Margin bands per category
│   └── tier_priors.json     Footfall + ticket size per tier
├── models/quantile_lgbm/    Trained q10/q50/q90 models
├── demo/streamlit_app.py    Streamlit UI with 4 demo scenarios
├── examples/run_sample.py   End-to-end smoke test
└── tests/                   39 pytest tests
```

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `KIRANA_GEO_STUB` | `0` | Set to `1` to skip Overpass and use deterministic stub (for tests / offline dev) |

---

## Dependencies

No GPU required. All models run on CPU.

```
fastapi, uvicorn, pydantic        API
pillow, numpy                     Image processing
lightgbm, scikit-learn            Quantile calibration
requests, requests-cache          Overpass geo (sqlite cache)
streamlit                         Demo UI
pytest                            Tests
```
