# Kirana Underwriting

Remote cash-flow underwriting for Indian kirana stores using only photos + GPS.
Estimates daily sales / monthly revenue / monthly income as **calibrated ranges**,
not single-point predictions.

This repository is the **Phase 1 skeleton**: the full end-to-end pipeline runs
with real ingestion, real structural-economic fusion, real fraud validators, and
*stubbed* vision/geo feature extractors. Phases 2–5 swap in real models.

## Architecture

```
[images + GPS]
   -> ingestion & QA          (real)
   -> vision feature extract  (stub, deterministic from image stats)
   -> geo feature extract     (stub, deterministic from lat/lng)
   -> structural fusion       (real economic priors)
   -> fraud validators        (real cross-signal checks)
   -> decision layer          (real)
   -> JSON underwriting verdict
```

## Quickstart

```bash
cd kirana-underwriting
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 1. Run the API
uvicorn api.main:app --reload --port 8000

# 2. In another shell, run the Streamlit demo
streamlit run demo/streamlit_app.py

# 3. Or hit the API directly
python -m examples.run_sample
```

## Layout

```
kirana-underwriting/
├── api/                FastAPI app + schemas
├── pipelines/
│   ├── ingest/         image quality & shop-type routing
│   ├── vision/         visual feature extractors (stubbed)
│   ├── geo/            geo-spatial feature extractors (stubbed)
│   ├── fusion/         structural prior + cash-flow estimator
│   ├── fraud/          cross-signal fraud validators
│   └── decision/       final JSON assembler
├── data/lookups/       price bands, margins, tier priors
├── demo/               Streamlit UI
├── examples/           sample request runner
└── tests/              unit tests
```

## What's real vs stubbed (Phase 1)

| Stage | Status |
|---|---|
| Ingestion / image QA | Real (blur, brightness, dedup, coverage) |
| Shop-type router | Real (rule-based on image stats) |
| Vision features | **Stub** — derived deterministically from image hash + stats |
| Geo features | **Stub** — derived deterministically from lat/lng |
| Structural fusion | Real economic equations |
| Quantile ML head | **Stub** — placeholder, returns prior ± 25% |
| Fraud validators | Real cross-signal logic |
| Decision layer | Real |
| API + demo | Real |
```
