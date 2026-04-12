"""Streamlit demo — Kirana Cash Flow Underwriting Platform.

Runs the full pipeline in-process. Includes:
  - Image upload + GPS input
  - Pre-loaded demo scenarios (clean / fraud / edge case)
  - Cash flow estimation with ranges
  - Loan sizing + EMI table
  - Peer benchmarking
  - Full explainability

  streamlit run demo/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from api.schemas import GPS, OwnerDeclared  # noqa: E402
from pipelines.decision import assemble_decision  # noqa: E402
from pipelines.fraud import run_fraud_validators  # noqa: E402
from pipelines.fusion import estimate_cashflow  # noqa: E402
from pipelines.geo import extract_geo_features  # noqa: E402
from pipelines.ingest import assess_images  # noqa: E402
from pipelines.lending.loan_sizer import size_loan  # noqa: E402
from pipelines.lending.peer_benchmark import benchmark  # noqa: E402
from pipelines.vision import extract_vision_features  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic image generators for demo scenarios
# ---------------------------------------------------------------------------

def _synth(seed, size=(800, 600), style="busy"):
    rng = np.random.default_rng(seed)
    if style == "busy":
        arr = rng.integers(80, 220, size=(size[1], size[0], 3), dtype=np.uint8)
        for _ in range(100):
            x, y = rng.integers(0, size[0] - 60), rng.integers(0, size[1] - 60)
            w, h = rng.integers(20, 80), rng.integers(30, 100)
            arr[y : y + h, x : x + w] = rng.integers(40, 255, size=3)
        # Horizontal shelf lines
        for row in range(0, size[1], 80):
            arr[row : row + 3] = [60, 60, 60]
        return arr
    if style == "dim":
        return rng.integers(10, 60, size=(size[1], size[0], 3), dtype=np.uint8)
    if style == "exterior":
        arr = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        arr[: size[1] // 3] = [170, 200, 235]
        arr[size[1] // 3 :] = rng.integers(80, 190, size=(size[1] - size[1] // 3, size[0], 3))
        return arr
    if style == "staged":
        # Very uniform bright shelves (fraud signal: too perfect)
        arr = np.full((size[1], size[0], 3), 180, dtype=np.uint8)
        for row in range(0, size[1], 50):
            arr[row : row + 2] = [80, 80, 80]
        for col in range(0, size[0], 40):
            color = rng.integers(100, 240, size=3)
            arr[:, col : col + 35] = color
        return arr
    return rng.integers(0, 255, size=(size[1], size[0], 3), dtype=np.uint8)


def _to_png(arr):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


DEMO_SCENARIOS = {
    "Clean Metro Shop (Mumbai)": {
        "description": "Well-stocked packaged kirana on a collector road in Mumbai. Should APPROVE.",
        "lat": 19.076, "lng": 72.878,
        "owner": {"shop_size_sqft": 180, "monthly_rent": 15000, "years_in_operation": 7},
        "images": [
            ("interior_shelves.png", _to_png(_synth(1, style="busy"))),
            ("counter.png", _to_png(_synth(2, style="busy"))),
            ("exterior.png", _to_png(_synth(3, style="exterior"))),
        ],
    },
    "Fraud: Staged Inventory": {
        "description": "Suspiciously perfect shelves in a low-demand area. Should REJECT.",
        "lat": 24.55, "lng": 84.12,
        "owner": {},
        "images": [
            ("staged_interior.png", _to_png(_synth(20, style="staged"))),
            ("staged_counter.png", _to_png(_synth(21, style="staged"))),
            ("exterior.png", _to_png(_synth(22, style="exterior"))),
        ],
    },
    "Edge: Poor Inputs (dim + duplicates)": {
        "description": "Two dim duplicate images, no exterior. Should REJECT or VERIFY.",
        "lat": 21.15, "lng": 79.09,
        "owner": {},
        "images": [
            ("dim1.png", _to_png(_synth(10, style="dim"))),
            ("dim2.png", _to_png(_synth(10, style="dim"))),
        ],
    },
    "Rural Small Shop (Karnataka)": {
        "description": "Small tier-2/3 shop with limited inventory. Should VERIFY.",
        "lat": 15.50, "lng": 75.90,
        "owner": {"shop_size_sqft": 80, "years_in_operation": 12},
        "images": [
            ("interior.png", _to_png(_synth(5, style="busy", size=(600, 800)))),
            ("exterior.png", _to_png(_synth(6, style="exterior"))),
        ],
    },
}


def run_pipeline(files, lat, lng, owner_kw):
    records = assess_images(files)
    gps = GPS(lat=lat, lng=lng)
    owner = OwnerDeclared(**{k: v for k, v in owner_kw.items() if v})
    vision = extract_vision_features(records)
    geo = extract_geo_features(gps)
    est = estimate_cashflow(vision, geo, owner)
    flags = run_fraud_validators(records, vision, geo, est)
    decision = assemble_decision(vision, geo, est, flags)
    loan = size_loan(decision, est)
    peer = benchmark(vision, geo, est)
    return records, vision, geo, est, flags, decision, loan, peer


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Kirana Underwriting", page_icon="🏪", layout="wide")

st.markdown("""
<style>
.rec-approve { background:#d4edda; border-left:5px solid #28a745; padding:12px 16px; border-radius:4px; font-size:1.2em; font-weight:bold; color:#155724; }
.rec-verify  { background:#fff3cd; border-left:5px solid #ffc107; padding:12px 16px; border-radius:4px; font-size:1.2em; font-weight:bold; color:#856404; }
.rec-reject  { background:#f8d7da; border-left:5px solid #dc3545; padding:12px 16px; border-radius:4px; font-size:1.2em; font-weight:bold; color:#721c24; }
.peer-bar { height:20px; border-radius:10px; background:linear-gradient(90deg, #e8e8e8 0%, #4A90D9 50%, #e8e8e8 100%); position:relative; margin:8px 0; }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🏪 Kirana Cash Flow Underwriting")
st.markdown(
    "Estimate a kirana store's cash flow from **photos + GPS**. "
    "Includes loan sizing, peer benchmarking, and fraud detection."
)

# ---------------------------------------------------------------------------
# Input mode: upload or demo scenario
# ---------------------------------------------------------------------------
tab_upload, tab_demo = st.tabs(["📸 Upload Photos", "🧪 Demo Scenarios"])

run_result = None

with tab_upload:
    with st.sidebar:
        st.header("📍 Location")
        c1, c2 = st.columns(2)
        lat = c1.number_input("Latitude", value=19.0760, format="%.6f")
        lng = c2.number_input("Longitude", value=72.8777, format="%.6f")
        st.header("📋 Owner Declared")
        shop_size = st.number_input("Shop size (sqft)", min_value=0.0, value=0.0, step=10.0)
        rent = st.number_input("Monthly rent (₹)", min_value=0.0, value=0.0, step=500.0)
        years = st.number_input("Years in operation", min_value=0.0, value=0.0, step=1.0)

    uploads = st.file_uploader(
        "Upload 3-5 store photos", type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    if uploads:
        cols = st.columns(min(len(uploads), 5))
        for i, u in enumerate(uploads[:5]):
            cols[i].image(u, caption=u.name, use_container_width=True)

    if uploads and st.button("🔍 Run Analysis", type="primary", use_container_width=True, key="upload_run"):
        owner_kw = {
            "shop_size_sqft": shop_size or None,
            "monthly_rent": rent or None,
            "years_in_operation": years or None,
        }
        files = [(u.name, u.getvalue()) for u in uploads]
        with st.spinner("Running pipeline..."):
            run_result = run_pipeline(files, lat, lng, owner_kw)

with tab_demo:
    st.markdown("Select a pre-loaded scenario to see the system in action:")
    scenario_name = st.selectbox("Scenario", list(DEMO_SCENARIOS.keys()))
    sc = DEMO_SCENARIOS[scenario_name]
    st.info(sc["description"])

    # Show scenario images
    cols = st.columns(min(len(sc["images"]), 4))
    for i, (name, data) in enumerate(sc["images"][:4]):
        cols[i].image(data, caption=name, use_container_width=True)

    if st.button("🔍 Run Scenario", type="primary", use_container_width=True, key="demo_run"):
        with st.spinner("Running pipeline..."):
            run_result = run_pipeline(sc["images"], sc["lat"], sc["lng"], sc["owner"])


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------
if run_result:
    records, vision, geo, est, flags, decision, loan, peer = run_result

    st.divider()

    # === RECOMMENDATION BANNER ===
    rec = decision.recommendation
    rec_html = {
        "approve_for_underwriting": '<div class="rec-approve">APPROVE FOR UNDERWRITING</div>',
        "needs_verification":       '<div class="rec-verify">NEEDS FIELD VERIFICATION</div>',
        "reject_or_field_visit":    '<div class="rec-reject">REJECT / FIELD VISIT REQUIRED</div>',
    }
    st.markdown(rec_html.get(rec, ""), unsafe_allow_html=True)
    st.markdown("")

    # === KEY METRICS ===
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Daily Sales", f"₹{decision.daily_sales_range[0]:,.0f} – {decision.daily_sales_range[1]:,.0f}")
    m2.metric("Monthly Revenue", f"₹{decision.monthly_revenue_range[0]:,.0f} – {decision.monthly_revenue_range[1]:,.0f}")
    m3.metric("Monthly Income", f"₹{decision.monthly_income_range[0]:,.0f} – {decision.monthly_income_range[1]:,.0f}")
    m4.metric("Confidence", f"{decision.confidence_score:.0%}")

    st.divider()

    # === TWO-COLUMN MAIN BODY ===
    col_left, col_right = st.columns(2)

    # --- LEFT: Risk + Loan + Peer ---
    with col_left:
        # Risk flags
        st.subheader("🚩 Risk Flags")
        if decision.risk_flags:
            for detail in decision.explainability["flag_details"]:
                sev = detail["severity"]
                icon = {"low": "ℹ️", "medium": "⚠️", "high": "🚨"}[sev]
                st.markdown(f"{icon} **{detail['code']}** ({sev}) — {detail['message']}")
        else:
            st.success("No risk flags.")

        # Loan sizing
        st.subheader("💳 Loan Sizing")
        if loan.eligible:
            lc1, lc2 = st.columns(2)
            lc1.metric("Max Eligible", f"₹{loan.max_eligible_loan:,.0f}")
            lc2.metric("Recommended", f"₹{loan.recommended_loan:,.0f}")

            if loan.confidence_haircut > 0:
                st.caption(f"Confidence haircut: {loan.confidence_haircut:.0%} | FOIR ceiling: {loan.foir_ceiling:.0%}")

            st.markdown("**EMI Options (recommended loan):**")
            emi_data = []
            for opt in loan.emi_options:
                emi_data.append({
                    "Tenure": f"{opt.tenure_months} months",
                    "EMI": f"₹{opt.emi_amount:,.0f}",
                    "Total Interest": f"₹{opt.total_interest:,.0f}",
                    "FOIR": f"{opt.foir:.0%}",
                })
            st.dataframe(emi_data, use_container_width=True, hide_index=True)
        else:
            st.error(f"Not eligible: {loan.ineligible_reason}")

        # Peer benchmarking
        st.subheader("📊 Peer Benchmarking")
        st.markdown(f"**Cohort:** {peer.peer_cohort} ({peer.cohort_size} shops)")

        pc1, pc2, pc3 = st.columns(3)
        pc1.metric("This Shop", f"₹{peer.this_shop_daily:,.0f}/day")
        pc2.metric("Cohort Median", f"₹{peer.cohort_median_daily:,.0f}/day")
        pc3.metric("Percentile", f"{peer.percentile:.0f}th")

        # Visual percentile bar
        pct = min(max(peer.percentile, 2), 98)
        st.markdown(
            f'<div style="background:#e8e8e8;height:24px;border-radius:12px;position:relative;margin:8px 0">'
            f'<div style="position:absolute;left:25%;top:0;bottom:0;width:1px;background:#999"></div>'
            f'<div style="position:absolute;left:50%;top:0;bottom:0;width:1px;background:#999"></div>'
            f'<div style="position:absolute;left:75%;top:0;bottom:0;width:1px;background:#999"></div>'
            f'<div style="position:absolute;left:{pct}%;top:-2px;width:16px;height:28px;'
            f'background:#4A90D9;border-radius:8px;transform:translateX(-50%)"></div>'
            f'</div>'
            f'<div style="display:flex;justify-content:space-between;font-size:11px;color:#666">'
            f'<span>P0</span><span>P25</span><span>P50</span><span>P75</span><span>P100</span></div>',
            unsafe_allow_html=True,
        )
        st.markdown(f"*{peer.interpretation}*")

    # --- RIGHT: Features + QA ---
    with col_right:
        # Structural vs ML
        st.subheader("📐 Estimate Comparison")
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Structural", f"₹{decision.explainability['structural_point_estimate']:,.0f}/day")
        ec2.metric("ML Calibrated", f"₹{decision.explainability['model_point_estimate']:,.0f}/day")
        ec3.metric("Demand/Supply Gap", f"{decision.explainability['prior_disagreement']:.0%}")

        # Image QA
        st.subheader("📷 Image Quality")
        qa_data = [{
            "": "✅" if r.report.quality_pass else "❌",
            "File": r.report.filename,
            "Role": r.report.role,
            "Blur": f"{r.report.blur_score:.0f}",
            "Notes": ", ".join(r.report.notes) or "—",
        } for r in records]
        st.dataframe(qa_data, use_container_width=True, hide_index=True)

        # Vision features
        st.subheader("👁️ Vision")
        vf = decision.explainability["vision_features"]
        vc1, vc2 = st.columns(2)
        vc1.metric("Shop Type", vf["shop_type"])
        vc1.metric("Shelf Density", f"{vf['shelf_density_index']:.2f}")
        vc1.metric("SKU Diversity", f"{vf['sku_diversity']:.2f}")
        vc1.metric("POS Score", f"{vf['pos_sophistication']:.2f}")
        vc2.metric("Inventory", f"₹{vf['inventory_value_est']:,.0f}")
        vc2.metric("Refill Signal", f"{vf['refill_signal']:.2f}")
        vc2.metric("Store Size", f"{vf['store_size_sqft']:.0f} sqft")
        vc2.metric("Coverage", f"{vf['coverage_score']:.0%}")

        # Geo features
        st.subheader("🌍 Geo")
        gf = decision.explainability["geo_features"]
        gc1, gc2 = st.columns(2)
        gc1.metric("Tier", gf["tier"])
        gc1.metric("Road", gf["road_class"])
        gc1.metric("Demand", f"{gf['geo_demand_score']:.2f}")
        gc2.metric("Population", f"{gf['catchment_population_1km']:,}")
        gc2.metric("Competition", f"{gf['competition_density']:.2f}")
        gc2.metric("POI Footfall", f"{gf['poi_footfall_score']:.2f}")

    # === CATEGORY MIX ===
    st.divider()
    st.subheader("📦 Category Mix")
    mix = vf.get("category_mix", {})
    if mix:
        mix_cols = st.columns(len(mix))
        max_val = max(mix.values()) if mix.values() else 1
        for i, (cat, wt) in enumerate(sorted(mix.items(), key=lambda x: -x[1])):
            with mix_cols[i]:
                bar_h = int(wt / max_val * 80) + 10
                color = "#4A90D9" if wt == max_val else "#8FBDE8"
                st.markdown(
                    f"<div style='text-align:center'>"
                    f"<div style='background:{color};height:{bar_h}px;width:36px;"
                    f"margin:0 auto;border-radius:4px 4px 0 0'></div>"
                    f"<div style='font-size:10px;margin-top:4px'>{cat.replace('_',' ')}</div>"
                    f"<div style='font-weight:bold;font-size:12px'>{wt:.0%}</div></div>",
                    unsafe_allow_html=True,
                )

    # === FULL JSON ===
    with st.expander("📄 Full Response JSON"):
        full = {
            "underwriting": decision.model_dump(),
            "loan_sizing": {
                "eligible": loan.eligible,
                "recommended_loan": loan.recommended_loan,
                "max_eligible_loan": loan.max_eligible_loan,
                "emi_options": [
                    {"tenure": o.tenure_months, "emi": o.emi_amount, "foir": o.foir}
                    for o in loan.emi_options
                ],
                "confidence_haircut": loan.confidence_haircut,
                "foir_ceiling": loan.foir_ceiling,
            },
            "peer_benchmark": {
                "cohort": peer.peer_cohort,
                "percentile": peer.percentile,
                "cohort_median": peer.cohort_median_daily,
                "this_shop": peer.this_shop_daily,
            },
        }
        st.code(json.dumps(full, indent=2, default=str), language="json")
