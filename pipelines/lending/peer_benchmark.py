"""Peer benchmarking: compare this shop against similar shops.

Generates a synthetic "peer cohort" from the same tier and shop type using
the structural priors. This tells the credit officer:

  "This shop's estimated daily sales of ₹7,500 places it at the 62nd
   percentile among tier-2 packaged kirana stores."

In production, this would query a database of previously underwritten shops.
For the demo, we generate the peer distribution from the same priors + noise
that the synthetic training data uses — self-consistent and defensible.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from api.schemas import CashFlowEstimate, GeoFeatures, VisionFeatures


LOOKUPS = Path(__file__).resolve().parents[2] / "data" / "lookups"
TIER_PRIORS = json.loads((LOOKUPS / "tier_priors.json").read_text())
CATEGORY_MARGINS = json.loads((LOOKUPS / "category_margins.json").read_text())


@dataclass
class PeerBenchmark:
    peer_cohort: str           # e.g. "tier2 packaged_kirana"
    cohort_size: int           # simulated
    percentile: float          # 0..100
    cohort_median_daily: float
    cohort_p25_daily: float
    cohort_p75_daily: float
    cohort_min_daily: float
    cohort_max_daily: float
    this_shop_daily: float
    interpretation: str


def _simulate_cohort(
    tier: str,
    shop_type: str,
    n: int = 500,
    seed: int = 123,
) -> np.ndarray:
    """Generate n synthetic daily-sales values for the cohort."""
    rng = np.random.default_rng(seed)
    tp = TIER_PRIORS[tier]

    base_footfall = tp["daily_footfall_base"]
    ticket_median = tp["ticket_size_median"]

    # Vary footfall, conversion, and ticket across the cohort.
    footfalls = base_footfall * rng.lognormal(0, 0.35, size=n)
    conversions = rng.beta(4, 6, size=n) * 0.5 + 0.2   # 0.2 .. 0.7
    tickets = ticket_median * rng.lognormal(0, 0.25, size=n)

    # Shop-type adjustment.
    if shop_type == "street_stall":
        footfalls *= 0.5
        tickets *= 0.7
    elif shop_type == "loose_goods":
        tickets *= 1.3  # higher ₹ per transaction for dry fruits etc.

    daily_sales = footfalls * conversions * tickets
    return daily_sales


def benchmark(
    vision: VisionFeatures,
    geo: GeoFeatures,
    estimate: CashFlowEstimate,
) -> PeerBenchmark:
    """Compare this shop against a simulated peer cohort."""
    cohort_label = f"{geo.tier} {vision.shop_type}"
    cohort = _simulate_cohort(geo.tier, vision.shop_type)

    this_daily = estimate.structural_point_estimate

    percentile = float(np.searchsorted(np.sort(cohort), this_daily) / len(cohort) * 100)
    percentile = round(min(max(percentile, 0), 100), 1)

    median = float(np.median(cohort))
    p25 = float(np.percentile(cohort, 25))
    p75 = float(np.percentile(cohort, 75))

    # Interpretation.
    if percentile >= 75:
        interp = (
            f"This shop is in the top quartile of {cohort_label} stores. "
            "Strong performer — higher loan sizes may be justified."
        )
    elif percentile >= 50:
        interp = (
            f"This shop is above the median for {cohort_label} stores. "
            "Solid performance — standard loan terms appropriate."
        )
    elif percentile >= 25:
        interp = (
            f"This shop is below the median for {cohort_label} stores. "
            "Conservative sizing recommended — verify with field visit."
        )
    else:
        interp = (
            f"This shop is in the bottom quartile of {cohort_label} stores. "
            "Elevated risk — consider smaller loan or shorter tenure."
        )

    return PeerBenchmark(
        peer_cohort=cohort_label,
        cohort_size=len(cohort),
        percentile=percentile,
        cohort_median_daily=round(median, 0),
        cohort_p25_daily=round(p25, 0),
        cohort_p75_daily=round(p75, 0),
        cohort_min_daily=round(float(cohort.min()), 0),
        cohort_max_daily=round(float(cohort.max()), 0),
        this_shop_daily=round(this_daily, 0),
        interpretation=interp,
    )
