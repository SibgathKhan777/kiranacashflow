"""Loan sizing and EMI affordability calculator.

Given the estimated monthly income, determines:
  - Maximum eligible loan amount
  - Recommended loan amount (conservative)
  - EMI at various tenures
  - FOIR (Fixed Obligation to Income Ratio) check
  - Debt service coverage ratio

This is what turns an ML demo into a credit product. An NBFC can plug
these numbers directly into their loan origination system.

Assumptions (configurable):
  - NBFC microfinance rates: 18-26% annual
  - FOIR ceiling: 50% of net income (RBI guideline for unsecured MSME)
  - Existing obligations: assumed 0 unless declared
  - Tenure: 6/12/18/24 months (typical kirana working capital loans)
"""
from __future__ import annotations

from dataclasses import dataclass

from api.schemas import CashFlowEstimate, UnderwriteResponse


@dataclass
class EMIOption:
    tenure_months: int
    emi_amount: float
    total_repayment: float
    total_interest: float
    foir: float  # EMI / monthly_income


@dataclass
class LoanSizing:
    # Income basis
    monthly_income_low: float
    monthly_income_high: float
    monthly_income_midpoint: float

    # Loan amounts
    max_eligible_loan: float     # at FOIR ceiling
    recommended_loan: float      # conservative (uses income_low)
    max_loan_tenure: int         # months

    # EMI options at recommended loan
    emi_options: list[EMIOption]

    # Risk adjustments
    confidence_haircut: float    # 0..1, how much we discounted for uncertainty
    risk_adjustment: str         # reason for any adjustment
    foir_ceiling: float          # applied FOIR limit

    # Decision
    eligible: bool
    ineligible_reason: str | None


# Standard NBFC microfinance rates.
ANNUAL_RATE_LOW = 0.18    # 18% — best-case (good profile)
ANNUAL_RATE_HIGH = 0.26   # 26% — high-risk profile
TENURES = [6, 12, 18, 24]

# FOIR = Fixed Obligation to Income Ratio.
# RBI/NBFC guideline: EMI should not exceed 40-50% of net monthly income.
FOIR_CEILING_CLEAN = 0.50      # good confidence, no flags
FOIR_CEILING_RISKY = 0.35      # low confidence or flags present
MIN_MONTHLY_INCOME = 8_000     # below this, most NBFCs won't lend


def _monthly_rate(annual: float) -> float:
    return annual / 12.0


def _emi(principal: float, monthly_rate: float, tenure: int) -> float:
    """Standard reducing-balance EMI formula."""
    if monthly_rate == 0:
        return principal / tenure
    r = monthly_rate
    n = tenure
    return principal * r * (1 + r) ** n / ((1 + r) ** n - 1)


def _max_loan_from_emi(max_emi: float, monthly_rate: float, tenure: int) -> float:
    """Inverse EMI: given max affordable EMI, what's the max principal?"""
    if monthly_rate == 0:
        return max_emi * tenure
    r = monthly_rate
    n = tenure
    return max_emi * ((1 + r) ** n - 1) / (r * (1 + r) ** n)


def size_loan(
    response: UnderwriteResponse,
    estimate: CashFlowEstimate,
    existing_obligations: float = 0.0,
) -> LoanSizing:
    """Size a loan based on the underwriting verdict.

    Uses the *conservative* (low) end of the income range as the basis,
    further haircut by confidence score and risk flags.
    """
    inc_lo = estimate.monthly_income_range[0]
    inc_hi = estimate.monthly_income_range[1]
    inc_mid = (inc_lo + inc_hi) / 2

    # --- Confidence haircut ---
    # Low confidence means we should be more conservative.
    conf = response.confidence_score
    if conf >= 0.7:
        haircut = 0.0
    elif conf >= 0.4:
        haircut = 0.15 + (0.7 - conf) * 0.5  # 15-30% haircut
    else:
        haircut = 0.40  # 40% haircut for very low confidence

    # --- FOIR ceiling ---
    has_high_flags = any(
        f in response.risk_flags
        for f in [
            "inventory_footfall_mismatch",
            "staged_inventory_suspect",
            "stock_photo_suspect",
            "demand_supply_disagreement",
        ]
    )
    foir = FOIR_CEILING_RISKY if has_high_flags or conf < 0.5 else FOIR_CEILING_CLEAN

    # --- Effective income for sizing ---
    effective_income = inc_lo * (1 - haircut)
    max_emi_affordable = max(effective_income * foir - existing_obligations, 0)

    # --- Eligibility check ---
    if response.recommendation == "reject_or_field_visit":
        return LoanSizing(
            monthly_income_low=inc_lo,
            monthly_income_high=inc_hi,
            monthly_income_midpoint=inc_mid,
            max_eligible_loan=0,
            recommended_loan=0,
            max_loan_tenure=0,
            emi_options=[],
            confidence_haircut=haircut,
            risk_adjustment="rejected_by_underwriting",
            foir_ceiling=foir,
            eligible=False,
            ineligible_reason=f"Underwriting recommendation: {response.recommendation}. "
                              f"Flags: {', '.join(response.risk_flags) or 'none'}.",
        )

    if effective_income < MIN_MONTHLY_INCOME:
        return LoanSizing(
            monthly_income_low=inc_lo,
            monthly_income_high=inc_hi,
            monthly_income_midpoint=inc_mid,
            max_eligible_loan=0,
            recommended_loan=0,
            max_loan_tenure=0,
            emi_options=[],
            confidence_haircut=haircut,
            risk_adjustment="income_below_minimum",
            foir_ceiling=foir,
            eligible=False,
            ineligible_reason=f"Effective monthly income ₹{effective_income:,.0f} "
                              f"is below the minimum threshold of ₹{MIN_MONTHLY_INCOME:,}.",
        )

    # --- Rate selection ---
    rate = ANNUAL_RATE_LOW if conf >= 0.7 and not has_high_flags else ANNUAL_RATE_HIGH
    mr = _monthly_rate(rate)

    # --- Maximum eligible loan (at longest tenure) ---
    max_tenure = TENURES[-1]
    max_eligible = _max_loan_from_emi(max_emi_affordable, mr, max_tenure)

    # --- Recommended loan: 70% of max (conservative) ---
    recommended = max_eligible * 0.70

    # --- EMI options ---
    emi_options = []
    for t in TENURES:
        e = _emi(recommended, mr, t)
        total = e * t
        emi_options.append(EMIOption(
            tenure_months=t,
            emi_amount=round(e, 0),
            total_repayment=round(total, 0),
            total_interest=round(total - recommended, 0),
            foir=round(e / max(effective_income, 1), 3),
        ))

    risk_adj = "none"
    if haircut > 0:
        risk_adj = f"confidence_haircut_{haircut:.0%}"
    if has_high_flags:
        risk_adj += "+risk_flags_present"

    return LoanSizing(
        monthly_income_low=round(inc_lo, 0),
        monthly_income_high=round(inc_hi, 0),
        monthly_income_midpoint=round(inc_mid, 0),
        max_eligible_loan=round(max_eligible, 0),
        recommended_loan=round(recommended, 0),
        max_loan_tenure=max_tenure,
        emi_options=emi_options,
        confidence_haircut=round(haircut, 3),
        risk_adjustment=risk_adj,
        foir_ceiling=foir,
        eligible=True,
        ineligible_reason=None,
    )
