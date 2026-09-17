"""
predict.py
----------
Loads the trained pipeline and scores new applicants.

Use as a library:
    from predict import score_applicant
    score_applicant({...})

Or from the command line (scores a built-in sample applicant):
    python src/predict.py
"""

import os
import sys

import joblib
import pandas as pd

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import FEATURES  # noqa: E402

MODEL_PATH = os.path.join("models", "risk_model.joblib")

# Probability thresholds -> risk band, decision
RISK_BANDS = [
    (0.20, "LOW",       "A", "Approve"),
    (0.40, "MODERATE",  "B", "Approve with standard terms"),
    (0.60, "HIGH",      "C", "Refer to manual review"),
    (1.01, "VERY HIGH", "D", "Decline / require collateral"),
]

_cache = {}


def load_model(path: str = MODEL_PATH):
    """Load and cache the trained pipeline bundle."""
    if path not in _cache:
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found. Run: python src/train.py")
        _cache[path] = joblib.load(path)
    return _cache[path]


def risk_band(prob: float):
    for cutoff, band, grade, action in RISK_BANDS:
        if prob < cutoff:
            return band, grade, action
    return RISK_BANDS[-1][1:]


def _rule_flags(a: dict) -> list:
    """Simple, human-readable reasons behind the score (transparency layer)."""
    flags = []
    dti = a.get("debt_to_income") or 0
    if dti > 0.45:
        flags.append(f"Debt-to-income ratio is high ({dti:.2f})")
    cs = a.get("credit_score")
    if cs is not None and cs < 600:
        flags.append(f"Credit score below 600 ({int(cs)})")
    if (a.get("missed_payments_12m") or 0) >= 2:
        flags.append(f"{int(a['missed_payments_12m'])} missed payments in last 12 months")
    if (a.get("existing_loans") or 0) >= 3:
        flags.append(f"{int(a['existing_loans'])} existing loans already open")
    if (a.get("employment_years") or 0) < 1:
        flags.append("Less than 1 year of employment history")
    if a.get("annual_income") and a.get("loan_amount"):
        ratio = a["loan_amount"] / a["annual_income"]
        if ratio > 1.0:
            flags.append(f"Loan is {ratio:.1f}x annual income")
    if not flags:
        flags.append("No major risk flags detected")
    return flags


def score_applicant(applicant: dict, model_path: str = MODEL_PATH) -> dict:
    """Score a single applicant dict and return a full risk assessment."""
    bundle = load_model(model_path)
    row = pd.DataFrame([{f: applicant.get(f) for f in FEATURES}])
    prob = float(bundle["pipeline"].predict_proba(row)[0, 1])
    band, grade, action = risk_band(prob)
    return {
        "probability_of_default": round(prob, 4),
        "risk_score": round((1 - prob) * 1000),      # 0-1000, higher = safer
        "risk_band": band,
        "risk_grade": grade,
        "recommended_action": action,
        "key_factors": _rule_flags(applicant),
        "model_used": bundle["model_name"],
    }


def score_batch(df: pd.DataFrame, model_path: str = MODEL_PATH) -> pd.DataFrame:
    """Score a DataFrame of applicants; returns the frame with risk columns appended."""
    bundle = load_model(model_path)
    probs = bundle["pipeline"].predict_proba(df[FEATURES])[:, 1]
    out = df.copy()
    out["probability_of_default"] = probs.round(4)
    out["risk_score"] = ((1 - probs) * 1000).round().astype(int)
    bands = [risk_band(p) for p in probs]
    out["risk_band"] = [b[0] for b in bands]
    out["risk_grade"] = [b[1] for b in bands]
    out["recommended_action"] = [b[2] for b in bands]
    return out


SAMPLE_APPLICANT = {
    "age": 34,
    "annual_income": 850000,
    "employment_years": 6.0,
    "loan_amount": 900000,
    "loan_term_months": 48,
    "interest_rate": 11.5,
    "credit_score": 640,
    "existing_loans": 2,
    "missed_payments_12m": 1,
    "debt_to_income": 0.26,
    "home_ownership": "RENT",
    "loan_purpose": "debt_consolidation",
}

if __name__ == "__main__":
    result = score_applicant(SAMPLE_APPLICANT)
    print("Applicant risk assessment")
    print("-" * 40)
    for k, v in result.items():
        if isinstance(v, list):
            print(f"{k}:")
            for item in v:
                print(f"   - {item}")
        else:
            print(f"{k}: {v}")
