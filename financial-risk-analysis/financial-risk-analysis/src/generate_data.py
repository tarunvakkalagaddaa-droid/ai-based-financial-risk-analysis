"""
generate_data.py
----------------
Creates a synthetic loan-applicant dataset for the Financial Risk Analysis project.

Each row = one loan applicant.
Target column = `default` (1 = applicant defaulted, 0 = repaid).

Run:  python src/generate_data.py
Out:  data/loan_data.csv
"""

import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_SAMPLES = 8000

HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE"]
LOAN_PURPOSE = ["personal", "education", "medical", "business", "home_improvement", "debt_consolidation"]


def generate(n=N_SAMPLES, seed=RANDOM_STATE) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(21, 66, n)
    employment_years = np.clip(rng.gamma(shape=2.0, scale=3.0, size=n), 0, 40).round(1)

    # Income grows mildly with age and experience
    annual_income = (
        250000
        + age * 6000
        + employment_years * 12000
        + rng.normal(0, 150000, n)
    ).clip(120000, 4_000_000).round(-3)

    loan_amount = (annual_income * rng.uniform(0.15, 1.6, n)).clip(50000, 5_000_000).round(-3)
    loan_term_months = rng.choice([12, 24, 36, 48, 60, 84], n, p=[.10, .18, .30, .18, .18, .06])
    interest_rate = (7.5 + rng.normal(0, 2.2, n)).clip(5.0, 24.0).round(2)

    existing_loans = rng.poisson(1.1, n).clip(0, 8)
    missed_payments_12m = rng.poisson(0.5, n).clip(0, 12)

    # Credit score penalised by missed payments and number of open loans
    credit_score = (
        720
        - missed_payments_12m * 28
        - existing_loans * 9
        + employment_years * 2
        + rng.normal(0, 55, n)
    ).clip(300, 900).round().astype(int)

    monthly_income = annual_income / 12
    monthly_emi = loan_amount / loan_term_months
    debt_to_income = (monthly_emi / monthly_income).clip(0.01, 3.0).round(3)

    home_ownership = rng.choice(HOME_OWNERSHIP, n, p=[.45, .25, .30])
    loan_purpose = rng.choice(LOAN_PURPOSE, n, p=[.28, .12, .10, .15, .15, .20])

    # ---- Latent risk score -> probability of default (logistic link) ----
    risk = (
        -1.35
        + 3.10 * debt_to_income
        - 0.0085 * (credit_score - 650)
        + 0.34 * missed_payments_12m
        + 0.16 * existing_loans
        - 0.045 * employment_years
        - 0.020 * (age - 40)
        + 0.30 * (interest_rate - 10) / 5
        + np.where(home_ownership == "RENT", 0.30, np.where(home_ownership == "OWN", -0.28, 0.0))
        + np.where(loan_purpose == "debt_consolidation", 0.38,
          np.where(loan_purpose == "business", 0.22,
          np.where(loan_purpose == "education", -0.18, 0.0)))
        + rng.normal(0, 0.55, n)          # irreducible noise
    )
    prob_default = 1 / (1 + np.exp(-risk))
    default = rng.binomial(1, prob_default)

    df = pd.DataFrame({
        "applicant_id": [f"APP{100000 + i}" for i in range(n)],
        "age": age,
        "annual_income": annual_income,
        "employment_years": employment_years,
        "loan_amount": loan_amount,
        "loan_term_months": loan_term_months,
        "interest_rate": interest_rate,
        "credit_score": credit_score,
        "existing_loans": existing_loans,
        "missed_payments_12m": missed_payments_12m,
        "debt_to_income": debt_to_income,
        "home_ownership": home_ownership,
        "loan_purpose": loan_purpose,
        "default": default,
    })

    # Inject a little real-world messiness (missing values)
    for col, frac in [("credit_score", 0.02), ("employment_years", 0.015)]:
        idx = rng.choice(n, int(n * frac), replace=False)
        df.loc[idx, col] = np.nan

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    data = generate()
    path = os.path.join("data", "loan_data.csv")
    data.to_csv(path, index=False)
    print(f"Saved {len(data):,} rows -> {path}")
    print(f"Default rate: {data['default'].mean():.2%}")
