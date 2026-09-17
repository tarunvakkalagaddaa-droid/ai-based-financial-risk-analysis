# AI-Based Financial Risk Analysis

A machine-learning system that estimates the credit risk of loan applicants. It predicts
the probability that an applicant will default, converts that into a 0–1000 risk score and
a risk grade (A–D), and recommends an action (approve / review / decline).

## What it does

1. **Data** – generates a realistic synthetic dataset of 8,000 loan applicants (or use your own CSV).
2. **Preprocessing** – imputes missing values, scales numeric features, one-hot encodes categoricals.
3. **Modelling** – trains and compares Logistic Regression, Random Forest and Gradient Boosting, then keeps the best by ROC-AUC.
4. **Scoring** – converts the predicted default probability into a risk band, grade and lending decision, with plain-English reasons.
5. **Dashboard** – a Streamlit app for scoring one applicant interactively or a whole portfolio via CSV upload.

## Project structure

```
financial-risk-analysis/
├── app.py                    # Streamlit dashboard
├── requirements.txt
├── data/
│   └── loan_data.csv         # generated dataset
├── models/
│   └── risk_model.joblib     # trained pipeline
├── reports/                  # metrics, plots, leaderboard
└── src/
    ├── generate_data.py      # synthetic data generator
    ├── preprocessing.py      # feature lists + sklearn pipeline
    ├── train.py              # training, evaluation, model selection
    └── predict.py            # risk scoring for single / batch applicants
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

Run these from the project root, in order:

```bash
python src/generate_data.py    # create data/loan_data.csv
python src/train.py            # train, evaluate, save models/risk_model.joblib
python src/predict.py          # score a sample applicant
streamlit run app.py           # launch the dashboard
```

## Features used

| Type | Features |
|------|----------|
| Numeric | age, annual_income, employment_years, loan_amount, loan_term_months, interest_rate, credit_score, existing_loans, missed_payments_12m, debt_to_income |
| Categorical | home_ownership, loan_purpose |

Target: `default` (1 = defaulted, 0 = repaid)

## Risk bands

| Default probability | Band | Grade | Action |
|---|---|---|---|
| < 20% | LOW | A | Approve |
| 20–40% | MODERATE | B | Approve with standard terms |
| 40–60% | HIGH | C | Refer to manual review |
| > 60% | VERY HIGH | D | Decline / require collateral |

## Sample results

On the generated dataset (20% held-out test set), the selected model reached roughly:

| Model | ROC-AUC | CV ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.769 | 0.756 | 0.619 |
| Gradient Boosting | 0.754 | 0.744 | 0.547 |
| Random Forest | 0.753 | 0.750 | 0.597 |

Top risk drivers: debt-to-income ratio, credit score, loan purpose (debt consolidation),
home ownership and missed payments.

Your numbers will differ slightly if you change the random seed or dataset size.

## Using your own data

Replace `data/loan_data.csv` with a file containing the same columns (including `default`)
and re-run `python src/train.py`. To add or remove features, edit the lists in
`src/preprocessing.py` — training and prediction both read from that single source.

## Notes

The dataset is synthetic and generated for demonstration and coursework purposes. The model
is not tuned or validated for real lending decisions; a production system would need real
historical data, fairness and bias testing, regulatory review, and monitoring for drift.
