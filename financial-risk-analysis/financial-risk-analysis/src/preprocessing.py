"""
preprocessing.py
----------------
Column definitions and the sklearn preprocessing pipeline.
Kept in one place so training and prediction always use identical transformations.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "default"
ID_COL = "applicant_id"

NUMERIC_FEATURES = [
    "age",
    "annual_income",
    "employment_years",
    "loan_amount",
    "loan_term_months",
    "interest_rate",
    "credit_score",
    "existing_loans",
    "missed_payments_12m",
    "debt_to_income",
]

CATEGORICAL_FEATURES = [
    "home_ownership",
    "loan_purpose",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def build_preprocessor() -> ColumnTransformer:
    """Median-impute + scale numerics; mode-impute + one-hot encode categoricals."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, NUMERIC_FEATURES),
        ("cat", categorical_pipe, CATEGORICAL_FEATURES),
    ])


def split_xy(df):
    """Return (X, y) with only the modelling columns."""
    X = df[FEATURES].copy()
    y = df[TARGET].copy() if TARGET in df.columns else None
    return X, y
