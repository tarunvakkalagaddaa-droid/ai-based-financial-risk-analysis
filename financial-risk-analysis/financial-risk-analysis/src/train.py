"""
train.py
--------
Trains and compares several classifiers for loan-default risk, picks the best by
ROC-AUC, evaluates it, and saves the fitted pipeline to models/risk_model.joblib.

Run:  python src/train.py
"""

import json
import os
import sys

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocessing import (  # noqa: E402
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TARGET,
    build_preprocessor,
    split_xy,
)

DATA_PATH = os.path.join("data", "loan_data.csv")
MODEL_PATH = os.path.join("models", "risk_model.joblib")
REPORT_DIR = "reports"
RANDOM_STATE = 42


def get_candidate_models() -> dict:
    return {
        "logistic_regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=15,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=250, learning_rate=0.06, max_depth=3, random_state=RANDOM_STATE
        ),
    }


def evaluate(name, y_true, y_pred, y_proba) -> dict:
    return {
        "model": name,
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_proba), 4),
    }


def feature_names(pipeline) -> list:
    pre = pipeline.named_steps["preprocessor"]
    ohe = pre.named_transformers_["cat"].named_steps["onehot"]
    return NUMERIC_FEATURES + list(ohe.get_feature_names_out(CATEGORICAL_FEATURES))


def save_plots(y_test, y_proba, importance_df):
    os.makedirs(REPORT_DIR, exist_ok=True)

    fpr, tpr, _ = roc_curve(y_test, y_proba)
    plt.figure(figsize=(5.5, 5))
    plt.plot(fpr, tpr, lw=2, label=f"AUC = {roc_auc_score(y_test, y_proba):.3f}")
    plt.plot([0, 1], [0, 1], "--", color="grey", lw=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve - default prediction")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "roc_curve.png"), dpi=130)
    plt.close()

    top = importance_df.head(12).iloc[::-1]
    plt.figure(figsize=(7, 5))
    plt.barh(top["feature"], top["importance"], color="#2f6f8f")
    plt.xlabel("Importance")
    plt.title("Top drivers of credit risk")
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, "feature_importance.png"), dpi=130)
    plt.close()


def main():
    if not os.path.exists(DATA_PATH):
        raise SystemExit(f"{DATA_PATH} not found. Run: python src/generate_data.py")

    df = pd.read_csv(DATA_PATH)
    X, y = split_xy(df)
    print(f"Dataset: {X.shape[0]:,} rows x {X.shape[1]} features | default rate {y.mean():.2%}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    results, fitted = [], {}
    for name, estimator in get_candidate_models().items():
        pipe = Pipeline([("preprocessor", build_preprocessor()), ("model", estimator)])
        pipe.fit(X_train, y_train)

        y_proba = pipe.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)

        row = evaluate(name, y_test, y_pred, y_proba)
        cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="roc_auc", n_jobs=-1)
        row["cv_roc_auc"] = round(cv.mean(), 4)

        results.append(row)
        fitted[name] = pipe
        print(f"{name:<22} ROC-AUC {row['roc_auc']:.4f} | CV {row['cv_roc_auc']:.4f} | F1 {row['f1']:.4f}")

    leaderboard = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
    best_name = leaderboard.loc[0, "model"]
    best_model = fitted[best_name]
    print(f"\nBest model: {best_name}\n")

    y_proba = best_model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)
    print("Classification report")
    print(classification_report(y_test, y_pred, target_names=["repaid", "default"], zero_division=0))
    print("Confusion matrix (rows = actual)")
    print(confusion_matrix(y_test, y_pred), "\n")

    # ---- Feature importance / coefficients ----
    names = feature_names(best_model)
    estimator = best_model.named_steps["model"]
    if hasattr(estimator, "feature_importances_"):
        scores = estimator.feature_importances_
    else:
        scores = np.abs(estimator.coef_[0])
    importance = (
        pd.DataFrame({"feature": names, "importance": scores})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    print("Top 8 risk drivers")
    print(importance.head(8).to_string(index=False))

    # ---- Persist artifacts ----
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)
    joblib.dump({"pipeline": best_model, "model_name": best_name, "features": names}, MODEL_PATH)

    leaderboard.to_csv(os.path.join(REPORT_DIR, "model_comparison.csv"), index=False)
    importance.to_csv(os.path.join(REPORT_DIR, "feature_importance.csv"), index=False)
    with open(os.path.join(REPORT_DIR, "metrics.json"), "w") as f:
        json.dump({"best_model": best_name, "leaderboard": results}, f, indent=2)
    save_plots(y_test, y_proba, importance)

    print(f"\nSaved model  -> {MODEL_PATH}")
    print(f"Saved reports-> {REPORT_DIR}/")


if __name__ == "__main__":
    main()
