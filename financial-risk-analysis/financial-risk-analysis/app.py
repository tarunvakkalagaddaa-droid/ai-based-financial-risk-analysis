"""
app.py
------
Streamlit dashboard for the AI-Based Financial Risk Analysis project.

Run:  streamlit run app.py
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from predict import score_applicant, score_batch  # noqa: E402
from preprocessing import FEATURES  # noqa: E402

st.set_page_config(page_title="AI Financial Risk Analysis", page_icon="📊", layout="wide")

BAND_COLOR = {"LOW": "#1b7f4b", "MODERATE": "#b8860b", "HIGH": "#c25a1a", "VERY HIGH": "#b12727"}

st.title("AI-Based Financial Risk Analysis")
st.caption("Machine-learning credit risk scoring for loan applicants")

tab_single, tab_batch, tab_model = st.tabs(["Single applicant", "Batch scoring", "Model performance"])

# ------------------------------------------------------------------ single
with tab_single:
    st.subheader("Applicant details")
    c1, c2, c3 = st.columns(3)

    with c1:
        age = st.number_input("Age", 21, 70, 34)
        annual_income = st.number_input("Annual income (₹)", 100_000, 10_000_000, 850_000, step=25_000)
        employment_years = st.number_input("Years employed", 0.0, 40.0, 6.0, step=0.5)
        home_ownership = st.selectbox("Home ownership", ["RENT", "OWN", "MORTGAGE"])

    with c2:
        loan_amount = st.number_input("Loan amount (₹)", 50_000, 5_000_000, 900_000, step=25_000)
        loan_term_months = st.selectbox("Loan term (months)", [12, 24, 36, 48, 60, 84], index=3)
        interest_rate = st.slider("Interest rate (%)", 5.0, 24.0, 11.5, 0.25)
        loan_purpose = st.selectbox(
            "Loan purpose",
            ["personal", "education", "medical", "business", "home_improvement", "debt_consolidation"],
        )

    with c3:
        credit_score = st.slider("Credit score", 300, 900, 640)
        existing_loans = st.number_input("Existing loans", 0, 10, 2)
        missed_payments_12m = st.number_input("Missed payments (12m)", 0, 12, 1)

    monthly_emi = loan_amount / loan_term_months
    debt_to_income = round(monthly_emi / (annual_income / 12), 3)
    st.info(f"Estimated monthly EMI: ₹{monthly_emi:,.0f}  •  Debt-to-income ratio: {debt_to_income}")

    if st.button("Assess risk", type="primary"):
        applicant = {
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
        }
        try:
            result = score_applicant(applicant)
        except FileNotFoundError as e:
            st.error(str(e))
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Probability of default", f"{result['probability_of_default']*100:.1f}%")
            m2.metric("Risk score (0-1000)", result["risk_score"])
            m3.metric("Risk grade", result["risk_grade"])

            color = BAND_COLOR.get(result["risk_band"], "#444")
            st.markdown(
                f"<div style='padding:14px;border-radius:8px;background:{color};color:#fff;"
                f"font-size:17px;'><b>{result['risk_band']} RISK</b> — {result['recommended_action']}</div>",
                unsafe_allow_html=True,
            )
            st.progress(min(result["probability_of_default"], 1.0))

            st.markdown("**Key factors considered**")
            for factor in result["key_factors"]:
                st.write("•", factor)
            st.caption(f"Model: {result['model_used']}")

# ------------------------------------------------------------------ batch
with tab_batch:
    st.subheader("Score a portfolio")
    st.write("Upload a CSV containing these columns:")
    st.code(", ".join(FEATURES))

    uploaded = st.file_uploader("CSV file", type="csv")
    if uploaded is not None:
        df = pd.read_csv(uploaded)
        missing = [c for c in FEATURES if c not in df.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            scored = score_batch(df)
            st.success(f"Scored {len(scored):,} applicants")
            k1, k2, k3 = st.columns(3)
            k1.metric("Average default probability", f"{scored['probability_of_default'].mean()*100:.1f}%")
            k2.metric("High / very-high risk", int(scored["risk_band"].isin(["HIGH", "VERY HIGH"]).sum()))
            k3.metric("Portfolio exposure", f"₹{scored['loan_amount'].sum():,.0f}"
                      if "loan_amount" in scored else "n/a")
            st.bar_chart(scored["risk_band"].value_counts())
            st.dataframe(scored.head(200), use_container_width=True)
            st.download_button(
                "Download scored CSV",
                scored.to_csv(index=False).encode(),
                "scored_applicants.csv",
                "text/csv",
            )

# ------------------------------------------------------------------ model
with tab_model:
    st.subheader("Model comparison")
    comp_path = os.path.join("reports", "model_comparison.csv")
    imp_path = os.path.join("reports", "feature_importance.csv")

    if os.path.exists(comp_path):
        st.dataframe(pd.read_csv(comp_path), use_container_width=True)
    else:
        st.warning("Run `python src/train.py` first to generate reports.")

    c1, c2 = st.columns(2)
    if os.path.exists(os.path.join("reports", "roc_curve.png")):
        c1.image(os.path.join("reports", "roc_curve.png"), caption="ROC curve")
    if os.path.exists(os.path.join("reports", "feature_importance.png")):
        c2.image(os.path.join("reports", "feature_importance.png"), caption="Feature importance")

    if os.path.exists(imp_path):
        st.markdown("**Risk drivers**")
        st.dataframe(pd.read_csv(imp_path).head(15), use_container_width=True)
