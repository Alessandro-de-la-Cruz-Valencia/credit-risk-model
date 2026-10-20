"""Streamlit demo: enter a client's data, get the probability of default, the decision and the
main SHAP reasons. Run with `streamlit run app/app.py` after `python -m src.train`."""

import json
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.explain import make_explainer, shap_values, top_reasons  # noqa: E402
from src.features import FEATURES  # noqa: E402

MODEL_PATH = ROOT / "models" / "credit_model.joblib"
METADATA_PATH = ROOT / "models" / "metadata.json"

st.set_page_config(page_title="Credit Risk Scoring", page_icon="💳", layout="wide")


@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text())
    return model, metadata, make_explainer(model.base)


if not MODEL_PATH.exists():
    st.error("No trained model found. Run `python -m src.train` from the project root first.")
    st.stop()

model, metadata, explainer = load_artifacts()
medians = metadata["feature_medians"]
threshold = metadata["threshold"]

st.title("Credit Risk Scoring")
st.caption(
    "Probability of serious delinquency (90+ days past due) in the next 2 years. "
    "Portfolio project trained on the Kaggle *Give Me Some Credit* data: not for real decisions."
)

with st.sidebar:
    st.header("Model")
    m = metadata["test_metrics"]
    st.metric("Test ROC-AUC", f"{m['roc_auc']:.4f}")
    st.metric("Test Gini", f"{m['gini']:.4f}")
    st.metric("Test KS", f"{m['ks']:.4f}")
    st.write(f"Approval cut-off: **PD < {threshold:.2%}** (KS point on out-of-fold predictions)")
    op = metadata["operating_point_test"]
    st.write(
        f"On the test set this approves {op['approval_rate']:.1%} of clients with a "
        f"{op['bad_rate_approved']:.2%} default rate among the approved."
    )

with st.form("client"):
    st.subheader("Client data")
    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", 18, 110, int(medians["age"]))
        income_missing = st.checkbox("Income not reported")
        income = st.number_input(
            "Monthly income (USD)",
            0,
            1_000_000,
            int(medians["MonthlyIncome"]),
            step=100,
            disabled=income_missing,
        )
        dependents_missing = st.checkbox("Dependents not reported")
        dependents = st.number_input(
            "Dependents", 0, 20, int(medians["NumberOfDependents"]), disabled=dependents_missing
        )
    with c2:
        utilization = st.number_input(
            "Credit card / line utilization (balance / limit)",
            0.0,
            100.0,
            round(medians["RevolvingUtilizationOfUnsecuredLines"], 3),
            step=0.05,
            format="%.3f",
        )
        debt_ratio = st.number_input(
            "Debt ratio (monthly debt payments / income)",
            0.0,
            100_000.0,
            round(medians["DebtRatio"], 3),
            step=0.05,
            format="%.3f",
        )
        open_lines = st.number_input(
            "Open credit lines and loans", 0, 100, int(medians["NumberOfOpenCreditLinesAndLoans"])
        )
        real_estate = st.number_input(
            "Real estate loans", 0, 60, int(medians["NumberRealEstateLoansOrLines"])
        )
    with c3:
        late_30 = st.number_input("Times 30-59 days late (last 2 years)", 0, 20, 0)
        late_60 = st.number_input("Times 60-89 days late (last 2 years)", 0, 20, 0)
        late_90 = st.number_input("Times 90+ days late", 0, 20, 0)
    submitted = st.form_submit_button("Score client", type="primary")

client = pd.DataFrame(
    [
        {
            "RevolvingUtilizationOfUnsecuredLines": utilization,
            "age": age,
            "NumberOfTime30-59DaysPastDueNotWorse": late_30,
            "DebtRatio": debt_ratio,
            "MonthlyIncome": np.nan if income_missing else income,
            "NumberOfOpenCreditLinesAndLoans": open_lines,
            "NumberOfTimes90DaysLate": late_90,
            "NumberRealEstateLoansOrLines": real_estate,
            "NumberOfTime60-89DaysPastDueNotWorse": late_60,
            "NumberOfDependents": np.nan if dependents_missing else dependents,
        }
    ]
)[FEATURES]

pd_value = float(model.predict_proba(client)[0, 1])
approved = pd_value < threshold

st.subheader("Result")
r1, r2, r3 = st.columns(3)
r1.metric("Probability of default", f"{pd_value:.2%}")
r2.metric("Cut-off", f"{threshold:.2%}")
r3.metric(
    "Decision",
    "APPROVE" if approved else "REJECT",
    f"{(pd_value - threshold) * 100:+.2f} pp vs cut-off",
    delta_color="inverse",
)
if approved:
    st.success("Risk below the cut-off: the application would be approved.")
else:
    st.error("Risk above the cut-off: the application would be rejected or sent to review.")

st.subheader("Main reasons (SHAP)")
reasons = top_reasons(shap_values(model.base, client, explainer)[0], n=5)
left, right = st.columns([1, 1])
with left:
    fig, ax = plt.subplots(figsize=(7, 3.2))
    plot_df = reasons.iloc[::-1]
    colors = ["#C44E52" if s > 0 else "#4C72B0" for s in plot_df["shap"]]
    ax.barh(plot_df["label"], plot_df["shap"], color=colors)
    ax.axvline(0, color="grey", lw=1)
    ax.set_xlabel("Impact on log-odds of default (red raises risk, blue lowers it)")
    fig.tight_layout()
    st.pyplot(fig)
with right:
    table = reasons[["label", "value", "effect"]].rename(
        columns={"label": "Feature", "value": "Value", "effect": "Effect"}
    )
    st.dataframe(table, hide_index=True, use_container_width=True)
st.caption(
    "Values are shown after the model's cleaning step (capping and imputation), which is what "
    "the model actually sees."
)
