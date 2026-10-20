"""SHAP explanations for the LightGBM pipeline.

SHAP values are computed on the tree model's raw output (log-odds of default), on the features
after preprocessing. For one client they add up to: base value + sum(SHAP) = model log-odds, so
each value says how much that feature pushed this client's risk up (+) or down (-).
"""

import numpy as np
import pandas as pd
import shap

# Plain-language names for the app and the plots.
FEATURE_LABELS = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit card / credit line utilization",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "Times 30-59 days late",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines and loans",
    "NumberOfTimes90DaysLate": "Times 90+ days late",
    "NumberRealEstateLoansOrLines": "Real estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "Times 60-89 days late",
    "NumberOfDependents": "Dependents",
    "MonthlyIncome_missing": "Income not reported",
    "NumberOfDependents_missing": "Dependents not reported",
    "LatePayment_special_code": "Special late-payment code (96/98)",
    "TotalPastDue": "Total late payments",
}


def split_pipeline(pipeline):
    """Return (preprocessor, tree model) from a fitted pipeline built by `build_model_pipeline`."""
    return pipeline.named_steps["preprocess"], pipeline.named_steps["model"]


def make_explainer(pipeline) -> shap.TreeExplainer:
    _, model = split_pipeline(pipeline)
    return shap.TreeExplainer(model)


def shap_values(pipeline, X: pd.DataFrame, explainer: shap.TreeExplainer | None = None):
    """SHAP explanation object for raw rows `X` (preprocessing is applied here)."""
    preprocess, _ = split_pipeline(pipeline)
    explainer = explainer or make_explainer(pipeline)
    X_proc = preprocess.transform(X)
    explanation = explainer(X_proc)
    # Binary LightGBM returns one output; keep the class-1 slice if a 3-D array comes back.
    if explanation.values.ndim == 3:
        explanation = explanation[:, :, 1]
    return explanation


def top_reasons(explanation_row, n: int = 5) -> pd.DataFrame:
    """The `n` features with the largest absolute SHAP value for one client."""
    df = pd.DataFrame(
        {
            "feature": explanation_row.feature_names,
            "value": np.asarray(explanation_row.data, dtype=float),
            "shap": np.asarray(explanation_row.values, dtype=float),
        }
    )
    df["label"] = df["feature"].map(FEATURE_LABELS).fillna(df["feature"])
    df["effect"] = np.where(df["shap"] > 0, "raises risk", "lowers risk")
    return df.reindex(df["shap"].abs().sort_values(ascending=False).index).head(n)
