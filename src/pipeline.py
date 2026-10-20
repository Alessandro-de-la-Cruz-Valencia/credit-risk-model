"""Reusable scikit-learn pipelines: cleaning -> imputation -> (optional) log + scaling -> model."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from src.features import CreditDataCleaner

# Right-skewed columns where a log makes the relationship closer to linear for logistic regression.
LOG_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "TotalPastDue",
]


def log1p_skewed(X: pd.DataFrame) -> pd.DataFrame:
    """log(1 + x) on the skewed, non-negative columns; other columns pass through unchanged."""
    X = X.copy()
    X[LOG_COLS] = np.log1p(X[LOG_COLS])
    return X


def build_preprocessor(
    scale: bool = False, log_transform: bool = False, cap_quantile: float = 0.99
) -> Pipeline:
    """Cleaning rules + median imputation, optionally followed by a log and standard scaling.

    The median is used because income and the ratios are very skewed, so the mean would be pulled
    by outliers. Log and scaling only matter for linear models; tree models are unaffected by
    monotonic transformations. Output stays a pandas DataFrame so feature names reach the model
    (needed for SHAP).
    """
    steps = [
        ("clean", CreditDataCleaner(cap_quantile=cap_quantile)),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if log_transform:
        steps.append(("log", FunctionTransformer(log1p_skewed, feature_names_out="one-to-one")))
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")


def build_model_pipeline(
    model: BaseEstimator, scale: bool = False, log_transform: bool = False
) -> Pipeline:
    """Full pipeline: everything is fitted together, so cross-validation never leaks test folds."""
    preprocess = build_preprocessor(scale=scale, log_transform=log_transform)
    return Pipeline([("preprocess", preprocess), ("model", model)])
