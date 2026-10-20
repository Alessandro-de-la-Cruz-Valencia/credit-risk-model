"""Reusable scikit-learn pipelines: cleaning -> imputation -> (optional) scaling -> model."""

from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.features import CreditDataCleaner


def build_preprocessor(scale: bool = False, cap_quantile: float = 0.99) -> Pipeline:
    """Cleaning rules + median imputation, optionally followed by standard scaling.

    The median is used because income and the ratios are very skewed, so the mean would be pulled
    by outliers. Scaling only matters for linear models; tree models do not need it.
    Output stays a pandas DataFrame so feature names reach the model (needed for SHAP).
    """
    steps = [
        ("clean", CreditDataCleaner(cap_quantile=cap_quantile)),
        ("impute", SimpleImputer(strategy="median")),
    ]
    if scale:
        steps.append(("scale", StandardScaler()))
    return Pipeline(steps).set_output(transform="pandas")


def build_model_pipeline(model: BaseEstimator, scale: bool = False) -> Pipeline:
    """Full pipeline: everything is fitted together, so cross-validation never leaks test folds."""
    return Pipeline([("preprocess", build_preprocessor(scale=scale)), ("model", model)])
