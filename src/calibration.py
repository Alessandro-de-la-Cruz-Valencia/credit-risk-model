"""Probability calibration fitted on out-of-fold predictions.

Why out-of-fold: predictions a model makes on its own training rows are over-confident. Cross-
validated predictions behave like predictions on new clients, so a calibrator learned on them
transfers to the final model trained on all of the training data. The base model is kept intact,
so SHAP can still explain it directly.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from src.validation import make_cv

EPS = 1e-6


def out_of_fold_proba(pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """Probability of default for every training row, predicted by a model that did not see it."""
    return cross_val_predict(pipeline, X, y, cv=make_cv(), method="predict_proba", n_jobs=-1)[:, 1]


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, EPS, 1 - EPS)
    return np.log(p / (1 - p))


class ProbabilityCalibrator(BaseEstimator):
    """Map raw scores to calibrated probabilities.

    - "sigmoid" (Platt scaling): logistic regression on the logit of the score. Monotonic and
      smooth, so the ranking (AUC, Gini, KS) is unchanged.
    - "isotonic": a free-form non-decreasing step function. More flexible, but ties can lower AUC.
    """

    def __init__(self, method: str = "sigmoid"):
        self.method = method

    def fit(self, scores, y):
        scores = np.asarray(scores, dtype=float)
        if self.method == "sigmoid":
            self.model_ = LogisticRegression(C=1e6).fit(_logit(scores).reshape(-1, 1), y)
        elif self.method == "isotonic":
            self.model_ = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(scores, y)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        return self

    def transform(self, scores) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)
        if self.method == "sigmoid":
            return self.model_.predict_proba(_logit(scores).reshape(-1, 1))[:, 1]
        return self.model_.predict(scores)


class CalibratedModel(ClassifierMixin, BaseEstimator):
    """A fitted pipeline plus an optional fitted calibrator, exposing `predict_proba`."""

    def __init__(self, base, calibrator: ProbabilityCalibrator | None = None):
        self.base = base
        self.calibrator = calibrator

    @property
    def classes_(self):
        return self.base.classes_

    def predict_proba(self, X) -> np.ndarray:
        p = self.base.predict_proba(X)[:, 1]
        if self.calibrator is not None:
            p = self.calibrator.transform(p)
        return np.column_stack([1 - p, p])

    def predict(self, X, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


def expected_calibration_error(y_true, proba, n_bins: int = 10) -> float:
    """Average gap between predicted and observed default rate, over quantile bins of the score,
    weighted by bin size."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(proba)})
    df["bin"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    grouped = df.groupby("bin", observed=True).agg(
        p=("p", "mean"), y=("y", "mean"), n=("y", "size")
    )
    return float((grouped["n"] * (grouped["p"] - grouped["y"]).abs()).sum() / grouped["n"].sum())
