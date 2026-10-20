"""Stratified cross-validation on the training set (the test set is never used here)."""

import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.model_selection import StratifiedKFold

from src.data import RANDOM_STATE
from src.metrics import evaluate

N_SPLITS = 5


def make_cv(n_splits: int = N_SPLITS) -> StratifiedKFold:
    """Stratified so every fold keeps the ~6.7% default rate; shuffled with a fixed seed."""
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)


def cross_validate_pipeline(
    pipeline: BaseEstimator, X: pd.DataFrame, y: pd.Series, n_splits: int = N_SPLITS
) -> pd.DataFrame:
    """Fit a fresh copy of the full pipeline on each training fold and score the held-out fold.

    Returns one row per fold with every metric from `src.metrics.evaluate`.
    """
    rows = []
    for fold, (train_idx, valid_idx) in enumerate(make_cv(n_splits).split(X, y), start=1):
        model = clone(pipeline).fit(X.iloc[train_idx], y.iloc[train_idx])
        proba = model.predict_proba(X.iloc[valid_idx])[:, 1]
        rows.append({"fold": fold, **evaluate(y.iloc[valid_idx], proba)})
    return pd.DataFrame(rows).set_index("fold")


def summarize(folds: pd.DataFrame) -> pd.Series:
    """Mean and standard deviation across folds, e.g. roc_auc_mean / roc_auc_std."""
    stats = folds.agg(["mean", "std"])
    return pd.Series({f"{m}_{s}": stats.loc[s, m] for m in folds.columns for s in stats.index})
