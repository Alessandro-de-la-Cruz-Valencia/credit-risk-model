"""End-to-end training: `python -m src.train`.

Stages are added as the project grows. Every result written to reports/ comes from this script,
so the numbers in the README can be reproduced from scratch.
"""

import pandas as pd

from src.data import PROJECT_ROOT, TARGET, load_raw, split_train_test
from src.models import make_logistic
from src.validation import cross_validate_pipeline, summarize

METRICS_DIR = PROJECT_ROOT / "reports" / "metrics"


def run_baseline(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Logistic regression with and without the log transform, 5-fold stratified CV on train."""
    variants = {
        "logistic_scaled": make_logistic(log_transform=False),
        "logistic_log_scaled": make_logistic(log_transform=True),
    }
    folds = {name: cross_validate_pipeline(pipe, X, y) for name, pipe in variants.items()}
    summary = pd.DataFrame({name: summarize(f) for name, f in folds.items()}).T
    pd.concat(folds, names=["model"]).to_csv(METRICS_DIR / "cv_baseline_folds.csv")
    summary.to_csv(METRICS_DIR / "cv_baseline_summary.csv")

    # Coefficients of the chosen variant refit on the whole training set (per 1 std change).
    chosen = variants["logistic_log_scaled"].fit(X, y)
    coefs = pd.Series(
        chosen.named_steps["model"].coef_[0],
        index=chosen.named_steps["preprocess"].get_feature_names_out(),
        name="coefficient",
    ).sort_values(key=abs, ascending=False)
    coefs.to_csv(METRICS_DIR / "logistic_coefficients.csv")
    return summary


def main() -> None:
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    train, _test = split_train_test(load_raw())
    X, y = train.drop(columns=TARGET), train[TARGET]

    print("== Baseline: logistic regression (5-fold stratified CV on train) ==")
    print(run_baseline(X, y).round(4).to_string())


if __name__ == "__main__":
    main()
