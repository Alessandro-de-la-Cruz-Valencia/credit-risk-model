"""End-to-end training: `python -m src.train`.

Order matters and mirrors how the model would be built for real:
1. Split once; everything up to step 6 only uses the training set (cross-validation).
2. Baseline: logistic regression.
3. Main model: LightGBM with a randomized hyperparameter search.
4. Class imbalance: class_weight="balanced" vs nothing, for both models.
5. Calibration check on out-of-fold predictions; choose the operating cut-off on them too.
6. Fit the final model on all training data and save it.
7. Only now: evaluate once on the untouched test set and write every number in reports/.
8. SHAP explanations.
"""

import json
import platform
import time

import joblib
import lightgbm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sklearn
from sklearn.model_selection import RandomizedSearchCV

from src.calibration import (
    CalibratedModel,
    ProbabilityCalibrator,
    expected_calibration_error,
    out_of_fold_proba,
)
from src.data import PROJECT_ROOT, RANDOM_STATE, TARGET, load_raw, split_train_test
from src.evaluation import (
    cutoff_table,
    ks_threshold,
    plot_calibration,
    plot_roc_pr,
    threshold_report,
)
from src.explain import FEATURE_LABELS, make_explainer, shap_values
from src.features import FEATURES
from src.metrics import evaluate
from src.models import LGBM_PARAM_DISTRIBUTIONS, make_lightgbm, make_logistic
from src.validation import cross_validate_pipeline, make_cv, summarize

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_DIR = REPORTS_DIR / "metrics"
FIG_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "credit_model.joblib"
METADATA_PATH = MODELS_DIR / "metadata.json"

N_SEARCH_ITER = 30
SHAP_SAMPLE = 3000


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def run_lightgbm_search(X: pd.DataFrame, y: pd.Series) -> dict:
    """Randomized search (N_SEARCH_ITER candidates x 5 folds), optimizing ROC-AUC."""
    search = RandomizedSearchCV(
        make_lightgbm(),
        LGBM_PARAM_DISTRIBUTIONS,
        n_iter=N_SEARCH_ITER,
        scoring="roc_auc",
        cv=make_cv(),
        n_jobs=-1,
        random_state=RANDOM_STATE,
        refit=False,
    )
    search.fit(X, y)
    results = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
    cols = ["rank_test_score", "mean_test_score", "std_test_score"] + [
        c for c in results.columns if c.startswith("param_")
    ]
    results[cols].to_csv(METRICS_DIR / "lgbm_search_results.csv", index=False)
    best = {k.removeprefix("model__"): _to_builtin(v) for k, v in search.best_params_.items()}
    with open(METRICS_DIR / "lgbm_best_params.json", "w") as f:
        json.dump({"params": best, "cv_roc_auc": float(search.best_score_)}, f, indent=2)
    return best


def run_imbalance_comparison(X, y, lgbm_params: dict) -> pd.DataFrame:
    """Same models, with and without class_weight='balanced'. Ranking metrics barely depend on
    the weights, but the probabilities do (Brier), which matters for a PD model."""
    variants = {
        "logistic_none": make_logistic(class_weight=None),
        "logistic_balanced": make_logistic(class_weight="balanced"),
        "lightgbm_none": make_lightgbm(class_weight=None, **lgbm_params),
        "lightgbm_balanced": make_lightgbm(class_weight="balanced", **lgbm_params),
    }
    folds = {name: cross_validate_pipeline(pipe, X, y) for name, pipe in variants.items()}
    summary = pd.DataFrame({name: summarize(f) for name, f in folds.items()}).T
    pd.concat(folds, names=["model"]).to_csv(METRICS_DIR / "cv_imbalance_folds.csv")
    summary.to_csv(METRICS_DIR / "cv_imbalance_summary.csv")
    return summary


def run_calibration_check(oof: np.ndarray, y: pd.Series) -> tuple[str | None, pd.DataFrame]:
    """Compare raw, Platt and isotonic probabilities on out-of-fold predictions.

    The calibrators are themselves cross-fitted (learned on 4/5 of the OOF scores and applied to
    the remaining 1/5) so the comparison is not optimistic.
    """
    y_arr = y.to_numpy()
    candidates = {"raw": oof}
    for method in ("sigmoid", "isotonic"):
        cal = np.zeros_like(oof)
        for tr_idx, va_idx in make_cv().split(oof.reshape(-1, 1), y_arr):
            calibrator = ProbabilityCalibrator(method).fit(oof[tr_idx], y_arr[tr_idx])
            cal[va_idx] = calibrator.transform(oof[va_idx])
        candidates[method] = cal

    rows = []
    for name, p in candidates.items():
        m = evaluate(y_arr, p)
        rows.append(
            {
                "method": name,
                "brier": m["brier"],
                "ece": expected_calibration_error(y_arr, p),
                "roc_auc": m["roc_auc"],
                "mean_pd": float(p.mean()),
            }
        )
    table = pd.DataFrame(rows).set_index("method")
    table.loc["observed_default_rate"] = [np.nan, np.nan, np.nan, float(y_arr.mean())]
    table.to_csv(METRICS_DIR / "calibration_oof.csv")
    plot_calibration(
        {name: (y_arr, p) for name, p in candidates.items()},
        FIG_DIR / "calibration_oof.png",
        "Calibration on out-of-fold predictions (train)",
    )
    # Calibrate only if it clearly helps: at least 1% lower Brier than the raw probabilities.
    best = table.drop("observed_default_rate")["brier"].idxmin()
    if best == "raw" or table.loc[best, "brier"] > 0.99 * table.loc["raw", "brier"]:
        return None, table
    return best, table


def run_shap(pipeline, X_test: pd.DataFrame, proba_test: np.ndarray, threshold: float) -> None:
    """Global importance (beeswarm + bar) and three individual explanations."""
    sample = X_test.sample(min(SHAP_SAMPLE, len(X_test)), random_state=RANDOM_STATE)
    explainer = make_explainer(pipeline)
    explanation = shap_values(pipeline, sample, explainer)
    explanation.feature_names = [FEATURE_LABELS.get(f, f) for f in explanation.feature_names]

    importance = pd.Series(
        np.abs(explanation.values).mean(axis=0), index=explanation.feature_names
    ).sort_values(ascending=False)
    importance.rename("mean_abs_shap").to_csv(METRICS_DIR / "shap_importance.csv")

    shap.plots.beeswarm(explanation, max_display=14, show=False)
    plt.title("SHAP summary (test sample, log-odds of default)")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_beeswarm.png", dpi=120)
    plt.close("all")

    shap.plots.bar(explanation, max_display=14, show=False)
    plt.title("Global importance: mean |SHAP|")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "shap_bar.png", dpi=120)
    plt.close("all")

    # Individual clients: clearly low risk, close to the cut-off, clearly high risk.
    proba = pd.Series(proba_test, index=X_test.index)
    picks = {
        "low_risk": proba.idxmin(),
        "near_cutoff": (proba - threshold).abs().idxmin(),
        "high_risk": proba.idxmax(),
    }
    individual = {}
    for name, idx in picks.items():
        row = shap_values(pipeline, X_test.loc[[idx]], explainer)[0]
        row.feature_names = [FEATURE_LABELS.get(f, f) for f in row.feature_names]
        shap.plots.waterfall(row, max_display=10, show=False)
        plt.title(f"Client {idx} ({name.replace('_', ' ')}): PD = {proba[idx]:.1%}")
        plt.tight_layout()
        plt.savefig(FIG_DIR / f"shap_client_{name}.png", dpi=120)
        plt.close("all")
        individual[name] = {"client_id": int(idx), "pd": float(proba[idx])}
    with open(METRICS_DIR / "shap_individual_clients.json", "w") as f:
        json.dump(individual, f, indent=2)


def _to_builtin(value):
    return value.item() if isinstance(value, np.generic) else value


def main() -> None:
    for d in (METRICS_DIR, FIG_DIR, MODELS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    start = time.time()

    train, test = split_train_test(load_raw())
    X, y = train[FEATURES], train[TARGET]
    log(f"Train {X.shape}, test {test.shape} (test is not used until the final evaluation)")

    log("Baseline: logistic regression (5-fold stratified CV)")
    print(run_baseline(X, y).round(4).to_string())

    log(f"LightGBM randomized search ({N_SEARCH_ITER} candidates x 5 folds)")
    lgbm_params = run_lightgbm_search(X, y)
    print(json.dumps(lgbm_params, indent=2))

    log("Class imbalance: class_weight='balanced' vs none")
    imbalance = run_imbalance_comparison(X, y, lgbm_params)
    print(imbalance.round(4).to_string())
    # The PD model is chosen on ROC-AUC (ranking); Brier breaks near-ties in favor of
    # probabilities that are on the right scale.
    lgbm_rows = imbalance.loc[["lightgbm_none", "lightgbm_balanced"]]
    best_auc = lgbm_rows["roc_auc_mean"].max()
    close = lgbm_rows[lgbm_rows["roc_auc_mean"] >= best_auc - lgbm_rows["roc_auc_std"].max()]
    chosen_variant = close["brier_mean"].idxmin()
    class_weight = "balanced" if chosen_variant.endswith("balanced") else None
    log(f"Chosen LightGBM variant: {chosen_variant}")

    final_pipeline = make_lightgbm(class_weight=class_weight, **lgbm_params)
    log("Out-of-fold predictions for calibration and cut-off selection")
    oof = out_of_fold_proba(final_pipeline, X, y)
    calibration_method, calibration_table = run_calibration_check(oof, y)
    print(calibration_table.round(5).to_string())
    log(f"Calibration: {calibration_method or 'not needed'}")

    calibrator = None
    oof_final = oof
    if calibration_method:
        calibrator = ProbabilityCalibrator(calibration_method).fit(oof, y)
        oof_final = calibrator.transform(oof)
    threshold = ks_threshold(y, oof_final)
    log(f"Operating cut-off (KS point on out-of-fold PD): {threshold:.4f}")

    log("Fitting final models on the full training set")
    final_pipeline.fit(X, y)
    model = CalibratedModel(final_pipeline, calibrator)
    baseline = make_logistic(log_transform=True).fit(X, y)

    # ---- Final evaluation on the untouched test set ----
    log("Evaluating once on the test set")
    X_test, y_test = test[FEATURES], test[TARGET]
    proba = model.predict_proba(X_test)[:, 1]
    proba_raw = final_pipeline.predict_proba(X_test)[:, 1]
    proba_base = baseline.predict_proba(X_test)[:, 1]
    test_metrics = {
        "logistic_regression": evaluate(y_test, proba_base),
        "lightgbm_raw": evaluate(y_test, proba_raw),
        "lightgbm_final": evaluate(y_test, proba),
    }
    for name, p in [
        ("logistic_regression", proba_base),
        ("lightgbm_raw", proba_raw),
        ("lightgbm_final", proba),
    ]:
        test_metrics[name]["ece"] = expected_calibration_error(y_test, p)
    test_table = pd.DataFrame(test_metrics).T
    test_table.to_csv(METRICS_DIR / "test_metrics.csv")
    print(test_table.round(4).to_string())

    cutoffs = cutoff_table(y_test, proba)
    cutoffs.to_csv(METRICS_DIR / "test_cutoff_table.csv", index=False)
    print(cutoffs.round(4).to_string(index=False))
    operating_point = threshold_report(y_test, proba, threshold)
    print("Operating point on test:", {k: round(v, 4) for k, v in operating_point.items()})

    plot_roc_pr(
        {"LightGBM": (y_test, proba), "Logistic regression": (y_test, proba_base)},
        base_rate=float(y_test.mean()),
        path=FIG_DIR / "test_roc_pr.png",
    )
    plot_calibration(
        {"LightGBM final": (y_test, proba), "Logistic regression": (y_test, proba_base)},
        FIG_DIR / "calibration_test.png",
        "Calibration on the test set",
    )

    log("SHAP explanations")
    run_shap(final_pipeline, X_test, proba, threshold)

    # ---- Save model and metadata for the app ----
    joblib.dump(model, MODEL_PATH)
    metadata = {
        "model": "LightGBM",
        "class_weight": class_weight,
        "calibration": calibration_method,
        "threshold": threshold,
        "lgbm_params": lgbm_params,
        "test_metrics": test_metrics["lightgbm_final"],
        "operating_point_test": operating_point,
        "feature_medians": X.median().to_dict(),
        "train_default_rate": float(y.mean()),
        "versions": {
            "python": platform.python_version(),
            "scikit-learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
            "shap": shap.__version__,
        },
    }
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f, indent=2)
    with open(METRICS_DIR / "final_model_summary.json", "w") as f:
        json.dump({k: v for k, v in metadata.items() if k != "feature_medians"}, f, indent=2)
    log(f"Saved {MODEL_PATH.relative_to(PROJECT_ROOT)}. Total time {time.time() - start:.0f}s")


if __name__ == "__main__":
    main()
