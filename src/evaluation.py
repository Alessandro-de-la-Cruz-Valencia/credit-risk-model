"""Business-facing evaluation: cut-off tables and the plots used in the README."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

DEFAULT_APPROVAL_RATES = (0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.00)


def cutoff_table(y_true, proba, approval_rates=DEFAULT_APPROVAL_RATES) -> pd.DataFrame:
    """Approve the lowest-risk clients first and report, for each approval rate, the PD cut-off
    and the default rate among the approved (the "bad rate" the lender would carry)."""
    y = np.asarray(y_true)
    p = np.asarray(proba)
    order = np.argsort(p, kind="stable")
    y_sorted, p_sorted = y[order], p[order]
    n = len(y)
    rows = []
    for rate in approval_rates:
        k = int(round(rate * n))
        if k == 0:
            continue
        approved = y_sorted[:k]
        rows.append(
            {
                "approval_rate": k / n,
                "pd_cutoff": float(p_sorted[k - 1]),
                "approved": k,
                "defaults_approved": int(approved.sum()),
                "bad_rate_approved": float(approved.mean()),
                "defaults_rejected_share": float(
                    (y_sorted[k:].sum() / y.sum()) if y.sum() else np.nan
                ),
            }
        )
    return pd.DataFrame(rows)


def threshold_report(y_true, proba, threshold: float) -> dict[str, float]:
    """Approval rate and bad rate when every client with PD < threshold is approved."""
    y = np.asarray(y_true)
    approved = np.asarray(proba) < threshold
    return {
        "threshold": float(threshold),
        "approval_rate": float(approved.mean()),
        "bad_rate_approved": float(y[approved].mean()) if approved.any() else float("nan"),
        "defaults_caught": float(y[~approved].sum() / y.sum()),
    }


def ks_threshold(y_true, proba) -> float:
    """Score where the gap between the defaulters' and non-defaulters' cumulative distributions
    is largest (the KS point)."""
    fpr, tpr, thresholds = roc_curve(y_true, proba)
    return float(thresholds[np.argmax(tpr - fpr)])


def plot_roc_pr(curves: dict[str, tuple], base_rate: float, path) -> None:
    """ROC and precision-recall curves for several models. `curves` maps name -> (y, proba)."""
    fig, (ax_roc, ax_pr) = plt.subplots(1, 2, figsize=(12, 5))
    for name, (y, p) in curves.items():
        fpr, tpr, _ = roc_curve(y, p)
        prec, rec, _ = precision_recall_curve(y, p)
        ax_roc.plot(fpr, tpr, label=f"{name} (AUC {roc_auc_score(y, p):.4f})")
        ax_pr.plot(rec, prec, label=f"{name} (AP {average_precision_score(y, p):.4f})")
    ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curve (test)")
    ax_pr.axhline(base_rate, color="k", ls="--", lw=1, label=f"Random ({base_rate:.3f})")
    ax_pr.set(xlabel="Recall", ylabel="Precision", title="Precision-recall curve (test)")
    for ax in (ax_roc, ax_pr):
        ax.legend(loc="best", fontsize=9)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_calibration(curves: dict[str, tuple], path, title: str, n_bins: int = 15) -> None:
    """Reliability diagram on quantile bins. `curves` maps name -> (y, proba)."""
    fig, ax = plt.subplots(figsize=(6, 5))
    upper = 0.0
    for name, (y, p) in curves.items():
        frac_pos, mean_pred = calibration_curve(y, p, n_bins=n_bins, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", ms=4, label=name)
        upper = max(upper, mean_pred.max(), frac_pos.max())
    ax.plot([0, upper], [0, upper], "k--", lw=1, label="Perfect calibration")
    ax.set(xlabel="Mean predicted PD", ylabel="Observed default rate", title=title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
