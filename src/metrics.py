"""Credit-scoring metrics. All of them rank clients by predicted risk, none needs a cut-off."""

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve


def gini(y_true, y_score) -> float:
    """Gini = 2 * AUC - 1: the AUC rescaled so that 0 is random and 1 is perfect."""
    return 2 * roc_auc_score(y_true, y_score) - 1


def ks_statistic(y_true, y_score) -> float:
    """Kolmogorov-Smirnov: the largest gap between the cumulative score distributions of
    defaulters and non-defaulters, which equals max(TPR - FPR) over all thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def evaluate(y_true, y_score) -> dict[str, float]:
    """The metrics reported in every experiment.

    PR-AUC (average precision) is included because, with ~7% positives, it shows how well the
    model finds defaulters better than ROC-AUC does. The Brier score measures calibration.
    """
    return {
        "roc_auc": float(roc_auc_score(y_true, y_score)),
        "gini": float(gini(y_true, y_score)),
        "ks": ks_statistic(y_true, y_score),
        "pr_auc": float(average_precision_score(y_true, y_score)),
        "brier": float(brier_score_loss(y_true, y_score)),
    }
