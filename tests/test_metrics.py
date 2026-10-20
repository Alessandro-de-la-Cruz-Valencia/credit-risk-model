import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.metrics import evaluate, gini, ks_statistic


def test_perfect_and_random_scores():
    y = np.array([0, 0, 0, 1, 1])
    assert gini(y, [0.1, 0.2, 0.3, 0.8, 0.9]) == pytest.approx(1.0)
    assert ks_statistic(y, [0.1, 0.2, 0.3, 0.8, 0.9]) == pytest.approx(1.0)
    assert gini(y, [0.5] * 5) == pytest.approx(0.0)


def test_gini_matches_auc():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    s = rng.uniform(size=500) + 0.3 * y
    assert gini(y, s) == pytest.approx(2 * roc_auc_score(y, s) - 1)


def test_ks_against_manual_computation():
    y = np.array([0, 0, 1, 0, 1, 1])
    s = np.array([0.1, 0.4, 0.35, 0.5, 0.7, 0.8])
    # Max gap between cumulative distributions of positives and negatives.
    thresholds = np.unique(s)
    gaps = [abs((s[y == 1] >= t).mean() - (s[y == 0] >= t).mean()) for t in thresholds]
    assert ks_statistic(y, s) == pytest.approx(max(gaps))


def test_evaluate_returns_all_metrics():
    m = evaluate([0, 1, 0, 1], [0.2, 0.7, 0.4, 0.6])
    assert set(m) == {"roc_auc", "gini", "ks", "pr_auc", "brier"}
