import numpy as np
import pytest

from src.evaluation import cutoff_table, ks_threshold, threshold_report


def test_cutoff_table_approves_lowest_risk_first():
    y = np.array([0, 0, 0, 0, 1, 0, 0, 1, 1, 1])
    p = np.linspace(0.05, 0.95, 10)
    table = cutoff_table(y, p, approval_rates=(0.5, 1.0))
    first = table.iloc[0]
    assert first["approved"] == 5
    assert first["pd_cutoff"] == pytest.approx(p[4])
    assert first["bad_rate_approved"] == pytest.approx(0.2)
    assert first["defaults_rejected_share"] == pytest.approx(0.75)
    assert table.iloc[1]["bad_rate_approved"] == pytest.approx(y.mean())


def test_bad_rate_increases_with_approval_rate():
    rng = np.random.default_rng(0)
    p = rng.uniform(size=2000)
    y = (rng.uniform(size=2000) < p).astype(int)
    rates = cutoff_table(y, p)["bad_rate_approved"].to_numpy()
    assert np.all(np.diff(rates) >= 0)


def test_threshold_report():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.9])
    r = threshold_report(y, p, threshold=0.25)
    assert r["approval_rate"] == 0.5
    assert r["bad_rate_approved"] == 0.0
    assert r["defaults_caught"] == 1.0


def test_ks_threshold_separates_classes():
    y = np.array([0, 0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.3, 0.6, 0.7, 0.8])
    assert 0.3 < ks_threshold(y, p) <= 0.6
