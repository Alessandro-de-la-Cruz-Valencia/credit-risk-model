import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from src.calibration import CalibratedModel, ProbabilityCalibrator, expected_calibration_error


@pytest.fixture
def miscalibrated():
    rng = np.random.default_rng(0)
    true_p = rng.uniform(0.01, 0.3, 5000)
    y = (rng.uniform(size=5000) < true_p).astype(int)
    scores = np.clip(true_p * 2.5, 0, 0.99)  # systematically over-estimates risk
    return scores, y


@pytest.mark.parametrize("method", ["sigmoid", "isotonic"])
def test_calibrator_reduces_calibration_error(miscalibrated, method):
    scores, y = miscalibrated
    calibrated = ProbabilityCalibrator(method).fit(scores, y).transform(scores)
    assert expected_calibration_error(y, calibrated) < expected_calibration_error(y, scores)
    assert calibrated.mean() == pytest.approx(y.mean(), abs=0.01)


def test_sigmoid_keeps_ranking(miscalibrated):
    scores, y = miscalibrated
    calibrated = ProbabilityCalibrator("sigmoid").fit(scores, y).transform(scores)
    assert roc_auc_score(y, calibrated) == pytest.approx(roc_auc_score(y, scores))


def test_unknown_method_raises(miscalibrated):
    with pytest.raises(ValueError):
        ProbabilityCalibrator("magic").fit(*miscalibrated)


def test_calibrated_model_wraps_base(raw_df, target):
    from src.models import make_logistic

    base = make_logistic().fit(raw_df, target)
    raw = base.predict_proba(raw_df)[:, 1]
    calibrator = ProbabilityCalibrator("sigmoid").fit(raw, target)
    model = CalibratedModel(base, calibrator)
    proba = model.predict_proba(raw_df)
    assert np.allclose(proba.sum(axis=1), 1)
    assert np.allclose(proba[:, 1], calibrator.transform(raw))
    assert np.allclose(CalibratedModel(base).predict_proba(raw_df)[:, 1], raw)
