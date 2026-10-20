import numpy as np

from src.features import OUTPUT_COLS
from src.models import make_lightgbm, make_logistic
from src.pipeline import LOG_COLS, build_preprocessor


def test_preprocessor_outputs_complete_dataframe(raw_df):
    out = build_preprocessor().fit_transform(raw_df)
    assert list(out.columns) == OUTPUT_COLS
    assert not out.isna().any().any()


def test_scaled_preprocessor_is_standardized(raw_df):
    out = build_preprocessor(scale=True).fit_transform(raw_df)
    assert np.allclose(out.mean(), 0, atol=1e-8)


def test_log_transform_applies_log1p(raw_df):
    plain = build_preprocessor().fit_transform(raw_df)
    logged = build_preprocessor(log_transform=True).fit_transform(raw_df)
    assert np.allclose(logged[LOG_COLS], np.log1p(plain[LOG_COLS]))
    assert np.allclose(logged["age"], plain["age"])


def test_models_fit_and_predict_probabilities(raw_df, target):
    for pipe in (make_logistic(), make_lightgbm(n_estimators=20, min_child_samples=5)):
        proba = pipe.fit(raw_df, target).predict_proba(raw_df)[:, 1]
        assert proba.shape == (len(raw_df),)
        assert ((proba >= 0) & (proba <= 1)).all()
