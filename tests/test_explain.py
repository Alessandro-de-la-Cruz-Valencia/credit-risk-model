import numpy as np

from src.explain import shap_values, top_reasons
from src.models import make_lightgbm


def test_shap_values_add_up_to_model_output(raw_df, target):
    pipe = make_lightgbm(n_estimators=30, min_child_samples=5).fit(raw_df, target)
    explanation = shap_values(pipe, raw_df.head(5))
    raw_margin = pipe.predict_proba(raw_df.head(5), raw_score=True)
    reconstructed = explanation.base_values + explanation.values.sum(axis=1)
    assert np.allclose(reconstructed, raw_margin, atol=1e-6)


def test_top_reasons_sorted_by_absolute_impact(raw_df, target):
    pipe = make_lightgbm(n_estimators=30, min_child_samples=5).fit(raw_df, target)
    reasons = top_reasons(shap_values(pipe, raw_df.head(1))[0], n=3)
    assert len(reasons) == 3
    impacts = reasons["shap"].abs().to_numpy()
    assert np.all(np.diff(impacts) <= 0)
    assert set(reasons["effect"]) <= {"raises risk", "lowers risk"}
