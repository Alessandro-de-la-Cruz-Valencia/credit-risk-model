import numpy as np
import pandas as pd
import pytest

from src.features import LATE_COLS, OUTPUT_COLS, CreditDataCleaner


def test_output_columns_and_flags(raw_df):
    out = CreditDataCleaner().fit(raw_df).transform(raw_df)
    assert list(out.columns) == OUTPUT_COLS
    assert out["MonthlyIncome_missing"].sum() == 40
    assert out["NumberOfDependents_missing"].sum() == 10
    assert out["LatePayment_special_code"].sum() == 4


def test_special_late_codes_become_missing(raw_df):
    out = CreditDataCleaner().fit(raw_df).transform(raw_df)
    assert out.loc[50:53, LATE_COLS].isna().all().all()
    assert out.loc[50:53, "TotalPastDue"].isna().all()
    assert out[LATE_COLS].max().max() < 96


def test_impossible_age_becomes_missing(raw_df):
    out = CreditDataCleaner().fit(raw_df).transform(raw_df)
    assert np.isnan(out.loc[60, "age"])


def test_caps_are_learned_in_fit_only(raw_df):
    cleaner = CreditDataCleaner(cap_quantile=0.95).fit(raw_df)
    cap = cleaner.caps_["RevolvingUtilizationOfUnsecuredLines"]
    assert cap == pytest.approx(raw_df["RevolvingUtilizationOfUnsecuredLines"].quantile(0.95))

    new = raw_df.copy()
    new["RevolvingUtilizationOfUnsecuredLines"] = 1e9
    out = cleaner.transform(new)
    assert (out["RevolvingUtilizationOfUnsecuredLines"] == cap).all()
    assert cleaner.caps_["RevolvingUtilizationOfUnsecuredLines"] == cap


def test_total_past_due_is_sum_of_late_columns(raw_df):
    out = CreditDataCleaner().fit(raw_df).transform(raw_df)
    valid = out["LatePayment_special_code"] == 0
    expected = out.loc[valid, LATE_COLS].sum(axis=1)
    pd.testing.assert_series_equal(out.loc[valid, "TotalPastDue"], expected, check_names=False)


def test_does_not_modify_input(raw_df):
    before = raw_df.copy()
    CreditDataCleaner().fit(raw_df).transform(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_rejects_missing_columns(raw_df):
    with pytest.raises(ValueError, match="Missing columns"):
        CreditDataCleaner().fit(raw_df.drop(columns="age"))


def test_rejects_invalid_quantile(raw_df):
    with pytest.raises(ValueError):
        CreditDataCleaner(cap_quantile=0.2).fit(raw_df)


def test_transform_before_fit_raises(raw_df):
    from sklearn.exceptions import NotFittedError

    with pytest.raises(NotFittedError):
        CreditDataCleaner().transform(raw_df)
