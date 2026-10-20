"""Cleaning rules learned in the EDA (notebooks/01_eda.ipynb), as a scikit-learn transformer.

Everything that depends on the data (capping thresholds) is learned in `fit`, so when the
transformer sits inside a Pipeline it only ever learns from the training fold.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

LATE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]
FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
# Heavy right tails found in the EDA. Late-payment counts are left uncapped: after removing the
# 96/98 codes their maximum is small and every extra event carries real risk.
CAP_COLS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberRealEstateLoansOrLines",
    "NumberOfDependents",
]
# Values in the late-payment columns that are not real counts (nobody is late 98 times in 2 years).
SPECIAL_LATE_CODES = (96, 98)
MIN_ADULT_AGE = 18

FLAG_COLS = ["MonthlyIncome_missing", "NumberOfDependents_missing", "LatePayment_special_code"]
ENGINEERED_COLS = ["TotalPastDue"]
OUTPUT_COLS = FEATURES + FLAG_COLS + ENGINEERED_COLS


class CreditDataCleaner(BaseEstimator, TransformerMixin):
    """Apply the EDA cleaning rules and leave remaining gaps as NaN for a downstream imputer.

    Steps, in order:
    1. Flag missing income / dependents (missingness is informative: those clients default less).
    2. Replace the 96/98 late-payment codes with NaN and flag them.
    3. Treat impossible ages (< 18, e.g. age 0) as missing.
    4. Cap heavy-tailed columns at the `cap_quantile` learned on the training data.
    5. Add `TotalPastDue`, the sum of the three late-payment counts.
    """

    def __init__(self, cap_quantile: float = 0.99):
        self.cap_quantile = cap_quantile

    def fit(self, X: pd.DataFrame, y=None):
        if not 0.5 < self.cap_quantile <= 1:
            raise ValueError("cap_quantile must be in (0.5, 1]")
        X = self._apply_rules(self._validate(X))
        self.caps_ = {col: float(X[col].quantile(self.cap_quantile)) for col in CAP_COLS}
        self.n_features_in_ = len(FEATURES)
        self.feature_names_in_ = np.array(FEATURES, dtype=object)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        check_is_fitted(self, "caps_")
        X = self._apply_rules(self._validate(X))
        for col, cap in self.caps_.items():
            X[col] = X[col].clip(upper=cap)
        # skipna=False: if the late counts are unknown, the total is unknown too.
        X["TotalPastDue"] = X[LATE_COLS].sum(axis=1, skipna=False)
        return X[OUTPUT_COLS]

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        return np.array(OUTPUT_COLS, dtype=object)

    @staticmethod
    def _validate(X: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(X, pd.DataFrame):
            raise TypeError("CreditDataCleaner expects a pandas DataFrame with the raw columns")
        missing = [col for col in FEATURES if col not in X.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        return X[FEATURES].astype(float)

    @staticmethod
    def _apply_rules(X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        X["MonthlyIncome_missing"] = X["MonthlyIncome"].isna().astype(float)
        X["NumberOfDependents_missing"] = X["NumberOfDependents"].isna().astype(float)

        special = X[LATE_COLS].isin(SPECIAL_LATE_CODES)
        X["LatePayment_special_code"] = special.any(axis=1).astype(float)
        X[LATE_COLS] = X[LATE_COLS].mask(special)

        X.loc[X["age"] < MIN_ADULT_AGE, "age"] = np.nan
        return X
