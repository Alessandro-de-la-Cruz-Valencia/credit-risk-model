import numpy as np
import pandas as pd
import pytest

from src.features import FEATURES


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small synthetic dataset with the raw Kaggle columns and the data problems seen in the EDA."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "RevolvingUtilizationOfUnsecuredLines": rng.uniform(0, 1.2, n),
            "age": rng.integers(21, 90, n).astype(float),
            "NumberOfTime30-59DaysPastDueNotWorse": rng.poisson(0.4, n).astype(float),
            "DebtRatio": rng.uniform(0, 2, n),
            "MonthlyIncome": rng.lognormal(8.5, 0.6, n),
            "NumberOfOpenCreditLinesAndLoans": rng.integers(0, 30, n).astype(float),
            "NumberOfTimes90DaysLate": rng.poisson(0.2, n).astype(float),
            "NumberRealEstateLoansOrLines": rng.integers(0, 5, n).astype(float),
            "NumberOfTime60-89DaysPastDueNotWorse": rng.poisson(0.2, n).astype(float),
            "NumberOfDependents": rng.integers(0, 5, n).astype(float),
        }
    )[FEATURES]
    df.loc[0:39, "MonthlyIncome"] = np.nan
    df.loc[0:9, "NumberOfDependents"] = np.nan
    late = [
        "NumberOfTime30-59DaysPastDueNotWorse",
        "NumberOfTime60-89DaysPastDueNotWorse",
        "NumberOfTimes90DaysLate",
    ]
    df.loc[50:52, late] = 98
    df.loc[53, late] = 96
    df.loc[60, "age"] = 0
    df.loc[61, "RevolvingUtilizationOfUnsecuredLines"] = 50_000
    return df


@pytest.fixture
def target(raw_df) -> pd.Series:
    """Target that depends on utilization and late payments, so models have signal to learn."""
    rng = np.random.default_rng(1)
    util = raw_df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=1.5)
    late = raw_df["NumberOfTimes90DaysLate"].clip(upper=5)
    logit = -3 + 2.5 * util + 1.0 * late
    return pd.Series((rng.uniform(size=len(raw_df)) < 1 / (1 + np.exp(-logit))).astype(int))
