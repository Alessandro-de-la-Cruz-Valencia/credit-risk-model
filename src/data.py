"""Data loading and the train/test split used everywhere in the project."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "cs-training.csv"

TARGET = "SeriousDlqin2yrs"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_raw(path: Path | str = RAW_DATA_PATH) -> pd.DataFrame:
    """Read the Kaggle CSV. The first, unnamed column is a row id, so it becomes the index."""
    df = pd.read_csv(path, index_col=0)
    df.index.name = "id"
    return df


def split_train_test(
    df: pd.DataFrame, test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified split so both sets keep the same default rate.

    The test set is created once, with a fixed seed, and is only used for the final evaluation.
    """
    train, test = train_test_split(
        df, test_size=test_size, stratify=df[TARGET], random_state=random_state
    )
    return train, test
