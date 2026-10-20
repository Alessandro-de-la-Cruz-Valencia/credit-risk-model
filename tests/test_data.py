import pandas as pd

from src.data import TARGET, load_raw, split_train_test


def test_load_raw_uses_first_column_as_index(tmp_path):
    path = tmp_path / "cs-training.csv"
    path.write_text(",SeriousDlqin2yrs,age\n1,0,45\n2,1,30\n")
    df = load_raw(path)
    assert list(df.columns) == [TARGET, "age"]
    assert df.index.tolist() == [1, 2]


def test_split_is_stratified_and_reproducible():
    df = pd.DataFrame({TARGET: [0] * 900 + [1] * 100, "x": range(1000)})
    train1, test1 = split_train_test(df)
    train2, test2 = split_train_test(df)
    assert len(test1) == 200
    assert test1[TARGET].mean() == 0.1
    assert test1.index.equals(test2.index)
    assert set(train1.index).isdisjoint(test1.index)
