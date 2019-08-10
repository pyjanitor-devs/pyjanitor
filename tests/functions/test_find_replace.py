import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def df():
    return pd.DataFrame(
        {"a": [1, np.nan, 3], "b": [2, 3, 1], "c": [2, np.nan, 2]}
    )


@pytest.mark.functions
def test_find_replace_single(df):
    assert df["a"].iloc[2] == 3
    df.find_replace("a", {3: 5})
    assert df["a"].iloc[2] == 5

    assert sum(df["c"] == 2) == 2
    assert sum(df["c"] == 5) == 0
    df.find_replace("c", {2: 5})
    assert sum(df["c"] == 2) == 0
    assert sum(df["c"] == 5) == 2


@pytest.mark.functions
def test_find_replace_null_raises_error(df):
    with pytest.raises(ValueError):
        df.find_replace("a", {np.nan: 5})
