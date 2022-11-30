import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import settings

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_remove_empty(df):
    """This test ensures that there are no columns that are completely null"""
    df = df.remove_empty()
    for col in df.columns:
        assert not pd.isna(df[col]).all()
    for r, d in df.iterrows():
        assert not pd.isna(d).all()


@pytest.mark.functions
def test_index_after_remove_empty():
    """This test ensures that the index is reset correctly"""
    df = pd.DataFrame()
    df["a"] = [1, np.nan, np.nan, 3, np.nan, 6]
    df["b"] = [1, np.nan, 1, 3, np.nan, 6]
    df_nonempty = df.remove_empty()
    assert np.array_equal(
        np.asarray(df_nonempty.index), np.asarray(range(0, len(df_nonempty)))
    )


@pytest.mark.functions
def test_reset_index_false():
    """Test output when reset_index is False"""
    df = pd.DataFrame()
    df["a"] = [1, np.nan, np.nan, 3, np.nan, 6]
    df["b"] = [1, np.nan, 1, 3, np.nan, 6]
    df_nonempty = df.remove_empty(reset_index=False)
    assert np.array_equal(
        df.notna().any(axis=1).to_numpy().nonzero()[0],
        df_nonempty.index.to_numpy(),
    )
