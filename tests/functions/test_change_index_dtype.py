import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df():
    """fixture dataframe"""
    return pd.DataFrame(
        {
            "fam_id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
        }
    )


@pytest.fixture
def df_multi():
    """MultiIndex dataframe fixture."""
    # https://stackoverflow.com/a/58328741/7175713
    tuples = list(
        zip(
            *[
                ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
                [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
            ]
        )
    )

    idx = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
    return pd.DataFrame(np.random.randn(8, 2), index=idx, columns=["A", "B"])


def test_type_axis(df):
    """Raise TypeError if wrong type is provided for axis."""
    with pytest.raises(TypeError, match="axis should be one of.+"):
        df.change_index_dtype(dtype=float, axis=1)


def test_axis_values(df):
    """Raise ValueError if wrong value is provided for axis."""
    msg = "axis should be either index or columns."
    with pytest.raises(ValueError, match=msg):
        df.change_index_dtype(dtype=float, axis="INDEX")


def test_type_dict_single_index(df):
    """Raise TypeError if dtype is a dict and single index."""
    msg = "Changing the dtype via a dictionary "
    msg = "is not supported for a single index."
    with pytest.raises(TypeError, match=msg):
        df.change_index_dtype(dtype={0: float})


def test_multiindex_diff_key_types(df_multi):
    """Test output if the keys in the dictionary are not the same type."""
    msg = "The levels in the dictionary "
    msg = "should be either all strings or all integers."
    with pytest.raises(TypeError, match=msg):
        df_multi.change_index_dtype(dtype={1: int, "first": "category"})


def test_multiindex_single_dtype(df_multi):
    """Test output if MultiIndex and a single dtype is passed."""
    actual = df_multi.change_index_dtype(dtype=str)
    expected = (
        df_multi.reset_index()
        .astype({"second": str, "first": str})
        .set_index(["first", "second"])
    )
    assert_frame_equal(expected, actual)


def test_multiindex_single_key(df_multi):
    """Test output if a dictionary is passed"""
    actual = df_multi.change_index_dtype(dtype={"second": int})
    expected = (
        df_multi.reset_index("second")
        .astype({"second": int})
        .set_index("second", append=True)
    )
    assert_frame_equal(expected, actual)


def test_multiindex_multiple_keys(df_multi):
    """Test output if a dictionary is passed"""
    actual = df_multi.change_index_dtype(
        dtype={"second": int, "first": "category"}
    )
    expected = (
        df_multi.reset_index()
        .astype({"second": int, "first": "category"})
        .set_index(["first", "second"])
    )
    assert_frame_equal(expected, actual)


def test_multiindex_multiple_keys_columns(df_multi):
    """Test output if a dictionary is passed"""
    actual = df_multi.T.change_index_dtype(
        dtype={"second": int, "first": "category"}, axis="columns"
    )
    expected = (
        df_multi.reset_index()
        .astype({"second": int, "first": "category"})
        .set_index(["first", "second"])
        .T
    )
    assert_frame_equal(expected, actual)


def test_single_index(df):
    """Test output if df.columns is not a multiindex"""
    actual = df.change_index_dtype(dtype=float)
    expected = df.set_index(np.arange(0.0, len(df)))
    assert_frame_equal(expected, actual)
