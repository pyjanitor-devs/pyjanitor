from functools import reduce

import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import settings
from pandas.testing import assert_frame_equal

from janitor.functions import expand_grid
from janitor.testing_utils.strategies import df_strategy
from janitor.testing_utils.strategies import categoricaldf_strategy


@given(df=df_strategy())
@settings(deadline=None)
def test_others_not_dict(df):
    """Raise Error if `others` is not a dictionary."""
    with pytest.raises(TypeError):
        df.expand_grid("frame", others=[2, 3])


@given(df=df_strategy())
@settings(deadline=None)
def test_others_none(df):
    """Return DataFrame if no `others`, and df exists."""
    assert_frame_equal(df.expand_grid("df"), df)


def test_others_empty():
    """Return None if no `others`."""
    assert (expand_grid(), None)  # noqa : F631


@given(df=df_strategy())
@settings(deadline=None)
def test_df_key(df):
    """Raise error if df exists and df_key is not supplied."""
    with pytest.raises(KeyError):
        expand_grid(df, others={"y": [5, 4, 3, 2, 1]})


@given(df=df_strategy())
@settings(deadline=None)
def test_df_key_hashable(df):
    """Raise error if df exists and df_key is not Hashable."""
    with pytest.raises(TypeError):
        expand_grid(df, df_key=["a"], others={"y": [5, 4, 3, 2, 1]})


def test_numpy_zero_d():
    """Raise ValueError if numpy array dimension is zero."""
    with pytest.raises(ValueError):
        expand_grid(others={"x": np.array([], dtype=int)})


def test_numpy_gt_2d():
    """Raise ValueError if numpy array dimension is greater than 2."""
    with pytest.raises(ValueError):
        expand_grid(others={"x": np.array([[[2, 3]]])})


def test_series_empty():
    """Raise ValueError if Series is empty."""
    with pytest.raises(ValueError):
        expand_grid(others={"x": pd.Series([], dtype=int)})


def test_dataframe_empty():
    """Raise ValueError if DataFrame is empty."""
    with pytest.raises(ValueError):
        expand_grid(others={"x": pd.DataFrame([])})


def test_index_empty():
    """Raise ValueError if Index is empty."""
    with pytest.raises(ValueError):
        expand_grid(others={"x": pd.Index([], dtype=int)})


@settings(deadline=None)
@given(df=df_strategy())
def test_series(df):
    """Test expand_grid output for Series input."""
    A = df["a"]
    B = df["cities"]
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    B = df.loc[:, ["cities"]]
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_series_dataframe(df):
    """Test expand_grid output for Series and DataFrame inputs."""
    A = df["a"]
    B = df.iloc[:, [1, 2]]
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_series_multiindex_dataframe(df):
    """
    Test expand_grid output
    if the DataFrame's columns is a MultiIndex.
    """
    A = df["a"]
    B = df.iloc[:, [1, 2]]
    B.columns = pd.MultiIndex.from_arrays([["C", "D"], B.columns])
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    expected = A.assign(key=1).merge(B.droplevel(level=1, axis=1), how="cross")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_tuples(
        [
            ("A", "a", ""),
            ("B", "C", "Bell__Chart"),
            ("B", "D", "decorated-elephant"),
        ],
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_numpy_1d(df):
    """Test expand_grid output for a 1D numpy array."""
    A = df["a"].to_numpy()
    B = df["cities"]
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]].rename(columns={"a": 0})
    B = df.loc[:, ["cities"]]
    expected = A.merge(B, how="cross")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=categoricaldf_strategy())
def test_numpy_2d(df):
    """Test expand_grid output for a 2D numpy array"""
    A = df["names"]
    base = df.loc[:, ["numbers"]].assign(num=df.numbers * 4)
    B = base.to_numpy(dtype=int)
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["names"]]
    B = base.set_axis([0, 1], axis=1)
    expected = A.merge(B, how="cross")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B", "B"], expected.columns]
    )
    assert_frame_equal(result, expected, check_dtype=False)


@settings(deadline=None)
@given(df=df_strategy())
def test_index(df):
    """Test expand_grid output for a pandas Index that has a name."""
    A = pd.Index(df["a"])
    B = df["cities"]
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    B = df.loc[:, ["cities"]]
    expected = A.merge(B, how="cross")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_index_name_none(df):
    """Test expand_grid output for a pandas Index without a name."""
    A = pd.Index(df["a"].array, name=None)
    B = df["cities"]
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    B = df.loc[:, ["cities"]]
    expected = A.merge(B, how="cross")
    expected.columns = pd.MultiIndex.from_arrays([["A", "B"], [0, "cities"]])
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=categoricaldf_strategy())
def test_multiindex(df):
    """Test expand_grid output for a pandas MultiIndex with a name."""
    A = df["names"]
    base = df.loc[:, ["numbers"]].assign(num=df.numbers * 4)
    B = pd.MultiIndex.from_frame(base)
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["names"]]
    B = base.copy()
    expected = A.merge(B, how="cross")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=categoricaldf_strategy())
def test_multiindex_names_none(df):
    """Test expand_grid output for a pandas MultiIndex without a name."""
    A = df["names"]
    base = df.loc[:, ["numbers"]].assign(num=df.numbers * 4)
    B = pd.MultiIndex.from_frame(base, names=[None, None])
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["names"]]
    B = base.copy()
    expected = A.merge(B, how="cross")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B", "B"], ["names", 0, 1]]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_pandas_extension_array(df):
    """Test expand_grid output for a pandas array."""
    A = df["a"]
    B = df["cities"].astype("string").array
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    B = df.loc[:, ["cities"]].astype("string").set_axis([0], axis=1)
    expected = A.merge(B, how="cross")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_sequence(df):
    """Test expand_grid output for list."""
    A = df["a"].to_list()
    B = df["cities"]
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]].rename(columns={"a": 0})
    B = df.loc[:, ["cities"]]
    expected = A.merge(B, how="cross")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected, check_dtype=False)


@settings(deadline=None)
@given(df=df_strategy())
def test_scalar(df):
    """Test expand_grid output for a scalar value."""
    A = df["a"]
    B = 2
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    B = pd.DataFrame([2])
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected, check_dtype=False)


@settings(deadline=None)
@given(df=df_strategy())
def test_chain_df(df):
    """Test expand_grid in a method-chain operation."""
    A = df["a"]
    B = df[["cities"]]
    others = {"A": A}
    result = B.expand_grid(df_key="city", others=others)
    A = df.loc[:, ["a"]]
    expected = B.assign(key=1).merge(A.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["city", "A"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_series_name(df):
    """Test expand_grid where the Series has no name."""
    A = df["a"].rename(None)
    B = df[["cities"]]
    others = {"A": A}
    result = B.expand_grid(df_key="city", others=others)
    A = df.loc[:, ["a"]]
    expected = B.assign(key=1).merge(A.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["city", "A"], ["cities", 0]]
    )
    assert_frame_equal(result, expected)


def test_extension_array():
    """Test output on an extension array"""
    others = dict(
        id=pd.Categorical(
            values=(2, 1, 1, 2, 1), categories=(1, 2, 3), ordered=True
        ),
        year=(2018, 2018, 2019, 2020, 2020),
        gender=pd.Categorical(("female", "male", "male", "female", "male")),
    )

    expected = expand_grid(others=others).droplevel(axis=1, level=-1)
    others = [pd.Series(val).rename(key) for key, val in others.items()]

    func = lambda x, y: pd.merge(x, y, how="cross")  # noqa: E731
    actual = reduce(func, others)
    assert_frame_equal(expected, actual, check_dtype=False)
