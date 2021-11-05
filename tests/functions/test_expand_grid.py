import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from pandas.testing import assert_frame_equal
from janitor.testing_utils.strategies import (
    df_strategy,
    categoricaldf_strategy,
)
from janitor.functions import expand_grid


@given(df=df_strategy())
def test_others_not_dict(df):
    """Raise Error if `others` is not a dictionary."""
    with pytest.raises(TypeError):
        df.expand_grid("frame", others=[2, 3])


@given(df=df_strategy())
def test_others_None(df):
    """Return DataFrame if no `others`."""
    assert_frame_equal(df.expand_grid("df"), df)


def test_others_empty():
    """Return None if no `others`."""
    assert (expand_grid(), None)  # noqa : F631


@given(df=df_strategy())
def test_df_key(df):
    """Raise error if dataframe key is not supplied."""
    with pytest.raises(KeyError):
        expand_grid(df, others={"y": [5, 4, 3, 2, 1]})


@given(df=df_strategy())
def test_df_key_hashable(df):
    """Raise error if dataframe key is not Hashable."""
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


def test_Series_empty():
    """Raise ValueError if Series is empty."""
    with pytest.raises(ValueError):
        expand_grid(others={"x": pd.Series([], dtype=int)})


def test_DataFrame_empty():
    """Raise ValueError if Series is empty."""
    with pytest.raises(ValueError):
        expand_grid(others={"x": pd.DataFrame([])})


def test_Index_empty():
    """Raise ValueError if Index is empty."""
    with pytest.raises(ValueError):
        expand_grid(others={"x": pd.Index([], dtype=int)})


@settings(deadline=None)
@given(df=df_strategy())
def test_Series(df):
    """Test expand_grid output"""
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
def test_Series_DataFrame(df):
    """Test expand_grid output"""
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
def test_Series_MultiIndex_DataFrame(df):
    """Test expand_grid output"""
    A = df["a"]
    B = df.iloc[:, [1, 2]]
    B.columns = pd.MultiIndex.from_arrays([["C", "D"], B.columns])
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    B.columns = B.columns.map("_".join)
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_numpy_1d(df):
    """Test expand_grid output"""
    A = df["a"].to_numpy()
    B = df["cities"]
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]].rename(columns={"a": 0})
    B = df.loc[:, ["cities"]]
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=categoricaldf_strategy())
def test_numpy_2d(df):
    """Test expand_grid output"""
    A = df["names"]
    base = df.loc[:, ["numbers"]].assign(num=df.numbers * 4)
    B = base.to_numpy(dtype=int)
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["names"]]
    B = base.set_axis([0, 1], axis=1)
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_Index(df):
    """Test expand_grid output"""
    A = pd.Index(df["a"])
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
@given(df=categoricaldf_strategy())
def test_MultiIndex(df):
    """Test expand_grid output"""
    A = df["names"]
    base = df.loc[:, ["numbers"]].assign(num=df.numbers * 4)
    B = pd.MultiIndex.from_frame(base)
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["names"]]
    B = base.copy()
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_Pandas_extension_array(df):
    """Test expand_grid output"""
    A = df["a"]
    B = df["cities"].astype("string").array
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]]
    B = df.loc[:, ["cities"]].astype("string").set_axis([0], axis=1)
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_sequence(df):
    """Test expand_grid output"""
    A = df["a"].to_list()
    B = df["cities"]
    others = {"A": A, "B": B}
    result = expand_grid(others=others)
    A = df.loc[:, ["a"]].rename(columns={"a": 0})
    B = df.loc[:, ["cities"]]
    expected = A.assign(key=1).merge(B.assign(key=1), on="key")
    expected = expected.drop(columns="key")
    expected.columns = pd.MultiIndex.from_arrays(
        [["A", "B"], expected.columns]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None)
@given(df=df_strategy())
def test_scalar(df):
    """Test expand_grid output"""
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
    assert_frame_equal(result, expected)
