import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401
from janitor.functions.expand_grid import expand_grid
from janitor.testing_utils.strategies import (
    categoricaldf_strategy,
    df_strategy,
)


@pytest.fixture
def df():
    """fixture dataframe"""
    return pd.DataFrame(
        {
            "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
            "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
        }
    )


def test_others_not_dict(df):
    """Raise Error if `others` is not a dictionary."""
    with pytest.raises(TypeError, match="others should be one of.+"):
        df.expand_grid("frame", others=[2, 3])


@pytest.mark.xfail(reason="others should be supplied.")
def test_others_none(df):
    """Return DataFrame if no `others`, and df exists."""
    assert_frame_equal(df.expand_grid("df", others={}), df)


@pytest.mark.xfail(reason="others should not be an empty dict.")
def test_others_empty():
    """Return empty dict if no `others`."""
    assert (expand_grid(others={}), {})  # noqa : F631


@pytest.mark.xfail(reason="df_key is deprecated.")
def test_df_key(df):
    """Raise error if df exists and df_key is not supplied."""
    with pytest.raises(KeyError):
        expand_grid(df, others={"y": [5, 4, 3, 2, 1]})


@pytest.mark.xfail(reason="df_key is deprecated.")
def test_df_key_hashable(df):
    """Raise error if df exists and df_key is not Hashable."""
    with pytest.raises(TypeError):
        expand_grid(df, df_key=["a"], others={"y": [5, 4, 3, 2, 1]})


def test_scalar():
    """Raise TypeError if scalar value is provided."""
    with pytest.raises(TypeError, match="Expected a list-like object.+"):
        expand_grid(others={"x": 1})


def test_numpy_zero_d():
    """Raise ValueError if numpy array dimension is zero."""
    with pytest.raises(ValueError, match="Kindly provide a non-empty array.+"):
        expand_grid(others={"x": np.array([], dtype=int)})


def test_numpy_gt_2d():
    """Raise ValueError if numpy array dimension is greater than 2."""
    with pytest.raises(ValueError, match="expand_grid works only on 1D.+"):
        expand_grid(others={"x": np.array([[[2, 3]]])})


def test_series_empty():
    """Raise ValueError if Series is empty."""
    with pytest.raises(ValueError, match="Kindly provide a non-empty array.+"):
        expand_grid(others={"x": pd.Series([], dtype=int)})


def test_dataframe_empty():
    """Raise ValueError if DataFrame is empty."""
    with pytest.raises(ValueError, match="Kindly provide a non-empty array.+"):
        expand_grid(others={"x": pd.DataFrame([])})


def test_index_empty():
    """Raise ValueError if Index is empty."""
    with pytest.raises(ValueError, match="Kindly provide a non-empty array.+"):
        expand_grid(others={"x": pd.Index([], dtype=int)})


def test_dup_series(df):
    """
    Raise If key is duplicated.
    """
    others = {
        ("A", "B"): df.iloc[:, :2],
        "A": df["famid"],
    }
    with pytest.raises(ValueError, match="A is duplicated.+"):
        expand_grid(others=others)


def test_dup_array(df):
    """
    Raise If key is duplicated.
    """
    others = {
        ("A", "B"): df.iloc[:, :2],
        "A": df["famid"].to_numpy(),
    }
    with pytest.raises(ValueError, match="A is duplicated.+"):
        expand_grid(others=others)


def test_dup_MultiIndex(df):
    """
    Raise If key is duplicated.
    """
    others = {
        "A": df["famid"],
        ("A", "B"): pd.MultiIndex.from_frame(df.iloc[:, :2]),
    }
    with pytest.raises(ValueError, match=r"A in \('A', 'B'\) is duplicated.+"):
        expand_grid(others=others)


def test_wrong_type_MultiIndex(df):
    """
    Raise If key is wrong tyupe.
    """
    others = {
        "A": df["famid"],
        "B": pd.MultiIndex.from_frame(df.iloc[:, :2]),
    }
    with pytest.raises(
        TypeError, match=r"Expected a tuple of labels as key.+"
    ):
        expand_grid(others=others)


def test_wrong_length_keys_MultiIndex(df):
    """
    Raise If key length does not match the number of levels.
    """
    others = {
        "A": df["famid"],
        ("A",): pd.MultiIndex.from_frame(df.iloc[:, :2]),
    }
    with pytest.raises(
        ValueError, match=r"The number of labels in \('A',\).+"
    ):
        expand_grid(others=others)


def test_dup_DataFrame(df):
    """
    Raise If key is duplicated.
    """
    others = {
        "A": df["famid"],
        ("A", "B"): df.iloc[:, :2],
    }
    with pytest.raises(ValueError, match=r"A in \('A', 'B'\) is duplicated.+"):
        expand_grid(others=others)


def test_wrong_length_keys_DataFrame(df):
    """
    Raise If key length does not match the number of columns.
    """
    others = {
        "A": df["famid"],
        ("A",): df.iloc[:, :2],
    }
    with pytest.raises(
        ValueError, match=r"The number of labels in \('A',\).+"
    ):
        expand_grid(others=others)


def test_wrong_type_DataFrame(df):
    """
    Raise If key is wrong tyupe.
    """
    others = {
        "A": df["famid"],
        "B": df.iloc[:, :2],
    }
    with pytest.raises(
        TypeError, match=r"Expected a tuple of labels as key.+"
    ):
        expand_grid(others=others)


def test_dup_2d_array(df):
    """
    Raise If key is duplicated.
    """
    others = {
        "A": df["famid"],
        ("A", "B"): df.iloc[:, :2].to_numpy(),
    }
    with pytest.raises(ValueError, match=r"A in \('A', 'B'\) is duplicated.+"):
        expand_grid(others=others)


def test_wrong_length_keys_2d_array(df):
    """
    Raise If key length does not match the number of columns.
    """
    others = {
        "A": df["famid"],
        ("A",): df.iloc[:, :2].to_numpy(),
    }
    with pytest.raises(
        ValueError, match=r"The number of labels in \('A',\).+"
    ):
        expand_grid(others=others)


def test_wrong_type_2d_array(df):
    """
    Raise If key is wrong tyupe.
    """
    others = {
        "A": df["famid"],
        "B": df.iloc[:, :2].to_numpy(),
    }
    with pytest.raises(
        TypeError, match=r"Expected a tuple of labels as key.+"
    ):
        expand_grid(others=others)


@settings(deadline=None, max_examples=10)
@given(df=df_strategy())
def test_various(df):
    """Test expand_grid output for various inputs."""
    A = df["a"]
    B = df["cities"]
    frame = df.loc[:, ["decorated-elephant", "animals@#$%^"]]
    MI = pd.MultiIndex.from_frame(frame)

    others = {
        "a": pd.Index(A),
        "cities": B,
        ("decorated-elephant", "animals@#$%^"): frame,
        ("1", "2"): MI,
        "1D": A.to_numpy(),
    }
    result = expand_grid(others=others)
    expected = (
        pd.merge(A, B, how="cross")
        .merge(frame, how="cross")
        .merge(frame.set_axis(["1", "2"], axis="columns"), how="cross")
        .merge(A.rename("1D"), how="cross")
    )
    assert_frame_equal(result, expected)


@settings(deadline=None, max_examples=10)
@given(df=df_strategy())
def test_variouss(df):
    """Test expand_grid output for various inputs."""
    A = df["animals@#$%^"].astype("category")
    B = df["cities"]
    frame = df.loc[:, ["cities", "animals@#$%^"]].set_axis(
        ["1D", "2D"], axis=1
    )

    others = {
        "animals@#$%^": A.array,
        "cities": B,
        ("1D", "2D"): frame.to_numpy(),
    }
    result = expand_grid(others=others)
    expected = pd.merge(A, B, how="cross").merge(frame, how="cross")
    assert_frame_equal(result, expected)


@settings(deadline=None, max_examples=10)
@given(df=categoricaldf_strategy())
def test_keys_mix(df):
    """Test expand_grid output for a mix of tuple and scalar keys."""
    A = df["names"]
    B = pd.MultiIndex.from_frame(df)
    B.names = ["foo", "bar"]
    others = {("names", "A"): A, ("foo", "bar"): B}
    result = expand_grid(others=others)
    expected = pd.merge(A, df, how="cross")
    expected.columns = pd.MultiIndex.from_tuples(
        [("names", "A"), ("foo", ""), ("bar", "")]
    )
    assert_frame_equal(result, expected)


@settings(deadline=None, max_examples=10)
@given(df=categoricaldf_strategy())
def test_keys_mixx(df):
    """Test expand_grid output for a mix of tuple and scalar keys."""
    A = df["names"]
    B = df["numbers"]
    C = df["numbers"].rename("c")
    others = {("names", "B", "C"): A, ("foo", "bar"): B, "c": C.tolist()}
    result = expand_grid(others=others)
    expected = pd.merge(A, B, how="cross").merge(C, how="cross")
    expected.columns = pd.MultiIndex.from_tuples(
        [("names", "B", "C"), ("foo", "bar", ""), ("c", "", "")]
    )
    assert_frame_equal(result, expected)
