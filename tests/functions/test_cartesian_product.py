import pandas as pd
import pytest
from hypothesis import given, settings
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401
from janitor.functions.expand_grid import cartesian_product
from janitor.testing_utils.strategies import (
    df_strategy,
)


def test_not_pandas_object():
    """Raise Error if `others` is not a pandas Index/Series/DataFrame."""
    with pytest.raises(
        TypeError, match=r"input should be either a Pandas DataFrame.+"
    ):
        cartesian_product({"x": [1, 2]})


def test_Series_duplicated_label():
    """Raise if Series's name is duplicated."""
    with pytest.raises(ValueError, match=r"Label x in the Series at.+"):
        cartesian_product(
            pd.Series([1, 2], name="x"), pd.Series([4, 5], name="x")
        )


def test_Index_duplicated_label():
    """Raise if Index's name is duplicated."""
    with pytest.raises(ValueError, match=r"Label x in the Index at.+"):
        cartesian_product(
            pd.Index([1, 2], name="x"), pd.Index([4, 5], name="x")
        )


def test_Series_no_label():
    """Raise if Series's name is None."""
    with pytest.raises(
        ValueError, match=r"Kindly ensure the Series at position.+"
    ):
        cartesian_product(pd.Series([1, 2]))


def test_Index_no_label():
    """Raise if Index's name is None."""
    with pytest.raises(
        ValueError, match=r"Kindly ensure the Index at position.+"
    ):
        cartesian_product(pd.Index([1, 2]))


def test_MultiIndex_duplicated_label():
    """Raise if the MultiIndex label is duplicated."""
    mi = pd.MultiIndex.from_arrays([[1, 2], [3, 4]], names=["x", "y"])
    with pytest.raises(ValueError, match=r"Label x in the MultiIndex.+"):
        cartesian_product(mi, mi)


def test_MultiIndex_None_in_label():
    """Raise if Index's names contain None."""
    with pytest.raises(
        ValueError, match=r"Kindly ensure all levels in the MultiIndex.+"
    ):
        cartesian_product(pd.MultiIndex.from_arrays([[1, 2], [3, 4]]))


def test_DataFrame_duplicated_label():
    """Raise if the DataFrame's columns is duplicated."""
    df = pd.DataFrame([[1, 2], [3, 4]], columns=["x", "y"])
    with pytest.raises(ValueError, match=r"Label x in the DataFrame at.+"):
        cartesian_product(df, df)


@settings(deadline=None, max_examples=10)
@given(df=df_strategy())
def test_cartesian_output(df):
    """Test cartesian product output for various inputs."""
    frame = df.drop(columns=["a", "cities"])
    index = pd.Index(df["a"], name="rar")
    mi = pd.MultiIndex.from_frame(frame, names=["mi1", "mi2", "mi3"])
    result = cartesian_product(df["a"], df["cities"], frame, index, mi)
    expected = (
        pd.merge(df["a"], df["cities"], how="cross")
        .merge(frame, how="cross")
        .merge(df["a"].rename("rar"), how="cross")
        .merge(frame.set_axis(["mi1", "mi2", "mi3"], axis=1), how="cross")
    )
    assert_frame_equal(result, expected)


def test_cartesian_output_tuple():
    """
    Test output if there is a MultiIndex column
    """
    df1 = pd.DataFrame({("x", "y"): range(1, 3), ("y", "x"): [2, 1]})
    df2 = pd.DataFrame({"x": [1, 2, 3], "y": [3, 2, 1]})
    df3 = pd.DataFrame({"a": [2, 3], "b": ["a", "b"]})
    result = cartesian_product(df1, df2, df3)
    expected = (
        df1.set_axis(["l", "r"], axis=1)
        .merge(df2, how="cross")
        .merge(df3, how="cross")
        .set_axis(result.columns, axis=1)
    )
    assert_frame_equal(result, expected)
