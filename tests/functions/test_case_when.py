import numpy as np
import pandas as pd
import pytest
from hypothesis import settings
from hypothesis import given
from pandas.testing import assert_frame_equal

from janitor.testing_utils.strategies import (
    categoricaldf_strategy,
    df_strategy,
)


def test_case_when_1():
    """Test case_when function."""
    df = pd.DataFrame(
        {
            "a": [0, 0, 1, 2, "hi"],
            "b": [0, 3, 4, 5, "bye"],
            "c": [6, 7, 8, 9, "wait"],
        }
    )
    expected = pd.DataFrame(
        {
            "a": [0, 0, 1, 2, "hi"],
            "b": [0, 3, 4, 5, "bye"],
            "c": [6, 7, 8, 9, "wait"],
            "value": ["x", 0, 8, 9, "hi"],
        }
    )
    result = df.case_when(
        ((df.a == 0) & (df.b != 0)) | (df.c == "wait"),
        df.a,
        (df.b == 0) & (df.a == 0),
        "x",
        default=df.c,
        column_name="value",
    )

    assert_frame_equal(result, expected)


def test_len_args(dataframe):
    """Raise ValueError if `args` length is less than 2."""
    with pytest.raises(
        ValueError,
        match="At least two arguments are required for the `args` parameter",
    ):
        dataframe.case_when(
            dataframe.a < 10, default="less_than_10", column_name="a"
        )


def test_args_even(dataframe):
    """Raise ValueError if `args` length is odd."""
    with pytest.raises(
        ValueError, match="The number of conditions and values do not match.+"
    ):
        dataframe.case_when(
            dataframe.a < 10,
            "less_than_10",
            dataframe.a == 5,
            default="five",
            column_name="a",
        )


def test_args_even_warning(dataframe):
    """
    Raise Warning if `args` length
    is odd and `default` is None.
    """
    with pytest.warns(DeprecationWarning):
        dataframe.case_when(
            dataframe.a < 10,
            "less_than_10",
            dataframe.a == 5,
            column_name="a",
        )


def test_column_name(dataframe):
    """Raise TypeError if `column_name` is not a string."""
    with pytest.raises(TypeError):
        dataframe.case_when(
            dataframe.a < 10,
            "less_than_10",
            default=dataframe.a,
            column_name=("a",),
        )


def test_default_ndim():
    """Raise ValueError if `default` ndim > 1."""
    df = pd.DataFrame({"a": range(20)})
    with pytest.raises(
        ValueError,
        match="The argument for the `default` parameter "
        "should either be a 1-D array.+",
    ):
        df.case_when(
            df.a < 10, "less_than_10", default=df.to_numpy(), column_name="a"
        )


@pytest.mark.xfail(reason="Error handled by pd.Series.mask")
def test_default_length():
    """Raise ValueError if `default` length != len(df)."""
    df = pd.DataFrame({"a": range(20)})
    with pytest.raises(
        ValueError,
        match=("The length of the argument for the `default` parameter is.+"),
    ):
        df.case_when(
            df.a < 10,
            "less_than_10",
            default=df.loc[:5, "a"],
            column_name="a",
        )


def test_error_multiple_conditions():
    """Raise ValueError for multiple conditions."""
    df = pd.DataFrame({"a": range(20)})
    with pytest.raises(ValueError):
        df.case_when(
            df.a < 10, "baby", df.a + 5, "kid", default=df.a, column_name="a"
        )


@given(df=df_strategy())
@settings(deadline=None)
def test_case_when_condition_callable(df):
    """Test case_when for callable."""
    result = df.case_when(
        lambda df: df.a < 10, "baby", default="bleh", column_name="bleh"
    )
    expected = np.where(df.a < 10, "baby", "bleh")
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=df_strategy())
@settings(deadline=None)
def test_case_when_condition_eval(df):
    """Test case_when for callable."""
    result = df.case_when("a < 10", "baby", default="bleh", column_name="bleh")
    expected = np.where(df.a < 10, "baby", "bleh")
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=df_strategy())
@settings(deadline=None)
def test_case_when_replacement_callable(df):
    """Test case_when for callable."""
    result = df.case_when(
        "a > 10",
        lambda df: df.a + 10,
        default=lambda df: df.a * 2,
        column_name="bleh",
    )
    expected = np.where(df.a > 10, df.a + 10, df.a * 2)
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
@settings(deadline=None)
def test_case_when_default_array(df):
    """
    Test case_when for scenarios where `default` is array-like
    """
    default = np.arange(len(df))
    result = df.case_when(
        "numbers > 1",
        lambda df: df.numbers + 10,
        default=default,
        column_name="bleh",
    )
    expected = np.where(df.numbers > 1, df.numbers + 10, default)
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected, check_dtype=False)


@given(df=categoricaldf_strategy())
@settings(deadline=None)
def test_case_when_default_list_like(df):
    """
    Test case_when for scenarios where `default` is list-like,
    but has no shape attribute.
    """
    default = range(len(df))
    result = df.case_when(
        "numbers > 1",
        lambda df: df.numbers + 10,
        default=default,
        column_name="bleh",
    )
    expected = np.where(df.numbers > 1, df.numbers + 10, default)
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
@settings(deadline=None)
def test_case_when_default_index(df):
    """
    Test case_when for scenarios where `default` is an index.
    """
    default = range(len(df))
    result = df.case_when(
        "numbers > 1",
        lambda df: df.numbers + 10,
        default=pd.Index(default),
        column_name="bleh",
    )
    expected = np.where(df.numbers > 1, df.numbers + 10, default)
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=df_strategy())
@settings(deadline=None)
def test_case_when_multiple_args(df):
    """Test case_when for multiple arguments."""
    result = df.case_when(
        df.a < 10,
        "baby",
        df.a.between(10, 20, "left"),
        "kid",
        lambda df: df.a.between(20, 30, "left"),
        "young",
        "30 <= a < 50",
        "mature",
        default="grandpa",
        column_name="elderly",
    )
    conditions = [
        df["a"] < 10,
        (df["a"] >= 10) & (df["a"] < 20),
        (df["a"] >= 20) & (df["a"] < 30),
        (df["a"] >= 30) & (df["a"] < 50),
    ]
    choices = ["baby", "kid", "young", "mature"]
    expected = np.select(conditions, choices, "grandpa")
    expected = df.assign(elderly=expected)
    assert_frame_equal(result, expected)
