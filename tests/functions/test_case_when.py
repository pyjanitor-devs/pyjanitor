import pandas as pd
import pytest
import numpy as np
from pandas.testing import assert_frame_equal
from hypothesis import assume, given
from janitor.testing_utils.strategies import (
    df_strategy,
    categoricaldf_strategy,
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
        df.c,
        column_name="value",
    )

    assert_frame_equal(result, expected)


@given(df=df_strategy())
def test_len_args(df):
    """Raise ValueError if `args` length is less than 3."""
    with pytest.raises(ValueError):
        df.case_when(df.a < 10, "less_than_10", column_name="a")


@given(df=df_strategy())
def test_args_even(df):
    """Raise ValueError if `args` length is even."""
    with pytest.raises(ValueError):
        df.case_when(
            df.a < 10, "less_than_10", df.a == 5, "five", column_name="a"
        )


@given(df=df_strategy())
def test_column_name(df):
    """Raise TypeError if `column_name` is not a string."""
    with pytest.raises(TypeError):
        df.case_when(df.a < 10, "less_than_10", df.a, column_name=("a",))


@given(df=df_strategy())
def test_default_ndim(df):
    """Raise ValueError if `default` ndim > 1."""
    with pytest.raises(ValueError):
        df.case_when(df.a < 10, "less_than_10", df, column_name="a")


@given(df=df_strategy())
def test_default_length(df):
    """Raise ValueError if `default` length != len(df)."""
    assume(len(df) > 10)
    with pytest.raises(ValueError):
        df.case_when(
            df.a < 10, "less_than_10", df.loc[:5, "a"], column_name="a"
        )


@given(df=df_strategy())
def test_error_multiple_conditions(df):
    """Raise ValueError for multiple conditions."""
    with pytest.raises(ValueError):
        df.case_when(df.a < 10, "baby", df.a + 5, "kid", df.a, column_name="a")


@given(df=df_strategy())
def test_case_when_condition_callable(df):
    """Test case_when for callable."""
    result = df.case_when(
        lambda df: df.a < 10, "baby", "bleh", column_name="bleh"
    )
    expected = np.where(df.a < 10, "baby", "bleh")
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=df_strategy())
def test_case_when_condition_eval(df):
    """Test case_when for callable."""
    result = df.case_when("a < 10", "baby", "bleh", column_name="bleh")
    expected = np.where(df.a < 10, "baby", "bleh")
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=df_strategy())
def test_case_when_replacement_callable(df):
    """Test case_when for callable."""
    result = df.case_when(
        "a > 10", lambda df: df.a + 10, lambda df: df.a * 2, column_name="bleh"
    )
    expected = np.where(df.a > 10, df.a + 10, df.a * 2)
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
def test_case_when_default_list(df):
    """
    Test case_when for scenarios where `default` is list-like,
    but not a Pandas or numpy object.
    """
    default = range(len(df))
    result = df.case_when(
        "numbers > 1", lambda df: df.numbers + 10, default, column_name="bleh"
    )
    expected = np.where(df.numbers > 1, df.numbers + 10, default)
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=categoricaldf_strategy())
def test_case_when_default_index(df):
    """Test case_when for scenarios where `default` is an index."""
    default = range(len(df))
    result = df.case_when(
        "numbers > 1",
        lambda df: df.numbers + 10,
        pd.Index(default),
        column_name="bleh",
    )
    expected = np.where(df.numbers > 1, df.numbers + 10, default)
    expected = df.assign(bleh=expected)
    assert_frame_equal(result, expected)


@given(df=df_strategy())
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
        "grandpa",
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
