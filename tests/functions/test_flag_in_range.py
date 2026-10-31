"""Tests for `flag_in_range` function."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions import flag_in_range


@pytest.fixture
def range_df():
    """A small DataFrame for range-flagging tests."""
    return pd.DataFrame({"a": [1, 5, 10, 15, 20]})


@pytest.mark.functions
def test_functional_default(range_df):
    """Checks default (inclusive) behaviour as a method call."""
    expected = range_df.copy()
    expected["in_range_flag"] = [0, 1, 1, 1, 0]

    df = range_df.flag_in_range(column_name="a", low=5, high=15)

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_non_method_functional(range_df):
    """Checks behaviour when `flag_in_range` is used as a function."""
    expected = range_df.copy()
    expected["in_range_flag"] = [0, 1, 1, 1, 0]

    df = flag_in_range(range_df, column_name="a", low=5, high=15)

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_exclusive_bounds(range_df):
    """Checks that exclusive bounds do not flag boundary values as in range."""
    expected = range_df.copy()
    expected["in_range_flag"] = [0, 0, 1, 0, 0]

    df = range_df.flag_in_range(column_name="a", low=5, high=15, inclusive=False)

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_rename_output_column(range_df):
    """Checks output column is renamed when `flag_column_name` is given."""
    expected = range_df.copy()
    expected["flag"] = [0, 1, 1, 1, 0]

    df = range_df.flag_in_range(
        column_name="a", low=5, high=15, flag_column_name="flag"
    )

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_fail_column_not_in_df(range_df):
    """Checks ValueError is raised when `column_name` is not in df."""
    with pytest.raises(ValueError):
        range_df.flag_in_range(column_name="z", low=5, high=15)


@pytest.mark.functions
def test_fail_flag_column_exists(range_df):
    """Checks ValueError is raised when `flag_column_name` already exists."""
    with pytest.raises(ValueError):
        range_df.flag_in_range(column_name="a", low=5, high=15, flag_column_name="a")


@pytest.mark.functions
def test_missing_values_not_in_range():
    """Checks that missing values are never flagged as in range."""
    df = pd.DataFrame({"a": [1.0, None, 10.0]})
    expected = df.copy()
    expected["in_range_flag"] = [0, 0, 1]

    result = df.flag_in_range(column_name="a", low=5, high=15)

    assert_frame_equal(result, expected, check_dtype=False)
