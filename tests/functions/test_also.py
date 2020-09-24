"""Unit tests for `.also()`."""
from unittest.mock import Mock

import pytest


def remove_first_two_letters_from_col_names(df):
    """Helper function to mutate dataframe by changing column names."""
    col_names = df.columns
    col_names = [name[2:] for name in col_names]
    df.columns = col_names
    return df


def remove_rows_3_and_4(df):
    """Helper function to mutate dataframe by removing rows."""
    df = df.drop(3, axis=0)
    df = df.drop(4, axis=0)
    return df


def drop_inplace(df):
    """
    Helper function to mutate dataframe by dropping a column.

    We usually would not use `inplace=True` in a block,
    but the intent here is to test that
    the in-place modification of a dataframe
    doesn't get passed through in the `.also()` function.
    Hence, we tell Flake8 to skip checking `PD002` on that line.

    .. # noqa: DAR101
    """
    df.drop(columns=[df.columns[0]], inplace=True)  # noqa: PD002


@pytest.mark.functions
def test_also_column_manipulation_no_change(dataframe):
    """Test that changed dataframe inside `.also()` doesn't get returned."""
    cols = tuple(dataframe.columns)
    df = dataframe.also(remove_first_two_letters_from_col_names)
    assert dataframe is df
    assert cols == tuple(df.columns)


@pytest.mark.functions
def test_also_remove_rows_no_change(dataframe):
    """Test that changed dataframe inside `.also()` doesn't get returned."""
    df = dataframe.also(remove_rows_3_and_4)
    rows = tuple(df.index)
    assert rows == (0, 1, 2, 3, 4, 5, 6, 7, 8)


@pytest.mark.functions
def test_also_runs_function(dataframe):
    """Test that `.also()` executes the function."""
    method = Mock(return_value=None)
    df = dataframe.also(method)
    assert id(df) == id(dataframe)
    assert method.call_count == 1


@pytest.mark.functions
def test_also_args(dataframe):
    """Test that the args are passed through to the function."""
    method = Mock(return_value=None)
    _ = dataframe.also(method, 5)

    assert method.call_args[0][1] == 5


@pytest.mark.functions
def test_also_kwargs(dataframe):
    """Test that the kwargs are passed through to the function."""
    method = Mock(return_value=None)
    _ = dataframe.also(method, n=5)

    assert method.call_args[1] == {"n": 5}


@pytest.mark.functions
def test_also_drop_inplace(dataframe):
    """Test that in-place modification of dataframe does not pass through."""
    cols = tuple(dataframe.columns)
    df = dataframe.also(drop_inplace)
    assert tuple(df.columns) == cols
