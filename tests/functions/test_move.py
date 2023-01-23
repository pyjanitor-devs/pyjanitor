import pytest
import numpy as np
import pandas as pd

from pandas.testing import assert_series_equal, assert_index_equal
from hypothesis import given
from hypothesis import settings

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
def test_move_row(dataframe):
    """
    Test function move() for rows with defaults.
    Case with row labels being integers.
    """
    # Setup
    source = 1
    target = 3
    row = dataframe.loc[source, :]

    # Exercise
    result = dataframe.move(source=source, target=target, axis=0)

    # Verify
    assert_series_equal(result.iloc[target - 1, :], row)


@pytest.mark.functions
def test_move_row_after(dataframe):
    """
    Test function move() for rows with position = 'after'.
    Case with row labels being integers.
    """
    # Setup
    source = 1
    target = 3
    row = dataframe.loc[source, :]

    # Exercise
    result = dataframe.move(
        source=source, target=target, position="after", axis=0
    )

    # Verify
    assert_series_equal(result.iloc[target, :], row)


@pytest.mark.functions
def test_move_row_strings(dataframe):
    """
    Test function move() for rows with defaults.
    Case with row labels being strings.
    """
    # Setup
    dataframe = dataframe.set_index("animals@#$%^").drop_duplicates()
    rows = dataframe.index
    source_index = 1
    target_index = 2
    source = rows[source_index]
    target = rows[target_index]
    row = dataframe.loc[source, :]

    # Exercise
    result = dataframe.move(source=source, target=target, axis=0)

    # Verify
    assert_series_equal(result.iloc[target_index - 1, :], row)


@pytest.mark.functions
def test_move_row_after_strings(dataframe):
    """
    Test function move() for rows with position = 'after'.
    Case with row labels being strings.
    """
    # Setup
    dataframe = dataframe.set_index("animals@#$%^").drop_duplicates()
    rows = dataframe.index
    source_index = 1
    target_index = 2
    source = rows[source_index]
    target = rows[target_index]
    row = dataframe.loc[source, :]

    # Exercise
    result = dataframe.move(
        source=source, target=target, position="after", axis=0
    )

    # Verify
    assert_series_equal(result.iloc[target_index, :], row)


@pytest.mark.functions
def test_move_col(dataframe):
    """
    Test function move() for columns with defaults.
    """
    # Setup
    columns = dataframe.columns
    source_index = 1
    target_index = 3
    source = columns[source_index]
    target = columns[target_index]
    col = dataframe[source]

    # Exercise
    result = dataframe.move(source=source, target=target, axis=1)

    # Verify
    assert_series_equal(result.iloc[:, target_index - 1], col)


@pytest.mark.functions
def test_move_col_after(dataframe):
    """
    Test function move() for columns with position = 'after'.
    """
    # Setup
    columns = dataframe.columns
    source_index = 1
    target_index = 3
    source = columns[source_index]
    target = columns[target_index]
    col = dataframe[source]

    # Exercise
    result = dataframe.move(
        source=source, target=target, position="after", axis=1
    )

    # Verify
    assert_series_equal(result.iloc[:, target_index], col)


@pytest.mark.functions
def test_move_invalid_args(dataframe):
    """Checks appropriate errors are raised with invalid args."""
    with pytest.raises(ValueError):
        # invalid position
        _ = dataframe.move("a", "cities", position="oops", axis=1)
    with pytest.raises(ValueError):
        # invalid axis
        _ = dataframe.move("a", "cities", axis="oops")
    with pytest.raises(KeyError):
        # invalid source row
        _ = dataframe.move(10_000, 0, axis=0)
    with pytest.raises(KeyError):
        # invalid target row
        _ = dataframe.move(0, 10_000, axis=0)
    with pytest.raises(KeyError):
        # invalid source column
        _ = dataframe.move("__oops__", "cities", axis=1)
    with pytest.raises(KeyError):
        # invalid target column
        _ = dataframe.move("a", "__oops__", axis=1)


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_move_reorder_columns(df):
    """Replicate reorder_columns"""
    assert all(
        df.move(source=df.columns, axis=1, position="after").columns
        == df.columns
    )

    assert all(df.move(source=df.index, position="before").index == df.index)

    assert all(
        df.move(source=["animals@#$%^", "Bell__Chart"], axis=1).columns
        == ["animals@#$%^", "Bell__Chart", "a", "decorated-elephant", "cities"]
    )


def test_move_unique():
    """Raise if the axis is not unique"""
    df = pd.DataFrame({"a": [2, 4, 6], "b": [1, 3, 5], "c": [7, 8, 9]})
    df.columns = ["a", "b", "b"]
    with pytest.raises(AssertionError):
        df.move(source="a", axis=1)


def test_move_multiindex():
    """Raise if the axis is a MultiIndex"""
    df = pd.DataFrame(
        {
            ("name", "a"): {0: "Wilbur", 1: "Petunia", 2: "Gregory"},
            ("names", "aa"): {0: 67, 1: 80, 2: 64},
            ("more_names", "aaa"): {0: 56, 1: 90, 2: 50},
        }
    )
    with pytest.raises(AssertionError):
        df.move(source="a", axis=1)


np.random.seed(9)
df = pd.DataFrame(np.random.random(size=(5, 10)), columns=list("abcdefghij"))


def test_move_source_target_seq():
    """Test output when both source and targets are sequences"""
    expected = df.move(source=["j", "a"], target=["c", "e"], axis=1).columns
    actual = pd.Index(
        ["b", "j", "a", "c", "d", "e", "f", "g", "h", "i"], dtype="object"
    )
    assert_index_equal(expected, actual)


def test_move_source_target_seq_after():
    """Test output when both source and targets are sequences"""
    expected = df.move(
        source=["j", "a"], target=["c", "e"], position="after", axis=1
    ).columns
    actual = pd.Index(
        ["b", "c", "d", "e", "j", "a", "f", "g", "h", "i"], dtype="object"
    )
    assert_index_equal(expected, actual)
