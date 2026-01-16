"""Tests for rle_id function."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401


@pytest.mark.functions
def test_rle_id_single_column():
    """Test rle_id with a single column."""
    df = pd.DataFrame(
        {"grp": ["A", "A", "B", "B", "A", "A"], "value": [1, 2, 3, 4, 5, 6]}
    )
    result = df.rle_id("grp")
    expected = df.assign(rle_id=[1, 1, 2, 2, 3, 3])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_multiple_columns():
    """Test rle_id with multiple columns."""
    df = pd.DataFrame(
        {
            "a": [1, 1, 2, 2, 1],
            "b": ["x", "x", "x", "y", "x"],
            "value": [10, 20, 30, 40, 50],
        }
    )
    result = df.rle_id(["a", "b"])
    expected = df.assign(rle_id=[1, 1, 2, 3, 4])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_empty_dataframe():
    """Test rle_id with an empty DataFrame."""
    df = pd.DataFrame({"grp": pd.Series([], dtype=str)})
    result = df.rle_id("grp")
    assert len(result) == 0
    assert "rle_id" in result.columns


@pytest.mark.functions
def test_rle_id_single_row():
    """Test rle_id with a single row DataFrame."""
    df = pd.DataFrame({"grp": ["A"], "value": [1]})
    result = df.rle_id("grp")
    expected = df.assign(rle_id=[1])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_all_same_values():
    """Test rle_id when all values are the same."""
    df = pd.DataFrame({"grp": ["A", "A", "A"], "value": [1, 2, 3]})
    result = df.rle_id("grp")
    expected = df.assign(rle_id=[1, 1, 1])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_all_different_values():
    """Test rle_id when all values are different."""
    df = pd.DataFrame({"grp": ["A", "B", "C"], "value": [1, 2, 3]})
    result = df.rle_id("grp")
    expected = df.assign(rle_id=[1, 2, 3])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_nan_handling():
    """Test rle_id treats consecutive NaN values as the same run."""
    df = pd.DataFrame(
        {"grp": ["A", "A", np.nan, np.nan, "B"], "value": [1, 2, 3, 4, 5]}
    )
    result = df.rle_id("grp")
    expected = df.assign(rle_id=[1, 1, 2, 2, 3])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_custom_column_name():
    """Test rle_id with a custom column name."""
    df = pd.DataFrame({"grp": ["A", "B", "B"]})
    result = df.rle_id("grp", new_column_name="run_id")
    assert "run_id" in result.columns
    assert "rle_id" not in result.columns
    assert result["run_id"].to_list() == [1, 2, 2]


@pytest.mark.functions
def test_rle_id_nonexistent_column():
    """Test rle_id raises ValueError for non-existent column."""
    df = pd.DataFrame({"grp": ["A", "B"]})
    with pytest.raises(ValueError, match="not present in dataframe"):
        df.rle_id("nonexistent")


@pytest.mark.functions
def test_rle_id_column_already_exists():
    """Test rle_id raises ValueError if new_column_name already exists."""
    df = pd.DataFrame({"grp": ["A", "B"], "rle_id": [1, 2]})
    with pytest.raises(ValueError, match="already present in dataframe"):
        df.rle_id("grp")


@pytest.mark.functions
def test_rle_id_numeric_column():
    """Test rle_id with numeric column."""
    df = pd.DataFrame({"grp": [1, 1, 2, 2, 1], "value": [10, 20, 30, 40, 50]})
    result = df.rle_id("grp")
    expected = df.assign(rle_id=[1, 1, 2, 2, 3])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_does_not_mutate():
    """Test that rle_id does not mutate the original DataFrame."""
    df = pd.DataFrame({"grp": ["A", "B", "B"], "value": [1, 2, 3]})
    original_columns = list(df.columns)
    _ = df.rle_id("grp")
    assert list(df.columns) == original_columns


@pytest.mark.functions
def test_rle_id_method_chaining():
    """Test that rle_id works with method chaining."""
    df = pd.DataFrame({"grp": ["A", "A", "B", "B", "A"], "value": [1, 2, 3, 4, 5]})
    result = df.rle_id("grp").groupby(["grp", "rle_id"])["value"].sum()
    expected = pd.Series(
        [3, 5, 7],
        index=pd.MultiIndex.from_tuples(
            [("A", 1), ("A", 3), ("B", 2)], names=["grp", "rle_id"]
        ),
        name="value",
    )
    pd.testing.assert_series_equal(result, expected)


@pytest.mark.functions
def test_rle_id_with_list_of_one_column():
    """Test rle_id with a list containing a single column."""
    df = pd.DataFrame({"grp": ["A", "A", "B"], "value": [1, 2, 3]})
    result = df.rle_id(["grp"])
    expected = df.assign(rle_id=[1, 1, 2])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_multiple_nan_transitions():
    """Test rle_id with multiple transitions involving NaN."""
    df = pd.DataFrame({"grp": ["A", np.nan, "A", np.nan], "value": [1, 2, 3, 4]})
    result = df.rle_id("grp")
    expected = df.assign(rle_id=[1, 2, 3, 4])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_tuple_column_name():
    """Test rle_id with tuple column name (MultiIndex-style)."""
    df = pd.DataFrame({("a", "b"): [1, 1, 2, 2], "value": [10, 20, 30, 40]})
    result = df.rle_id(("a", "b"))
    expected = df.assign(rle_id=[1, 1, 2, 2])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_tuple_as_multiple_columns():
    """Test rle_id with tuple interpreted as multiple columns."""
    df = pd.DataFrame(
        {"a": [1, 1, 2, 2], "b": ["x", "y", "y", "y"], "value": [10, 20, 30, 40]}
    )
    result = df.rle_id(("a", "b"))
    expected = df.assign(rle_id=[1, 2, 3, 3])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_all_nan():
    """Test rle_id with all NaN values starts from 1."""
    df = pd.DataFrame({"grp": [np.nan, np.nan, np.nan]})
    result = df.rle_id("grp")
    expected = df.assign(rle_id=[1, 1, 1])
    assert_frame_equal(result, expected)
