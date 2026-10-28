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


@pytest.mark.functions
def test_rle_id_groupby_single_column():
    """Test rle_id on GroupBy object restarts IDs within each group."""
    df = pd.DataFrame(
        {
            "group": ["X", "X", "X", "Y", "Y", "Y"],
            "grp": ["A", "A", "B", "A", "B", "B"],
            "value": [1, 2, 3, 4, 5, 6],
        }
    )
    result = df.groupby("group").rle_id("grp")
    expected = df.assign(rle_id=[1, 1, 2, 1, 2, 2])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_multiple_group_columns():
    """Test rle_id with multiple grouping columns."""
    df = pd.DataFrame(
        {
            "g1": ["A", "A", "A", "A", "B", "B"],
            "g2": ["X", "X", "Y", "Y", "X", "X"],
            "val": [1, 1, 2, 2, 3, 4],
            "data": [10, 20, 30, 40, 50, 60],
        }
    )
    result = df.groupby(["g1", "g2"]).rle_id("val")
    expected = df.assign(rle_id=[1, 1, 1, 1, 1, 2])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_single_group():
    """Test rle_id with only one group."""
    df = pd.DataFrame({"g": ["X", "X"], "v": ["A", "B"]})
    result = df.groupby("g").rle_id("v")
    expected = df.assign(rle_id=[1, 2])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_reused():
    """New column is visible on a groupby already used for other methods."""
    df = pd.DataFrame(
        {"group": ["X", "X", "Y", "Y"], "grp": ["A", "B", "A", "A"], "v": [1, 2, 3, 4]}
    )
    grouped = df.groupby("group")
    grouped.sum()
    grouped.get_columns("*")
    result = grouped.rle_id("grp")
    expected = df.assign(rle_id=[1, 2, 1, 1])
    assert_frame_equal(result.get_columns("*"), expected)
    assert_frame_equal(result.sum(), expected.groupby("group").sum())


@pytest.mark.functions
def test_rle_id_groupby_chained():
    """Chained rle_id calls keep earlier new columns."""
    df = pd.DataFrame(
        {"group": ["X", "X", "Y", "Y"], "grp": ["A", "B", "A", "A"], "v": [1, 1, 2, 3]}
    )
    result = (
        df.groupby("group")
        .rle_id("grp")
        .rle_id("v", new_column_name="rle_v")
        .get_columns("*")
    )
    expected = df.assign(rle_id=[1, 2, 1, 1], rle_v=[1, 1, 1, 2])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_groupby_after_mutate():
    """Accessors cached before the deprecated mutate do not leak through."""
    df = pd.DataFrame(
        {"group": ["X", "X", "Y", "Y"], "grp": ["A", "B", "A", "A"], "v": [1, 2, 3, 4]}
    )
    grouped = df.groupby("group")
    grouped.get_columns("*")
    with pytest.warns(FutureWarning):
        mutated = grouped.mutate({"v_sum": ("v", "sum")})
    result = mutated.rle_id("grp").get_columns("*")
    expected = df.assign(v_sum=[3, 3, 7, 7], rle_id=[1, 2, 1, 1])
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_rle_id_groupby_level():
    """Grouping by index level restarts IDs within each group."""
    df = pd.DataFrame({"k": [1, 1, 1, 2, 2, 2], "v": [1, 1, 2, 3, 3, 3]})
    df = df.set_index("k")
    result = df.groupby(level=0).rle_id("v")
    expected = df.assign(rle_id=[1, 1, 2, 1, 1, 1])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_series_key():
    """Grouping by an external Series restarts IDs within each group."""
    df = pd.DataFrame({"v": [1, 1, 2, 3, 3, 3]})
    result = df.groupby(pd.Series([1, 1, 1, 2, 2, 2])).rle_id("v")
    expected = df.assign(rle_id=[1, 1, 2, 1, 1, 1])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_column_selection():
    """A column selection on the groupby does not limit `column_names`."""
    df = pd.DataFrame({"k": [1, 1, 2], "v": [1, 2, 2], "w": [1, 1, 1]})
    result = df.groupby("k")[["v"]].rle_id("w")
    expected = df.assign(rle_id=[1, 1, 1])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_dropna_false():
    """With dropna=False, missing keys form their own group."""
    df = pd.DataFrame({"k": [1.0, np.nan, 1.0, np.nan], "v": [1, 1, 2, 2]})
    result = df.groupby("k", dropna=False).rle_id("v")
    expected = df.assign(rle_id=[1, 1, 2, 2])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_dropna_true():
    """With dropna=True, rows with missing keys get a missing ID."""
    df = pd.DataFrame({"k": [1.0, np.nan, 1.0], "v": [1, 1, 2]})
    result = df.groupby("k").rle_id("v")
    expected = df.assign(rle_id=[1.0, np.nan, 2.0])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_duplicate_index():
    """IDs are placed by position, not by (duplicate) index labels."""
    df = pd.DataFrame(
        {"k": [1, 2, 1, 2, 1], "v": [1, 1, 2, 1, 2]}, index=[0, 0, 1, 1, 0]
    )
    result = df.groupby("k").rle_id("v")
    expected = df.assign(rle_id=[1, 1, 2, 1, 2])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
@pytest.mark.parametrize("n", [1, 2])
def test_rle_id_groupby_equal_size_groups(n):
    """Multiple keys with equally sized groups (including one row) work."""
    df = pd.DataFrame({"k": [1, 2] * n, "k2": [0, 0] * n, "v": list(range(2 * n))})
    result = df.groupby(["k", "k2"]).rle_id("v")
    expected = df.assign(rle_id=[i // 2 + 1 for i in range(2 * n)])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_single_row_multiple_keys():
    """A single row grouped by multiple keys gets ID 1."""
    df = pd.DataFrame({"k": [1.0], "k2": [0], "v": [2.0]})
    result = df.groupby(["k", "k2"]).rle_id("v")
    expected = df.assign(rle_id=[1])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_empty():
    """An empty frame gets an empty integer ID column."""
    df = pd.DataFrame({"k": pd.Series([], dtype=int), "v": pd.Series([], dtype=int)})
    result = df.groupby("k").rle_id("v")
    expected = df.assign(rle_id=pd.Series([], dtype="int64"))
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
def test_rle_id_groupby_grouper_freq_unsorted():
    """pd.Grouper(freq=...) sorts by time; IDs follow that order."""
    df = pd.DataFrame(
        {
            "t": pd.to_datetime(
                [
                    "2026-01-02 06:00",
                    "2026-01-01 00:00",
                    "2026-01-02 00:00",
                    "2026-01-01 12:00",
                    "2026-01-01 06:00",
                ]
            ),
            "v": [1, 1, 2, 2, 1],
        }
    )
    result = df.groupby(pd.Grouper(key="t", freq="D")).rle_id("v")
    expected = df.loc[[1, 4, 3, 2, 0]].assign(rle_id=[1, 1, 2, 1, 2])
    assert_frame_equal(result.obj, expected)


@pytest.mark.functions
@pytest.mark.parametrize("dtype", ["string[pyarrow]", "string[python]", "str"])
def test_rle_id_string_dtypes(dtype):
    """String columns, including pyarrow-backed ones, give int64 IDs."""
    pytest.importorskip("pyarrow")
    df = pd.DataFrame(
        {"k": [1, 1, 2, 2], "w": pd.array(["x", "x", "y", None], dtype=dtype)}
    )
    assert_frame_equal(df.rle_id("w"), df.assign(rle_id=[1, 1, 2, 3]))
    result = df.groupby("k").rle_id("w")
    assert_frame_equal(result.obj, df.assign(rle_id=[1, 1, 1, 2]))


@pytest.mark.functions
def test_rle_id_nullable_values():
    """Nullable integer columns give int64 IDs."""
    df = pd.DataFrame(
        {"k": [1, 1, 2, 2], "w": pd.array([1, 1, 2, None], dtype="Int64")}
    )
    assert_frame_equal(df.rle_id("w"), df.assign(rle_id=[1, 1, 2, 3]))
    result = df.groupby("k").rle_id("w")
    assert_frame_equal(result.obj, df.assign(rle_id=[1, 1, 1, 2]))


@pytest.mark.functions
def test_rle_id_groupby_nullable_key_dropped():
    """Rows with a missing nullable key get a missing (float64) ID."""
    df = pd.DataFrame(
        {"k": pd.array([1, None, 1, 2], dtype="Int64"), "w": [1, 1, 2, 3]}
    )
    result = df.groupby("k").rle_id("w")
    expected = df.assign(rle_id=[1.0, np.nan, 2.0, 1.0])
    assert_frame_equal(result.obj, expected)
