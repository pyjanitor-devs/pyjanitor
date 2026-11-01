"""Tests for drop_constant_columns."""

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_drop_constant_columns(df_constant_columns):
    """Test that executes drop_constant_columns function."""
    processed_df = df_constant_columns.drop_constant_columns()
    expected_col_list = ["Bell__Chart", "decorated-elephant", "cities"]
    assert processed_df.columns.to_list() == expected_col_list
    data = {
        "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
        "decorated-elephant": [1, 2, 3] * 3,
        "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    }
    expected_df = pd.DataFrame(data)
    assert_frame_equal(processed_df, expected_df)


@pytest.mark.functions
def test_drop_constant_columns_with_missing_values():
    """Retain partial missingness and drop entirely missing columns."""
    df = pd.DataFrame(
        {
            "constant": [1, 1, 1],
            "value_with_missing": [5.0, float("nan"), float("nan")],
            "all_missing": [float("nan")] * 3,
            "varying": [1, 2, 3],
        }
    )
    original = df.copy(deep=True)
    expected = df[["value_with_missing", "varying"]]

    assert_frame_equal(df.drop_constant_columns(), expected)
    assert_frame_equal(df, original)


@pytest.mark.functions
@pytest.mark.parametrize(
    "dtype, values",
    [
        ("Int64", [5, pd.NA, pd.NA]),
        ("string", ["a", pd.NA, pd.NA]),
        ("category", ["a", None, None]),
    ],
)
def test_drop_constant_columns_nullable_dtypes(dtype, values):
    """Keep missing plus one value for pandas nullable dtypes."""
    df = pd.DataFrame(
        {
            "value_with_missing": pd.Series(values, dtype=dtype),
            "all_missing": pd.Series([pd.NA] * 3, dtype=dtype),
        }
    )
    assert_frame_equal(df.drop_constant_columns(), df[["value_with_missing"]])


@pytest.mark.functions
def test_drop_constant_columns_mixed_missing_sentinels():
    """Different object-dtype null markers count as one missing value."""
    df = pd.DataFrame(
        {
            "all_missing": pd.Series([None, float("nan"), pd.NA], dtype=object),
            "value_with_missing": pd.Series(
                ["A", None, float("nan")], dtype=object
            ),
            "constant": [1, 1, 1],
            "varying": [1, 2, 3],
        }
    )
    original = df.copy(deep=True)

    assert_frame_equal(
        df.drop_constant_columns(), df[["value_with_missing", "varying"]]
    )
    assert_frame_equal(df, original)


@pytest.mark.functions
def test_drop_constant_columns_empty_dataframe():
    """Keep columns if no rows are available to establish constants."""
    df = pd.DataFrame(columns=["a", "b"])

    assert_frame_equal(df.drop_constant_columns(), df)


@pytest.mark.functions
@pytest.mark.parametrize(
    "values, retain_object_column",
    [
        ([None, float("nan"), pd.NA], False),
        ([pd.NaT, None, float("nan")], False),
        ([None, None, None], False),
        ([1, None, pd.NA], True),
        (["a", float("nan"), None], True),
        ([1, 2, pd.NA], True),
    ],
)
def test_drop_constant_columns_object_null_semantics(values, retain_object_column):
    """Treat every object-dtype null as one level without merging real values."""
    df = pd.DataFrame(
        {
            "object_column": pd.Series(values, dtype=object),
            "varying": [1, 2, 3],
        }
    )
    original = df.copy(deep=True)
    expected = df if retain_object_column else df[["varying"]]

    assert_frame_equal(df.drop_constant_columns(), expected)
    assert_frame_equal(df, original)

