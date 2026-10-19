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
