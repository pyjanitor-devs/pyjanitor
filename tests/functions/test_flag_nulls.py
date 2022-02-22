"""Tests for `flag_nulls` function."""
import pytest
import pandas as pd
from pandas.testing import assert_frame_equal

from janitor.functions import flag_nulls


@pytest.mark.functions
def test_functional_on_all_columns(missingdata_df):
    """Checks `flag_nulls` behaviour on default (all) columns."""
    expected = missingdata_df.copy()
    expected["null_flag"] = [0, 1, 1] * 3

    df = missingdata_df.flag_nulls()

    assert_frame_equal(df, expected, check_dtype=False)

    # Should also be the same for explicit columns
    df = missingdata_df.flag_nulls(columns=["a", "Bell__Chart"])

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_non_method_functional(missingdata_df):
    """Checks the behaviour when `flag_nulls` is used as a function."""
    expected = missingdata_df.copy()
    expected["null_flag"] = [0, 1, 1] * 3

    df = flag_nulls(missingdata_df)

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_functional_on_some_columns(missingdata_df):
    """Checks the columns parameter when used as a method call."""
    expected = missingdata_df.copy()
    expected["null_flag"] = [0, 0, 1] * 3

    df = missingdata_df.flag_nulls(columns=["a"])

    assert_frame_equal(df, expected, check_dtype=False)

    # Testing when we provide the direct name
    df = missingdata_df.flag_nulls(columns="a")

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_columns_generic_hashable():
    """Checks flag_nulls behaviour when columns is a generic hashable."""
    df = pd.DataFrame(
        {
            25: ["w", "x", None, "z"],
            ("weird", "col"): [5, None, 7, 8],
            "normal": [1, 2, 3, 4],
        }
    )
    expected = df.copy().assign(null_flag=[0, 0, 1, 0])

    df = df.flag_nulls(columns=25)
    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_rename_output_column(missingdata_df):
    """Checks that output column is renamed properly when
    `column_name` is specified explicitly."""
    expected = missingdata_df.copy()
    expected["flag"] = [0, 1, 1] * 3

    df = missingdata_df.flag_nulls(column_name="flag")

    assert_frame_equal(df, expected, check_dtype=False)


@pytest.mark.functions
def test_fail_column_name_in_columns(missingdata_df):
    """Checks that the output `column_name` is not already in df
    columns.
    """
    with pytest.raises(ValueError):
        missingdata_df.flag_nulls(column_name="a")


@pytest.mark.functions
def test_fail_column_val_not_in_columns(missingdata_df):
    """Checks that ValueError is raised when specified `columns`
    is not present in the df columns.
    """
    with pytest.raises(ValueError):
        missingdata_df.flag_nulls(columns=["c"])
