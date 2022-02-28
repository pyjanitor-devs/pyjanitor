"""Tests for `currency_column_to_numeric` function."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions.currency_column_to_numeric import (
    _currency_column_to_numeric,
)


@pytest.fixture
def currency_df():
    df = pd.DataFrame(
        {
            "a_col": [" 24.56", "-", "(12.12)", "1,000,000"],
            "d_col": ["", "foo", "1.23 dollars", "-1,000 yen"],
        }
    )
    return df


@pytest.mark.functions
def test_invalid_cleaning_style(currency_df):
    """Ensures a ValueError is thrown if an invalid cleaning style is passed
    in."""
    with pytest.raises(ValueError):
        currency_df.currency_column_to_numeric(
            "a_col", cleaning_style="foobar"
        )


@pytest.mark.functions
def test_accounting_style(currency_df):
    """Checks that the accounting cleaning style is correctly applied."""
    result = currency_df.currency_column_to_numeric(
        "a_col",
        cleaning_style="accounting",
    )
    expected = pd.DataFrame(
        {
            "a_col": [24.56, 0, -12.12, 1_000_000],
            "d_col": ["", "foo", "1.23 dollars", "-1,000 yen"],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_default_cleaning_style(currency_df):
    """Checks that the default cleaning style is correctly applied."""
    result = currency_df.currency_column_to_numeric(
        "d_col",
    )
    expected = pd.DataFrame(
        {
            "a_col": [" 24.56", "-", "(12.12)", "1,000,000"],
            "d_col": [np.nan, np.nan, 1.23, -1_000],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_wrong_type_of_cast_non_numeric_values(currency_df):
    """Checks that a TypeError is raised when the values provided in the
    `cast_non_numeric` dict is not one of acceptable (int/float) type."""
    with pytest.raises(TypeError):
        _ = currency_df.currency_column_to_numeric(
            "d_col",
            cast_non_numeric={"foo": "zzzzz"},
        )


@pytest.mark.functions
def test_default_cleaning_style_with_cast(currency_df):
    """Checks that the cast_non_numeric parameter is correctly applied
    with the default cleaning style."""
    result = currency_df.currency_column_to_numeric(
        "d_col",
        cast_non_numeric={"foo": 999, "non-existent-col": 10},
    )
    expected = pd.DataFrame(
        {
            "a_col": [" 24.56", "-", "(12.12)", "1,000,000"],
            "d_col": [np.nan, 999, 1.23, -1_000],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_wrong_type_of_fill_all_non_numeric(currency_df):
    """Checks a TypeError is raised if a non-int/float is passed into
    fill_all_non_numeric."""
    with pytest.raises(TypeError):
        _ = currency_df.currency_column_to_numeric(
            "d_col",
            fill_all_non_numeric="zzzzz",
        )


@pytest.mark.functions
def test_default_cleaning_style_with_fill(currency_df):
    """Checks that the fill_all_non_numeric parameter is correctly applied
    with the default cleaning style.

    Note empty strings always remain as NaN.
    """
    result = currency_df.currency_column_to_numeric(
        "d_col",
        fill_all_non_numeric=995,
    )
    expected = pd.DataFrame(
        {
            "a_col": [" 24.56", "-", "(12.12)", "1,000,000"],
            "d_col": [np.nan, 995, 1.23, -1_000],
        }
    )
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_default_cleaning_style_with_remove(currency_df):
    """Checks that the remove_non_numeric parameter is correctly applied
    with the default cleaning style.

    Note that originally empty strings are retained, and always remain
    as NaN.
    """
    result = currency_df.currency_column_to_numeric(
        "d_col",
        cast_non_numeric={"non-existent-col": 10},
        remove_non_numeric=True,
    )
    expected = pd.DataFrame(
        {
            "a_col": [" 24.56", "(12.12)", "1,000,000"],
            "d_col": [np.nan, 1.23, -1_000],
        },
        index=[0, 2, 3],
    )
    assert_frame_equal(result, expected)


# Internal API
@pytest.mark.functions
def test_empty_input():
    """Checks empty input is processed properly."""
    assert _currency_column_to_numeric("") == "ORIGINAL_NA"


@pytest.mark.functions
def test_cast_non_numeric_true():
    """Checks behaviour of `cast_non_numeric` dict is correct."""
    assert _currency_column_to_numeric("foo", {"foo": 42}) == 42


@pytest.mark.functions
def test_cast_non_numeric_false():
    """Checks behaviour of `cast_non_numeric` dict is correct."""
    assert _currency_column_to_numeric("10 dollars", {"foo": 42}) == "10"


@pytest.mark.functions
def test_non_cast_input():
    """Checks default cleaning behaviour."""
    assert _currency_column_to_numeric("-1,000,000 yen") == "-1000000"
