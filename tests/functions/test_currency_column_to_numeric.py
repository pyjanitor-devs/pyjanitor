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
        currency_df.currency_column_to_numeric("a_col", cleaning_style="foobar")


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
def test_accounting_style_with_dollar_sign():
    """Test accounting style handles $ currency symbol."""
    df = pd.DataFrame({"amount": ["$24.56", "($12.12)", "$1,000,000", "-"]})
    result = df.currency_column_to_numeric("amount", cleaning_style="accounting")
    expected = pd.DataFrame({"amount": [24.56, -12.12, 1_000_000.0, 0.0]})
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_accounting_style_with_various_currency_symbols():
    """Test accounting style handles various currency symbols.

    Tests €, £, ¥, ₹, ₩ symbols.
    """
    df = pd.DataFrame(
        {"amount": ["€100.50", "£200.75", "¥300", "(€50.25)", "₹1,000", "₩500"]}
    )
    result = df.currency_column_to_numeric("amount", cleaning_style="accounting")
    expected = pd.DataFrame({"amount": [100.50, 200.75, 300.0, -50.25, 1000.0, 500.0]})
    assert_frame_equal(result, expected)


@pytest.mark.functions
def test_accounting_style_with_additional_currency_symbols():
    """Test accounting style handles additional currency symbols.

    Tests ₽ (Russian Ruble), ₱ (Philippine Peso), ฿ (Thai Baht), ₺ (Turkish Lira).
    """
    df = pd.DataFrame({"amount": ["₽1,500", "(₱250.75)", "฿3,000", "₺750.50"]})
    result = df.currency_column_to_numeric("amount", cleaning_style="accounting")
    expected = pd.DataFrame({"amount": [1500.0, -250.75, 3000.0, 750.50]})
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


@pytest.mark.functions
@pytest.mark.parametrize("cleaning_style", [None, "accounting"])
def test_currency_column_to_numeric_is_idempotent(cleaning_style):
    """Calling the conversion twice matches calling it once.

    ``janitor.utils.idempotent`` does not fit this check. It evaluates
    ``not (left == right)``, and DataFrame equality is a boolean frame, so
    that expression raises.
    """
    frame = pd.DataFrame(
        {
            "amount": ["$24.56", "($12.12)", "$1,000,000", "1.50"],
            "label": list("abcd"),
        }
    )

    def convert(data):
        return data.currency_column_to_numeric(
            "amount",
            cleaning_style=cleaning_style,
        )

    assert_frame_equal(convert(convert(frame)), convert(frame))


@pytest.mark.functions
@pytest.mark.parametrize(
    "kwargs",
    [
        {},
        {"cast_non_numeric": {"foo": 0}},
        {"fill_all_non_numeric": 0},
        {"remove_non_numeric": True},
    ],
)
def test_default_currency_cleaning_is_idempotent(kwargs):
    """Default cleaning, including its optional arguments, can be re-run."""
    frame = pd.DataFrame(
        {
            "amount": ["", "foo", "1.23 dollars", "-1,000 yen"],
            "label": list("abcd"),
        }
    )

    def convert(data):
        return data.currency_column_to_numeric("amount", **kwargs)

    assert_frame_equal(convert(convert(frame)), convert(frame))


@pytest.mark.functions
@pytest.mark.parametrize("cleaning_style", [None, "accounting"])
@pytest.mark.parametrize(
    ("values", "dtype"),
    [
        ([1, 0, -3], "int64"),
        ([1.5, np.nan, -2.0], "float64"),
        ([1, pd.NA, 3], "Int64"),
        ([1.5, pd.NA, -2.0], "Float64"),
    ],
)
def test_numeric_column_passes_through_unchanged(cleaning_style, values, dtype):
    """An already-numeric column is returned unchanged."""
    original = pd.DataFrame(
        {
            "amount": pd.Series(values, dtype=dtype),
            "label": ["a", "b", "c"],
        }
    )
    untouched = original.copy()

    result = original.currency_column_to_numeric(
        "amount",
        cleaning_style=cleaning_style,
    )

    assert_frame_equal(result, untouched)
    assert_frame_equal(original, untouched)
