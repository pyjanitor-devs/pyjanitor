"""Tests for `currency_column_to_numeric` function."""
import pytest

from janitor.functions.currency_column_to_numeric import (
    _currency_column_to_numeric,
)


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
