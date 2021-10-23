import pytest

from janitor.functions.currency_column_to_numeric import (
    _currency_column_to_numeric,
)


@pytest.mark.utils
def test_empty_input():
    assert _currency_column_to_numeric("") == "ORIGINAL_NA"


@pytest.mark.utils
def test_cast_non_numeric_true():
    assert _currency_column_to_numeric("foo", {"foo": 42}) == 42


@pytest.mark.utils
def test_cast_non_numeric_false():
    assert _currency_column_to_numeric("10 dollars", {"foo": 42}) == "10"


@pytest.mark.utils
def test_non_cast_input():
    assert _currency_column_to_numeric("-1,000,000 yen") == "-1000000"
