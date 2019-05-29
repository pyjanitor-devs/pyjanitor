import pytest
from janitor.utils import _clean_accounting_column


@pytest.mark.functions
def test_clean_accounting_column():
    test_str = "(1,000)"
    assert _clean_accounting_column(test_str) == float(-1000)


@pytest.mark.functions
def test_clean_accounting_column_zeroes():
    test_str = "()"
    assert _clean_accounting_column(test_str) == 0.00
