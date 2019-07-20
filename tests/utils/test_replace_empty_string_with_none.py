import pytest

from janitor.utils import (
    _replace_empty_string_with_none,
    _replace_original_empty_string_with_none,
)


@pytest.mark.function
def test_int_input():
    assert _replace_empty_string_with_none(4) == 4


@pytest.mark.function
def test_float_input():
    assert _replace_empty_string_with_none(4.0) == 4.0


@pytest.mark.function
def test_empty_string_input():
    assert _replace_empty_string_with_none("") is None


@pytest.mark.function
def test_replace_original_empty_string_with_none():
    assert _replace_original_empty_string_with_none("foo") == "foo"
    assert _replace_original_empty_string_with_none("ORIGINAL_NA") is None
