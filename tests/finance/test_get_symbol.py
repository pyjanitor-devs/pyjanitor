import pytest

from janitor.finance import get_symbol

"""
tests the convert_symbol helper function.

Test 1: GME is Gamestop Corp. Test should run fine.
Test 2: GME is not Globius Medical Inc.
Test 3: A little redundant, but it's another
    'happy path' to show get_symbol works for more
    abbreviations than just the one tested so far.
Test 4: ZZZZ does not belong to any company,
    it should therefore it should be None
"""


@pytest.mark.xfail(
    reason="Flaky, because it depends on internet connectivity."
)
def test_get_symbol():
    assert get_symbol("GME") == "GameStop Corp."
    assert get_symbol("GME") != "Globus Medical Inc."
    assert get_symbol("F") == "Ford Motor Company"
    assert get_symbol("ZZZZ") is None
