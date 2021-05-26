from datetime import datetime
from janitor import truncate_datetime
import pytest


'''
Creates 2 datetime objects, one
    will be a valid object and the
    other will be Nonetype.

Test 1 asserts valid object is truncated
    correctly, this is trivial on the
    first day of every month.

Test 2 asserts that the time[] value
    is accurate to the specified precision,
    in this case, month.

Test 3 asserts the valid object truncated
    correctly to the specified precision
    in this case, month.

Test 4 asserts that if bad data is passed
    it will return a Nonetype, and
    appropriate error handling will
    take care of it.

'''


@pytest.mark.functions
def test_trunc_datetime():
    x = datetime.now()
    x = truncate_datetime("month", x)
    y = datetime.now()
    y = truncate_datetime("mon", x)
    time = {
        "Year": [x.year],
        "Month": [x.month],
        "Day": [x.day],
        "Hour": [x.hour],
        "Minute": [x.minute],
        "Second": [x.second],
    }

    # time[] returns datetime object, needs indexing.
    assert time["Day"][0] == 1
    assert time["Month"][0] == datetime.now().month
    assert time["Month"][0] == x.month

    # bad data, error handling test
    assert y is None
