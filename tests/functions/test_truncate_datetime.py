from datetime import datetime

import pytest

from janitor.functions.truncate_datetime import _truncate_datetime

"""
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

"""


@pytest.mark.functions
def test__truncate_datetime():
    """Test for _truncate_datetime."""
    x = datetime.now()
    x = _truncate_datetime("month", x)

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
@pytest.mark.functions
def test___truncate_datetime_bad_data():
    """Test for _truncate_datetime with bad data passed in."""
    with pytest.raises(KeyError):
        y = datetime.now()
        y = _truncate_datetime("mon", y)
        assert y is None
