from datetime import datetime
from janitor import trunc_datetime
import pytest


@pytest.mark.functions
def test_trunc_datetime():
    x = datetime.now()
    x = trunc_datetime("month", x)
    y = datetime.now()
    y = trunc_datetime("mon", x)
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

    #bad data, error handling test
    assert y.month == None
