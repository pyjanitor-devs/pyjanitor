import datetime as dt

import janitor as jn
import pytest


@pytest.mark.functions
def test_convert_to_date_list():
    dates = ["2009-07-06", "40000", "40000.1", None]

    result = jn.convert_to_date(dates)

    expected = [
        dt.date(2009, 7, 6),
        dt.date(2009, 7, 6),
        dt.date(2009, 7, 6),
        None,
    ]
    assert result == expected


@pytest.mark.functions
def test_convert_to_date_string_failure_error():
    with pytest.raises(ValueError, match="Not all character strings converted"):
        jn.convert_to_date(["not-a-date"])


@pytest.mark.functions
def test_convert_to_date_string_failure_warning():
    with pytest.warns(UserWarning, match="Not all character strings converted"):
        result = jn.convert_to_date(
            ["not-a-date"], string_conversion_failure="warning"
        )
    assert result == [None]


@pytest.mark.functions
def test_convert_to_datetime_list():
    datetimes = ["2009-07-06", "40000.1", "40000", None]

    result = jn.convert_to_datetime(datetimes, tz=None)

    expected = [
        dt.datetime(2009, 7, 6, 0, 0),
        dt.datetime(2009, 7, 6, 2, 24),
        dt.datetime(2009, 7, 6, 0, 0),
        None,
    ]
    assert result == expected


@pytest.mark.functions
def test_convert_to_datetime_time_strings():
    result = jn.convert_to_datetime(["12:30 PM", "14:30:15", "0.5"], tz=None)

    expected = [
        dt.datetime(1899, 12, 30, 12, 30),
        dt.datetime(1899, 12, 30, 14, 30, 15),
        dt.datetime(1899, 12, 30, 12, 0),
    ]
    assert result == expected


@pytest.mark.functions
def test_convert_to_datetime_timezone():
    result = jn.convert_to_datetime(["2009-07-06"], tz="UTC")
    assert result[0].tzinfo is not None
    assert result[0].utcoffset() == dt.timedelta(0)


@pytest.mark.functions
def test_excel_time_to_numeric_examples():
    assert jn.excel_time_to_numeric("0.5") == 43200
    assert jn.excel_time_to_numeric("12:30 PM") == 45000
    assert jn.excel_time_to_numeric("14:30:15") == 52215


@pytest.mark.functions
def test_excel_time_to_numeric_invalid_string():
    with pytest.raises(ValueError, match="did not match an interpretable"):
        jn.excel_time_to_numeric("not a time")


@pytest.mark.functions
def test_sas_numeric_to_date_examples():
    assert jn.sas_numeric_to_date(date_num=15639) == dt.date(2002, 10, 26)

    datetime_value = jn.sas_numeric_to_date(datetime_num=1217083532, tz="UTC")
    assert datetime_value.utcoffset() == dt.timedelta(0)
    assert datetime_value.replace(tzinfo=None) == dt.datetime(
        1998, 7, 26, 14, 45, 32
    )

    assert jn.sas_numeric_to_date(time_num=3600) == dt.time(1, 0, 0)

    combined = jn.sas_numeric_to_date(date_num=15639, time_num=3600, tz="UTC")
    assert combined.replace(tzinfo=None) == dt.datetime(2002, 10, 26, 1, 0, 0)


@pytest.mark.functions
def test_sas_numeric_to_date_invalid_combinations():
    with pytest.raises(ValueError, match="Must not give both"):
        jn.sas_numeric_to_date(date_num=1, datetime_num=2)

    with pytest.raises(ValueError, match="same values are not NA"):
        jn.sas_numeric_to_date(date_num=[1, None], time_num=[None, 1])
