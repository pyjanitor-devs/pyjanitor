import numpy as np
import pandas as pd
import pytest


@pytest.mark.xfail
# pandas 1.1 raises a KeyError if columns/indices passed to loc does not exist
# pandas <1.1 raises a TypeError
def test_filter_date_column_name(date_dataframe):
    df = date_dataframe
    # `column_name` wasn't a string
    with pytest.raises(TypeError):
        df.filter_date(column_name=42)


def test_filter_date_year(date_dataframe):
    df = date_dataframe.filter_date(column_name="DATE", years=[2020])

    def _get_year(x):
        return x.year

    assert df.DATE.apply(_get_year).unique()[0] == 2020


def test_filter_date_years(date_dataframe):
    df = date_dataframe.filter_date(column_name="DATE", years=[2020, 2021])

    def _get_year(x):
        return x.year

    test_result = df.DATE.apply(_get_year).unique()
    expected_result = np.array([2020, 2021])

    assert np.array_equal(test_result, expected_result)


def test_filter_date_month(date_dataframe):
    df = date_dataframe.filter_date(column_name="DATE", months=range(10, 12))

    def _get_month(x):
        return x.month

    test_result = df.DATE.apply(_get_month).unique()
    expected_result = np.array([10, 11])

    assert np.array_equal(test_result, expected_result)


def test_filter_date_start(date_dataframe):
    start_date = "02/01/19"

    df = date_dataframe.filter_date(column_name="DATE", start_date=start_date)

    test_date = pd.to_datetime("01/31/19")
    test_result = df[df.DATE <= test_date]

    assert test_result.empty


def test_filter_date_start_and_end(date_dataframe):
    start_date = "02/01/19"
    end_date = "02/02/19"

    df = date_dataframe.filter_date(
        column_name="DATE", start_date=start_date, end_date=end_date
    )

    assert df.shape[0] == 2


def test_filter_different_date_format(date_dataframe):
    end_date = "01@@@@29@@@@19"
    format = "%m@@@@%d@@@@%y"
    df = date_dataframe.filter_date(
        column_name="DATE", end_date=end_date, format=format
    )

    assert df.shape[0] == 2


def test_column_date_options(date_dataframe):
    end_date = "01/29/19"
    column_date_options = {"dayfirst": True}

    # Parse the dates with the first value as the day. For our purposes this
    # basically is taking the month and turning it into a "day". It's
    # incorrect, but used here to see that the column_date_options parameter
    # can be passed through to the filter_date function.

    df = date_dataframe.filter_date(
        column_name="DATE",
        end_date=end_date,
        column_date_options=column_date_options,
    )
    assert df.shape[0] == 13
