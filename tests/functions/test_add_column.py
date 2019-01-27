import numpy as np
import pandas as pd
import pytest

from janitor.testing_utils.fixtures import dataframe


def test_add_column(dataframe):
    # col_name wasn't a string
    with pytest.raises(TypeError):
        dataframe.add_column(col_name=42, value=42)

    # column already exists
    with pytest.raises(ValueError):
        dataframe.add_column("a", 42)

    # too many values for dataframe num rows:
    with pytest.raises(ValueError):
        dataframe.add_column("toomany", np.ones(100))

    # functionality testing

    # column appears in DataFrame
    df = dataframe.add_column("fortytwo", 42)
    assert "fortytwo" in df.columns

    # values are correct in dataframe for scalar
    series = pd.Series([42] * len(dataframe))
    series.name = "fortytwo"
    pd.testing.assert_series_equal(df["fortytwo"], series)

    # scalar values are correct for strings
    # also, verify sanity check excludes strings, which have a length:

    df = dataframe.add_column("fortythousand", "test string")
    series = pd.Series(["test string"] * len(dataframe))
    series.name = "fortythousand"
    pd.testing.assert_series_equal(df["fortythousand"], series)

    # values are correct in dataframe for iterable
    vals = np.linspace(0, 43, len(dataframe))
    df = dataframe.add_column("fortythree", vals)
    series = pd.Series(vals)
    series.name = "fortythree"
    pd.testing.assert_series_equal(df["fortythree"], series)

    # fill_remaining works - iterable shorter than DataFrame
    vals = [0, 42]
    target = [0, 42] * 4 + [0]
    df = dataframe.add_column("fill_in_iterable", vals, fill_remaining=True)
    series = pd.Series(target)
    series.name = "fill_in_iterable"
    pd.testing.assert_series_equal(df["fill_in_iterable"], series)

    # fill_remaining works - value is scalar
    vals = 42
    df = dataframe.add_column("fill_in_scalar", vals, fill_remaining=True)
    series = pd.Series([42] * len(df))
    series.name = "fill_in_scalar"
    pd.testing.assert_series_equal(df["fill_in_scalar"], series)


def test_add_column_single_value(dataframe):
    df = dataframe.add_column("city_pop", 100)
    assert df.city_pop.mean() == 100


def test_add_column_iterator_repeat(dataframe):
    df = dataframe.add_column("city_pop", range(3), fill_remaining=True)
    assert df.city_pop.iloc[0] == 0
    assert df.city_pop.iloc[1] == 1
    assert df.city_pop.iloc[2] == 2
    assert df.city_pop.iloc[3] == 0
    assert df.city_pop.iloc[4] == 1
    assert df.city_pop.iloc[5] == 2


def test_add_column_raise_error(dataframe):
    with pytest.raises(Exception):
        dataframe.add_column("cities", 1)


def test_add_column_iterator_repeat_subtraction(dataframe):
    df = dataframe.add_column("city_pop", dataframe.a - dataframe.a)
    assert df.city_pop.sum() == 0
    assert df.city_pop.iloc[0] == 0
