import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import strategies as st

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(df=df_strategy())
def test_add_column_add_integer(df):
    # col_name wasn't a string
    with pytest.raises(TypeError):
        df.add_column(col_name=42, value=42)


@pytest.mark.functions
@given(df=df_strategy())
def test_add_column_already_exists(df):
    # column already exists
    with pytest.raises(ValueError):
        df.add_column("a", 42)


@pytest.mark.functions
@given(df=df_strategy())
def test_add_column_too_many(df):
    # too many values for dataframe num rows:
    with pytest.raises(ValueError):
        df.add_column("toomany", np.ones(100))


@pytest.mark.functions
@given(df=df_strategy())
def test_add_column_scalar(df):
    # column appears in DataFrame
    df = df.add_column("fortytwo", 42)
    assert "fortytwo" in df.columns

    # values are correct in dataframe for scalar
    series = pd.Series([42] * len(df))
    series.name = "fortytwo"
    pd.testing.assert_series_equal(df["fortytwo"], series)

    # scalar values are correct for strings
    # also, verify sanity check excludes strings, which have a length:


@pytest.mark.functions
@given(df=df_strategy())
def test_add_column_string(df):
    df = df.add_column("fortythousand", "test string")
    series = pd.Series(["test string"] * len(df))
    series.name = "fortythousand"
    pd.testing.assert_series_equal(df["fortythousand"], series)

    # values are correct in dataframe for iterable
    vals = np.linspace(0, 43, len(df))
    df = df.add_column("fortythree", vals)
    series = pd.Series(vals)
    series.name = "fortythree"
    pd.testing.assert_series_equal(df["fortythree"], series)


@pytest.mark.functions
@given(df=df_strategy(), vals=st.lists(elements=st.integers()))
def test_add_column_fill_remaining_iterable(df, vals):
    if len(vals) > len(df) or len(vals) == 0:
        with pytest.raises(ValueError):
            df = df.add_column("fill_in_iterable", vals, fill_remaining=True)
    else:
        df = df.add_column("fill_in_iterable", vals, fill_remaining=True)
        assert not pd.isnull(df["fill_in_iterable"]).any()


@pytest.mark.functions
@given(df=df_strategy())
def test_add_column_fill_scalar(df):
    # fill_remaining works - value is scalar
    vals = 42
    df = df.add_column("fill_in_scalar", vals, fill_remaining=True)
    series = pd.Series([42] * len(df))
    series.name = "fill_in_scalar"
    pd.testing.assert_series_equal(df["fill_in_scalar"], series)


@pytest.mark.functions
def test_add_column_single_value(dataframe):
    df = dataframe.add_column("city_pop", 100)
    assert df.city_pop.mean() == 100


@pytest.mark.functions
def test_add_column_iterator_repeat(dataframe):
    df = dataframe.add_column("city_pop", range(3), fill_remaining=True)
    assert df.city_pop.iloc[0] == 0
    assert df.city_pop.iloc[1] == 1
    assert df.city_pop.iloc[2] == 2
    assert df.city_pop.iloc[3] == 0
    assert df.city_pop.iloc[4] == 1
    assert df.city_pop.iloc[5] == 2


@pytest.mark.functions
def test_add_column_raise_error(dataframe):
    with pytest.raises(Exception):
        dataframe.add_column("cities", 1)


@pytest.mark.functions
def test_add_column_iterator_repeat_subtraction(dataframe):
    df = dataframe.add_column("city_pop", dataframe.a - dataframe.a)
    assert df.city_pop.sum() == 0
    assert df.city_pop.iloc[0] == 0


@pytest.mark.functions
def test_add_column_checkequality(df):
    new_df = df.add_column("fortytwo", 42)
    assert new_df is not df
