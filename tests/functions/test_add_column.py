import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis import strategies as st
from pandas.testing import assert_series_equal

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_add_column_add_integer(df):
    """col_name wasn't a string"""
    with pytest.raises(TypeError):
        df.add_column(column_name=42, value=42)


@pytest.mark.functions
def test_add_column_already_exists(dataframe):
    """column already exists"""
    with pytest.raises(
        ValueError,
        match="Attempted to add column that already exists",
    ):
        dataframe.add_column("a", 42)


@pytest.mark.functions
def test_add_column_too_many(dataframe):
    """too many values for dataframe num rows"""
    with pytest.raises(
        ValueError,
        match="`value` has more elements than number of rows",
    ):
        dataframe.add_column("toomany", np.ones(100))


@pytest.mark.functions
def test_add_column_too_few_but_no_fill_remaining(dataframe):
    """too few values for dataframe num rows"""
    with pytest.raises(
        ValueError,
        match="add iterable of values with length not equal",
    ):
        dataframe.add_column("toomany", np.ones(2), fill_remaining=False)


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_add_column_scalar(df):
    """Checks `add_column` works as expected when adding a numeric scalar
    to the column"""
    # column appears in DataFrame
    df = df.add_column("fortytwo", 42)
    assert "fortytwo" in df.columns

    # values are correct in dataframe for scalar
    series = pd.Series([42] * len(df))
    series.name = "fortytwo"
    assert_series_equal(df["fortytwo"], series)


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_add_column_string(df):
    """Checks `add_column` works as expected when adding a string scalar
    to the column.

    And ensure no errors are raised. The error checks on iterables should
    exclude strings, which also have a length.
    """
    df = df.add_column("fortythousand", "test string")
    series = pd.Series(["test string" for _ in range(len(df))])
    series.name = "fortythousand"
    assert_series_equal(df["fortythousand"], series)

    # values are correct in dataframe for iterable
    vals = np.linspace(0, 43, len(df))
    df = df.add_column("fortythree", vals)
    series = pd.Series(vals)
    series.name = "fortythree"
    assert_series_equal(df["fortythree"], series)


@pytest.mark.functions
def test_add_column_iterator_repeat_subtraction(dataframe):
    """Checks `add_column` works as expected when adding a pd Series
    to the column"""
    df = dataframe.add_column("city_pop", dataframe.a - dataframe.a)
    assert df.city_pop.sum() == 0
    assert df.city_pop.iloc[0] == 0


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_add_column_fill_scalar(df):
    """Checks the `fill_remaining` parameter works as expected when value
    is a scalar."""
    df = df.add_column("fill_in_scalar", 42, fill_remaining=True)
    series = pd.Series([42 for _ in range(len(df))])
    series.name = "fill_in_scalar"
    assert_series_equal(df["fill_in_scalar"], series)


@pytest.mark.functions
@given(df=df_strategy(), vals=st.lists(elements=st.integers()))
@settings(deadline=None)
def test_add_column_fill_remaining_iterable(df, vals: list):
    """Checks the `fill_remaining` parameter works as expected."""
    if len(vals) > len(df) or not vals:
        with pytest.raises(ValueError):
            df = df.add_column("fill_in_iterable", vals, fill_remaining=True)
    else:
        df = df.add_column("fill_in_iterable", vals, fill_remaining=True)
        assert not pd.isna(df["fill_in_iterable"]).any()


@pytest.mark.functions
def test_add_column_iterator_repeat(dataframe):
    """Fill remaining using a small iterator, with `fill_remaining` set to
    True."""
    df = dataframe.add_column("city_pop", range(3), fill_remaining=True)
    assert df.city_pop.iloc[0] == 0
    assert df.city_pop.iloc[1] == 1
    assert df.city_pop.iloc[2] == 2
    assert df.city_pop.iloc[3] == 0
    assert df.city_pop.iloc[4] == 1
    assert df.city_pop.iloc[5] == 2
