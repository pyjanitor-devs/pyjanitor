import datetime
import re

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal
from janitor.functions.utils import IndexLabel


@pytest.fixture
def dates():
    """pytest fixture"""
    start = datetime.datetime(2011, 1, 1)
    end = datetime.datetime(2012, 1, 1)
    rng = pd.date_range(start, end, freq="BM")
    return pd.DataFrame({"numbers": np.random.randn(len(rng))}, index=rng)


@pytest.fixture
def numbers():
    """pytest fixture"""
    return pd.DataFrame({"num": np.random.randn(20)})


@pytest.fixture
def multiindex():
    """pytest fixture."""
    arrays = [
        ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
        pd.Categorical(
            ["one", "two", "one", "two", "one", "two", "one", "two"]
        ),
    ]
    index = pd.MultiIndex.from_arrays(arrays, names=["first", "second"])
    return pd.DataFrame(np.random.randn(8, 4), index=index)


def test_number_not_found_index(numbers):
    """Raise KeyError if passed value is not found in the index."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        numbers.select_rows(2.5)


def test_string_not_found_numeric_index(numbers):
    """Raise KeyError if passed value is not found in the index."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        numbers.select_rows("2.5")


def test_regex_not_found_numeric_index(numbers):
    """Raise KeyError if passed value is not found in the index."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        numbers.select_rows(re.compile(".+"))


def test_regex_not_found_string_index(multiindex):
    """Raise KeyError if passed value is not found in the index."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        multiindex.droplevel("second").select_rows(re.compile("t.+"))


def test_date_not_found(dates):
    """Raise KeyError if passed value is not found in the index."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        dates.select_rows("2011-01-02")


def test_string_not_found_multi(multiindex):
    """Raise KeyError if passed string is not found in the MultiIndex."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        multiindex.droplevel("second").select_rows("2.5")


def test_tuple_not_found(multiindex):
    """Raise KeyError if passed tuple is not found in the index."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        multiindex.select_rows(("one", "bar"))


def test_list_not_found(numbers):
    """Raise KeyError if passed value in list is not found in the index."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        numbers.select_rows([2.5, 3])


def test_slice_unique():
    """
    Raise ValueError if the index is not unique.
    """
    not_unique = pd.DataFrame([], index=["code", "code2", "code1", "code"])
    with pytest.raises(
        ValueError,
        match="Non-unique index labels should be monotonic increasing.",
    ):
        not_unique.select_rows(slice("code", "code2"))


def test_unsorted_dates_slice(dates):
    """Raise Error if the dates are unsorted."""
    with pytest.raises(
        ValueError,
        match="The index is a DatetimeIndex and should be "
        "monotonic increasing.",
    ):
        dates.iloc[::-1].select_rows(slice("2011-01-31", "2011-03-31"))


def test_slice_start_presence(multiindex):
    """
    Raise ValueError if `rows` is a slice instance
    the start value is not present in the dataframe.
    """
    with pytest.raises(ValueError):
        multiindex.droplevel("first").select_rows(slice("bar", "one"))


def test_errors_indexlabel(multiindex):
    """
    Raise if `IndexLabel` is combined
    with other selection options
    """
    ix = IndexLabel("one")
    msg = "`IndexLabel` cannot be combined "
    msg += "with other selection options."
    with pytest.raises(NotImplementedError, match=msg):
        multiindex.select_rows(ix, "bar")


def test_slice_stop_presence(multiindex):
    """
    Raise ValueError if `rows` is a slice instance
    and the stop value is not present in the dataframe.
    """
    with pytest.raises(ValueError):
        multiindex.droplevel("second").select_rows(slice("bar", "one"))


def test_slice_step_presence():
    """
    Raise ValueError if `rows` is a slice instance,
    step value is provided, and the index is not unique.
    """
    index = [
        "Name",
        "code",
        "code",
        "code1",
        "code2",
        "code2",
        "code3",
        "id",
        "type",
        "type1",
        "type2",
        "type3",
    ]
    dups = pd.DataFrame([], index=index)
    with pytest.raises(ValueError):
        dups.select_rows(slice("code3", "code", 2))


def test_slice_dtypes(multiindex):
    """
    Raise ValueError if `columns_to_select` is a slice instance
    and either the start value or the stop value is not a string,
    or the step value is not an integer.
    """
    multi = multiindex.droplevel(1).sort_index()
    with pytest.raises(
        ValueError,
        match="The start value for the slice must either be `None`.+",
    ):
        multi.select_rows(slice(1, "foo"))
    with pytest.raises(
        ValueError,
        match="The stop value for the slice must either be `None`.+",
    ):
        multi.select_rows(slice("bar", 1))
    with pytest.raises(ValueError, match="The step value for the slice.+"):
        multi.select_rows(slice("bar", "foo", "two"))


def test_boolean_list_uneven_length(dates):
    """
    Raise ValueError if `rows` is a list of booleans
    and the length is unequal to the length of the dataframe's index
    """
    with pytest.raises(
        ValueError, match="The length of the list of booleans.+"
    ):
        dates.select_rows([True, False])


def test_invert_num(numbers):
    """Test output when rows are dropped."""
    expected = numbers.select_rows([4, 6, 10], invert=True)
    actual = numbers.drop([4, 6, 10])
    assert_frame_equal(expected, actual)


def test_level_string(multiindex):
    """Test output on a level of the MultiIndex."""
    expected = multiindex.select_rows(IndexLabel("one", level=1))
    actual = multiindex.xs(key="one", level=1, drop_level=False)
    assert_frame_equal(expected, actual)


def test_date_partial_output(dates):
    """Test output on a date"""
    expected = dates.select_rows("2011")
    actual = dates.loc["2011"]
    assert_frame_equal(expected, actual, check_freq=False)


def test_date_actual_output(dates):
    """Test output on a date"""
    expected = dates.select_rows("2011-01-31")
    actual = dates.loc[["2011-01-31"]]
    assert_frame_equal(expected, actual, check_freq=False)


def test_slice_dates(dates):
    """Test output of slice on dates."""
    slicer = slice("2011-01-31", "2011-03-31")
    expected = dates.select_rows(slicer)
    actual = dates.loc[slicer]
    assert_frame_equal(expected, actual, check_freq=False)


def test_slice_dates_inexact(dates):
    """Test output of slice on dates."""
    slicer = slice("2011-01", "2011-03")
    expected = dates.select_rows(slicer)
    actual = dates.loc[slicer]
    assert_frame_equal(expected, actual, check_freq=False)


def test_slice1(dates):
    """Test output of slice on index."""
    expected = dates.select_rows(slice(None, None))
    assert_frame_equal(expected, dates, check_freq=False)


def test_slice2(dates):
    """Test output of slice on index."""
    expected = dates.select_rows(slice(None, None, 2))
    assert_frame_equal(expected, dates.loc[::2], check_freq=False)


def test_slice3(dates):
    """Test output of slice on index."""
    expected = dates.select_rows(slice("2011-10", "2011-04", 2))
    assert_frame_equal(
        expected,
        dates.loc[slice("2011-04", "2011-10", 2)].loc[::-1],
        check_freq=False,
    )


def test_boolean_list(multiindex):
    """
    Test output for boolean list
    """
    booleans = [True, True, True, False, False, False, True, True]
    expected = multiindex.select_rows(booleans)
    assert_frame_equal(multiindex.loc[booleans], expected)


def test_callable(dates):
    """
    Test output for callable
    """
    func = lambda df: df.index.month == 4  # noqa : E731
    assert_frame_equal(
        dates.loc[func], dates.select_rows(func), check_freq=False
    )
