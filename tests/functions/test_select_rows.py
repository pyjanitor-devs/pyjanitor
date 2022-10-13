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
        match="Non-unique Index labels should be monotonic increasing.",
    ):
        not_unique.select_rows(slice("code", "code2"))


def test_unsorted_dates_slice(dates):
    """Raise Error if the dates are unsorted."""
    with pytest.raises(
        ValueError,
        match="The DatetimeIndex should be monotonic increasing.",
    ):
        dates.iloc[::-1].select_rows(slice("2011-01-31", "2011-03-31"))


def test_slice_start_presence(multiindex):
    """
    Raise ValueError if `rows` is a slice instance
    the start value is not present in the dataframe.
    """
    with pytest.raises(ValueError):
        multiindex.droplevel("first").select_rows(slice("bar", "one"))


def test_slice_stop_presence(multiindex):
    """
    Raise ValueError if `rows` is a slice instance
    and the stop value is not present in the dataframe.
    """
    with pytest.raises(ValueError):
        multiindex.droplevel("second").select_rows(slice("bar", "one"))


slicers = [slice("code2", "Name"), slice("code2", "Name", 2)]


@pytest.mark.parametrize("slicer", slicers)
def test_slice_reverse(slicer):
    """
    Test output for reverse slice
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
    actual = dups.select_rows(slicer)
    start = slicer.stop
    stop = slicer.start
    step = slicer.step
    expected = dups.loc[start:stop:step]
    expected = expected.loc[::-1]
    assert_frame_equal(actual, expected)


def test_slice_start(multiindex):
    """
    Raise ValueError if the search value
    is a slice instance  and the start value
    does not exist in the dataframe.
    """
    slicer = slice(1, "foo")
    msg = f"The start value for the slice {slicer}"
    msg += " must either be None or exist"
    msg += " in the dataframe's index."
    with pytest.raises(ValueError, match=re.escape(msg)):
        multiindex.select_rows(slicer)


def test_slice_stop(multiindex):
    """
    Raise ValueError if the search value
    is a slice instance  and the stop value
    does not exist in the dataframe
    """
    slicer = slice("bar", 1)
    msg = f"The stop value for the slice {slicer}"
    msg += " must either be None or exist"
    msg += " in the dataframe's index."
    with pytest.raises(ValueError, match=re.escape(msg)):
        multiindex.select_rows(slicer)


def test_slice_step(multiindex):
    """
    Raise ValueError if the search value
    is a slice instance and the step value
    is not an integer or None
    """
    slicer = slice("bar", "foo", "two")
    msg = f"The step value for the slice {slicer}"
    msg += " must either be an integer or None."
    with pytest.raises(ValueError, match=re.escape(msg)):
        multiindex.select_rows(slicer)


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


def test_multiindex_tuple_present(multiindex):
    """
    Test output for a MultiIndex and tuple passed.
    """
    assert_frame_equal(
        multiindex.select_rows(("bar", "one")),
        multiindex.loc[[("bar", "one")]],
    )


def test_dict_error(multiindex):
    """
    Raise if key in dict is tuple
    and value is not.
    """
    with pytest.raises(
        TypeError, match="If the level is a tuple, then a tuple of labels.+"
    ):
        multiindex.select_rows({(0, 1): "bar"})


def test_dict(multiindex):
    """Test output on a dict"""
    mapp = {"first": ["bar", "qux"], "second": "two"}
    expected = multiindex.select_rows(mapp)
    actual = multiindex.loc(axis=0)[["bar", "qux"], "two"]
    assert_frame_equal(expected, actual)


def test_dict_tuple(multiindex):
    """Test output on a dict"""
    mapp = {(0, 1): ("bar", "two")}
    expected = multiindex.select_rows(mapp)
    actual = multiindex.loc(axis=0)[("bar", "two"), slice(None)]
    assert_frame_equal(expected, actual)
