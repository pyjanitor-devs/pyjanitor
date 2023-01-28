import datetime
import re

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_frame_equal


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


def test_errors_MultiIndex_dict(multiindex):
    """
    Raise if `level` is an int/string
    and duplicated
    """
    ix = {"second": "one", 0: "bar"}
    msg = "The keys in the dictionary represent the levels "
    msg += "in the MultiIndex, and should either be all "
    msg += "strings or integers."
    with pytest.raises(TypeError, match=msg):
        multiindex.select_rows(ix)


def test_dict(multiindex):
    """Test output on a dict"""
    mapp = {"first": ["bar", "qux"], "second": "two"}
    expected = multiindex.select_rows(mapp)
    actual = multiindex.loc(axis=0)[["bar", "qux"], "two"]
    assert_frame_equal(expected, actual)


def test_boolean_multiindex(multiindex):
    """Raise if boolean length does not match index length"""
    with pytest.raises(IndexError):
        multiindex.select_rows(lambda df: [True, False])


def test_set(dates):
    """
    Test output if input is a set
    """
    assert_frame_equal(
        dates.select_rows({"2011-01-31"}),
        dates.loc[["2011-01-31"]],
        check_freq=False,
    )


def test_dict_single_index(dates):
    """
    Test output for dict on a single index
    """
    assert_frame_equal(
        dates.select_rows({"2011-01-31": 1.3}),
        dates.loc[["2011-01-31"]],
        check_freq=False,
    )


def test_array(dates):
    """Test output for pandas array"""
    arr = pd.array(["2011-01-31"])
    expected = dates.select_rows(arr)
    actual = dates.loc[arr]
    assert_frame_equal(expected, actual, check_freq=False)


def test_series(dates):
    """Test output for pandas Series"""
    arr = pd.Series(["2011-01-31"])
    expected = dates.select_rows(arr)
    actual = dates.loc[arr]
    assert_frame_equal(expected, actual, check_freq=False)


def test_numpy_array(dates):
    """Test output for pandas array"""
    arr = np.array(["2011-01-31"])
    expected = dates.select_rows(arr)
    actual = dates.loc[arr]
    assert_frame_equal(expected, actual)


def test_array_bool(dates):
    """Test output for pandas array"""
    arr = np.array([True, False]).repeat(6)
    expected = dates.select_rows(arr)
    actual = dates.loc[arr]
    assert_frame_equal(expected, actual)


def test_boolean_Index(dates):
    """Raise if boolean is not same length as index"""
    with pytest.raises(IndexError):
        arr = pd.Index([True, False]).repeat(4)
        dates.select_rows(arr)


def test_missing_all_array(dates):
    """Raise if none of the labels exist."""
    with pytest.raises(KeyError):
        arr = pd.array(["2011"])
        dates.select_rows(arr)


def test_missing_some_array(dates):
    """Raise if some of the labels do not exist."""
    with pytest.raises(KeyError):
        arr = pd.array(["2011", "2011-01-31"])
        dates.select_rows(arr)
