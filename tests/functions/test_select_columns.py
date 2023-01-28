import pandas as pd
import datetime
import numpy as np
import re
import pytest
from pandas.testing import assert_frame_equal
from itertools import product

from janitor.functions.utils import patterns, DropLabel
from pandas.api.types import is_numeric_dtype


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["a", "Bell__Chart", "cities"]),
        (True, ["decorated-elephant", "animals@#$%^"]),
    ],
)
def test_select_column_names(dataframe, invert, expected):
    "Base DataFrame"
    columns = ["a", "Bell__Chart", "cities"]
    df = dataframe.select_columns(columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (True, ["a", "Bell__Chart", "cities"]),
        (False, ["decorated-elephant", "animals@#$%^"]),
    ],
)
def test_select_column_names_droplabel(dataframe, invert, expected):
    "Base DataFrame"
    columns = ["a", "Bell__Chart", "cities"]
    df = dataframe.select_columns(DropLabel(columns), invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
def test_select_column_names_droplabel_multiple(dataframe):
    "Base DataFrame"
    columns = ["a", "Bell__Chart", "cities"]
    cols = [DropLabel(ent) for ent in columns]
    df = dataframe.select_columns(*cols)

    assert_frame_equal(df, dataframe.drop(columns=columns))


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["Bell__Chart", "a", "animals@#$%^"]),
        (True, ["decorated-elephant", "cities"]),
    ],
)
def test_select_column_names_glob_inputs(dataframe, invert, expected):
    "Base DataFrame"
    columns = ["Bell__Chart", "a*"]
    df = dataframe.select_columns(columns, invert=invert)

    assert_frame_equal(df, dataframe.loc(axis=1)[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "columns",
    [
        ["a", "Bell__Chart", "foo"],
        ["a", "Bell__Chart", "foo", "bar"],
        ["a*", "Bell__Chart", "foo"],
        ["a*", "Bell__Chart", "foo", "bar"],
    ],
)
def test_select_column_names_missing_columns(dataframe, columns):
    """Check that passing non-existent column names or search strings raises KeyError"""  # noqa: E501
    with pytest.raises(KeyError):
        dataframe.select_columns(columns)


@pytest.mark.xfail(reason="return whatever user passes")
@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["Bell__Chart", "a", "decorated-elephant"]),
        (True, ["animals@#$%^", "cities"]),
    ],
)
def test_select_unique_columns(dataframe, invert, expected):
    """Test that only unique columns are returned."""
    columns = ["Bell__*", slice("a", "decorated-elephant")]
    df = dataframe.select_columns(columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


@pytest.mark.functions
@pytest.mark.parametrize(
    "invert,expected",
    [
        (False, ["Bell__Chart", "decorated-elephant"]),
        (True, ["a", "animals@#$%^", "cities"]),
    ],
)
def test_select_callable_columns(dataframe, invert, expected):
    """Test that columns are returned when a callable is passed."""

    def columns(frame):
        return frame.columns.str.contains("[-,__]")

    df = dataframe.select_columns(columns, invert=invert)

    assert_frame_equal(df, dataframe[expected])


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
    return pd.DataFrame(np.random.randn(4, 8), columns=index)


def test_multiindex(multiindex):
    """
    Test output for a MultiIndex and tuple passed.
    """
    assert_frame_equal(
        multiindex.select_columns(("bar", "one")),
        multiindex.loc[:, [("bar", "one")]],
    )


@pytest.fixture
def df_dates():
    """pytest fixture"""
    start = datetime.datetime(2011, 1, 1)
    end = datetime.datetime(2012, 1, 1)
    rng = pd.date_range(start, end, freq="BM")
    return pd.DataFrame([np.random.randn(len(rng))], columns=rng)


@pytest.fixture
def df_strings():
    """pytest fixture."""
    return pd.DataFrame(
        {
            "id": [0, 1],
            "Name": ["ABC", "XYZ"],
            "code": [1, 2],
            "code1": [4, np.nan],
            "code2": ["8", 5],
            "type": ["S", "R"],
            "type1": ["E", np.nan],
            "type2": ["T", "U"],
            "code3": pd.Series(["a", "b"], dtype="category"),
            "type3": pd.to_datetime(
                [np.datetime64("2018-01-01"), datetime.datetime(2018, 1, 1)]
            ),
        }
    )


@pytest.fixture
def numbers():
    """pytest fixture"""
    return pd.DataFrame([np.random.randn(20)], columns=range(20))


def test_col_not_found(numbers):
    """
    Raise KeyError if the search value is a string,
    is not in df.columns,
    and df.columns is not date/string/categorical.
    """
    with pytest.raises(KeyError, match="No match was returned.+"):
        numbers.select_columns("sam")


def test_col_not_found3(df_dates):
    """
    Raise KeyError if the search value is not in df.columns,
    and df.columns is a datetime index.
    """
    with pytest.raises(KeyError):
        df_dates.select_columns("id")


def test_strings_cat(df_strings):
    """Test output on categorical columns"""
    df_strings.columns = df_strings.columns.astype("category")
    assert_frame_equal(
        df_strings.select_columns("id"), df_strings.loc[:, ["id"]]
    )
    assert_frame_equal(
        df_strings.select_columns("*type*"), df_strings.filter(like="type")
    )


def test_regex(df_strings):
    """Test output on regular expressions."""
    assert_frame_equal(
        df_strings.select_columns(re.compile(r"\d$")),
        df_strings.filter(regex=r"\d$"),
    )


def test_regex_cat(df_strings):
    """Test output on categorical columns"""
    df_strings.columns = df_strings.columns.astype("category")
    assert_frame_equal(
        df_strings.select_columns(re.compile(r"\d$")),
        df_strings.filter(regex=r"\d$"),
    )


def test_patterns_warning(df_strings):
    """
    Check that warning is raised if `janitor.patterns` is used.
    """
    with pytest.warns(DeprecationWarning):
        assert_frame_equal(
            df_strings.select_columns(patterns(r"\d$")),
            df_strings.filter(regex=r"\d$"),
        )


def test_regex_presence_string_column(df_strings):
    """
    Raise KeyError if search_value is a regex
    and does not exist in the dataframe's columns.
    """
    with pytest.raises(KeyError, match="No match was returned for.+"):
        df_strings.select_columns(re.compile("word"))


def test_regex_presence(df_dates):
    """
    Raise KeyError if search_value is a regex
    and the columns is not a string column.
    """
    with pytest.raises(KeyError, match=r"No match was returned.+"):
        df_dates.select_columns(re.compile(r"^\d+"))


def test_slice_unique():
    """
    Raise ValueError if the columns are not unique.
    """
    not_unique = pd.DataFrame([], columns=["code", "code2", "code1", "code"])
    with pytest.raises(
        ValueError,
        match="Non-unique Index labels should be monotonic increasing.",
    ):
        not_unique.select_columns(slice("code", "code2"))


def test_unsorted_dates_slice(df_dates):
    """Raise Error if the date column is unsorted."""
    df_dates = df_dates.iloc[:, ::-1]
    with pytest.raises(
        ValueError,
        match="The DatetimeIndex should be monotonic increasing.",
    ):
        df_dates.select_columns(slice("2011-01-31", "2011-03-31"))


slicers = [
    slice("code", "code2"),
    slice("code2", None),
    slice(None, "code2"),
    slice(None, None),
    slice(None, None, 2),
]
slicers = product(["df_strings"], slicers)


@pytest.mark.parametrize(
    "df_strings, slicer", slicers, indirect=["df_strings"]
)
def test_slice(df_strings, slicer):
    """Test output on slices."""
    assert_frame_equal(
        df_strings.select_columns(slicer), df_strings.loc[:, slicer]
    )


def test_slice_reverse(df_strings):
    """
    Test output on a reverse slice
    """
    actual = df_strings.select_columns(slice("code2", "code", -1))
    expected = df_strings.loc[
        :,
        [
            "code2",
            "code1",
            "code",
        ],
    ]

    assert_frame_equal(actual, expected)


def test_slice_dates(df_dates):
    """Test output of slice on date column."""
    actual = df_dates.select_columns(slice("2011-01-31", "2011-03-31"))
    expected = df_dates.loc[:, "2011-01-31":"2011-03-31"]
    assert_frame_equal(actual, expected)


def test_slice_dates_inexact(df_dates):
    """Test output of slice on date column."""
    actual = df_dates.select_columns(slice("2011-01", "2011-03"))
    expected = df_dates.loc[:, "2011-01":"2011-03"]
    assert_frame_equal(actual, expected)


def test_boolean_list_dtypes(df_dates):
    """
    Raise ValueError if the search value
    is a list of booleans and the length
    is unequal to the number of columns
    in the dataframe.
    """
    with pytest.raises(
        ValueError, match="The length of the list of booleans.+"
    ):
        df_dates.select_columns([True, False])


def test_list_boolean(df_dates):
    """Test output on a list of booleans."""
    booleans = np.repeat([True, False], 6)
    actual = df_dates.select_columns(booleans)
    expected = df_dates.loc[:, booleans]
    assert_frame_equal(actual, expected)


def test_number_dates(df_dates):
    """Raise if selecting number on a date column"""
    with pytest.raises(KeyError, match="No match was returned for 2.5"):
        df_dates.select_columns(2.5)


def test_callable(numbers):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and at lease one Series has a wrong data type
    that makes the callable unapplicable.
    """
    with pytest.raises(
        ValueError,
        match="The output of the applied callable "
        "should be a 1-D boolean array.",
    ):
        numbers.select_columns(lambda df: df + 3)


def test_callable_length(numbers):
    """
    Raise if the boolean output from the callable
    is not the same as the length of the columns.
    """
    with pytest.raises(
        IndexError, match="The boolean array output from the callable.+"
    ):
        numbers.select_columns(lambda df: [True, False])


@pytest.mark.functions
def test_callable_dtype(dataframe):
    """Test output when selecting columnns based on dtype"""
    expected = dataframe.select_dtypes("number")
    actual = dataframe.select_columns(is_numeric_dtype)
    assert_frame_equal(expected, actual)


def test_dict(multiindex):
    """Test output on a dict"""
    mapp = {"first": ["bar", "qux"], "second": "two"}
    expected = multiindex.select_columns(mapp)
    actual = multiindex.loc(axis=1)[["bar", "qux"], "two"]
    assert_frame_equal(expected, actual)


def test_dict_callable(multiindex):
    """Test output on a dict"""
    mapp = {"first": ["bar", "qux"], "second": lambda df: df == "two"}
    expected = multiindex.select_columns(mapp)
    actual = multiindex.loc(axis=1)[["bar", "qux"], "two"]
    assert_frame_equal(expected, actual)


def test_dict_regex(multiindex):
    """Test output on a dict"""
    mapp = {"first": ["bar", "qux"], "second": re.compile("tw.")}
    expected = multiindex.select_columns(mapp)
    actual = multiindex.loc(axis=1)[["bar", "qux"], "two"]
    assert_frame_equal(expected, actual)


def test_boolean_series_multi(multiindex):
    """Test boolean output on a MultiIndex"""
    mapp = pd.Series([True, False]).repeat(4)
    expected = multiindex.select_columns(mapp, "foo")
    actual = multiindex.loc(axis=1)["bar":"foo"]
    assert_frame_equal(expected, actual)


def test_boolean_list_multi(multiindex):
    """Test boolean output on a MultiIndex"""
    mapp = [True, True, True, True, False, False, False, False]
    expected = multiindex.select_columns(mapp, "foo")
    actual = multiindex.loc(axis=1)["bar":"foo"]
    assert_frame_equal(expected, actual)


def test_series_multi(multiindex):
    """Test pd.Series output on a MultiIndex"""
    expected = multiindex.select_columns(pd.Series("bar"), slice("baz", "foo"))
    actual = multiindex.loc(axis=1)["bar":"foo"]
    assert_frame_equal(expected, actual)


def test_glob_multi(multiindex):
    """Test fnmatch output on a MultiIndex"""
    expected = multiindex.select_columns("b*r")
    actual = multiindex.loc(axis=1)[["bar"]]
    assert_frame_equal(expected, actual)


def test_regex_multi(multiindex):
    """Test regex output on a MultiIndex"""
    expected = multiindex.select_columns(re.compile("b.r"))
    actual = multiindex.loc(axis=1)[["bar"]]
    assert_frame_equal(expected, actual)
