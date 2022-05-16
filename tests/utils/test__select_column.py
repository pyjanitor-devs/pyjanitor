import datetime
import re

import numpy as np
import pandas as pd
import pytest

from pandas.testing import assert_index_equal
from janitor.functions.utils import _select_column_names, patterns


@pytest.fixture
def df_dates():
    """pytest fixture"""
    start = datetime.datetime(2011, 1, 1)
    end = datetime.datetime(2012, 1, 1)
    rng = pd.date_range(start, end, freq="BM")
    return pd.DataFrame([np.random.randn(len(rng))], columns=rng)


@pytest.fixture
def df_numbers():
    """pytest fixture"""
    return pd.DataFrame([np.random.randn(20)], columns=range(20))


@pytest.fixture
def df():
    """pytest fixture."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "M_start_date_1": [201709, 201709, 201709],
            "M_end_date_1": [201905, 201905, 201905],
            "M_start_date_2": [202004, 202004, 202004],
            "M_end_date_2": [202005, 202005, 202005],
            "F_start_date_1": [201803, 201803, 201803],
            "F_end_date_1": [201904, 201904, 201904],
            "F_start_date_2": [201912, 201912, 201912],
            "F_end_date_2": [202007, 202007, 202007],
        }
    )


@pytest.fixture
def df1():
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
def df_tuple():
    "pytest fixture."
    frame = pd.DataFrame(
        {
            "A": {0: "a", 1: "b", 2: "c"},
            "B": {0: 1, 1: 3, 2: 5},
            "C": {0: 2, 1: 4, 2: 6},
        }
    )
    frame.columns = [list("ABC"), list("DEF")]
    return frame


def test_col_not_found(df):
    """Raise KeyError if `columns_to_select` is not in df.columns."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        _select_column_names(2.5, df)


def test_col_not_found1(df):
    """Raise KeyError if `columns_to_select` is not in df.columns."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        _select_column_names(1, df)


def test_col_not_found2(df):
    """Raise KeyError if `columns_to_select` is not in df.columns."""
    with pytest.raises(KeyError, match="No match was returned.+"):
        _select_column_names([3, "id"], df)


def test_col_not_found3(df_dates):
    """Raise KeyError if `columns_to_select` is not in df.columns."""
    with pytest.raises(KeyError):
        _select_column_names("id", df_dates)


def test_col_not_found4(df_numbers):
    """Raise KeyError if `columns_to_select` is not in df.columns."""
    with pytest.raises(KeyError, match=r"Strings\(.+\) can be applied.+"):
        _select_column_names("id", df_numbers)


def test_tuple(df_tuple):
    """Test _select_column_names function on tuple."""
    assert _select_column_names(("A", "D"), df_tuple) == [("A", "D")]


def test_strings(df1):
    """Test _select_column_names function on strings."""
    assert _select_column_names("id", df1) == ["id"]
    assert _select_column_names("*type*", df1) == [
        "type",
        "type1",
        "type2",
        "type3",
    ]


def test_strings_do_not_exist(df):
    """
    Raise KeyError if `columns_to_select` is a string
    and does not exist in the dataframe's columns.
    """
    with pytest.raises(KeyError, match="No match was returned for.+"):
        _select_column_names("word", df)


def test_strings_dates(df_dates):
    """Test output for datetime column."""
    assert _select_column_names("2011-01-31", df_dates), df_dates.loc[
        :, "2011-01-31"
    ].name


def test_unsorted_dates(df_dates):
    """Raise Error if the dates are unsorted."""
    df_dates = df_dates.iloc[:, [10, 4, 7, 2, 1, 3, 5, 6, 8, 9, 11, 0]]
    with pytest.raises(
        ValueError,
        match="The column is a DatetimeIndex and should be "
        "monotonic increasing.",
    ):
        _select_column_names("2011-01-31", df_dates)


def test_regex(df1):
    """Test _select_column_names function on regular expressions."""
    assert_index_equal(
        _select_column_names(re.compile(r"\d$"), df1),
        df1.filter(regex=r"\d$").columns,
    )


def test_patterns_warning(df1):
    """
    Check that warning is raised if `janitor.patterns` is used.
    """
    with pytest.warns(DeprecationWarning):
        assert_index_equal(
            _select_column_names(patterns(r"\d$"), df1),
            df1.filter(regex=r"\d$").columns,
        )


def test_regex_presence(df_dates):
    """
    Raise KeyError if `columns_to_select` is a regex
    and the columns is not a string column.
    """
    with pytest.raises(
        KeyError, match=r"Regular expressions\(.+\) can be applied.+"
    ):
        _select_column_names(re.compile(r"^\d+"), df_dates)


def test_slice_unique():
    """
    Raise ValueError if the columns are not unique.
    """
    not_unique = pd.DataFrame([], columns=["code", "code", "code1", "code2"])
    with pytest.raises(ValueError):
        _select_column_names(slice("code", "code2"), not_unique)


def test_slice_presence(df):
    """
    Raise ValueError if `columns_to_select` is a slice instance
    and either the start value or the end value is not present
    in the dataframe.
    """
    with pytest.raises(ValueError):
        _select_column_names(slice("Id", "M_start_date_1"), df)
    with pytest.raises(ValueError):
        _select_column_names(slice("id", "M_end_date"), df)


def test_slice_dtypes(df):
    """
    Raise ValueError if `columns_to_select` is a slice instance
    and either the start value or the stop value is not a string,
    or the step value is not an integer.
    """
    with pytest.raises(ValueError):
        _select_column_names(slice(1, "M_end_date_2"), df)
    with pytest.raises(ValueError):
        _select_column_names(slice("id", 2), df)
    with pytest.raises(ValueError):
        _select_column_names(slice("id", "M_end_date_2", "3"), df)


def test_unsorted_dates_slice(df_dates):
    """Raise Error if the dates are unsorted."""
    df_dates = df_dates.iloc[:, ::-1]
    with pytest.raises(
        ValueError,
        match="The column is a DatetimeIndex and should be "
        "monotonic increasing.",
    ):
        _select_column_names(slice("2011-01-31", "2011-03-31"), df_dates)


def test_slice(df1):
    """Test _select_column_names function on slices."""
    assert_index_equal(
        _select_column_names(slice("code", "code2"), df1),
        df1.loc[:, slice("code", "code2")].columns,
    )

    assert_index_equal(
        _select_column_names(slice("code2", None), df1),
        df1.loc[:, slice("code2", None)].columns,
    )

    assert_index_equal(
        _select_column_names(slice(None, "code2"), df1),
        df1.loc[:, slice(None, "code2")].columns,
    )

    assert_index_equal(
        _select_column_names(slice(None, None), df1), df1.columns
    )
    assert_index_equal(
        _select_column_names(slice(None, None, 2), df1),
        df1.loc[:, slice(None, None, 2)].columns,
    )
    assert _select_column_names(slice("code2", "code"), df1).tolist() == [
        "code2",
        "code1",
        "code",
    ]


def test_slice_dates(df_dates):
    """Test output of slice on date column."""
    assert_index_equal(
        _select_column_names(slice("2011-01-31", "2011-03-31"), df_dates),
        df_dates.loc[:, "2011-01-31":"2011-03-31"].columns,
    )


def test_slice_dates_inexact(df_dates):
    """Test output of slice on date column."""
    assert_index_equal(
        _select_column_names(slice("2011-01", "2011-03"), df_dates),
        df_dates.loc[:, "2011-01":"2011-03"].columns,
    )


@pytest.mark.xfail(reason="level parameter removed.")
def test_level_type(df_tuple):
    """Raise TypeError if `level` is the wrong type."""
    with pytest.raises(TypeError):
        _select_column_names("A", df_tuple)


@pytest.mark.xfail(reason="level parameter removed.")
def test_level_nonexistent(df_tuple):
    """
    Raise ValueError if column is a MultiIndex
    and level is `None`.
    """
    with pytest.raises(ValueError):
        _select_column_names("A", df_tuple)


@pytest.mark.xfail(reason="level parameter removed.")
def test_tuple_callable(df_tuple):
    """
    Raise ValueError if dataframe has MultiIndex columns
    and a callable is provided.
    """
    with pytest.raises(ValueError):
        _select_column_names(lambda df: df.name.startswith("A"), df_tuple)


@pytest.mark.xfail(reason="level parameter removed.")
def test_tuple_regex(df_tuple):
    """
    Raise ValueError if dataframe has MultiIndex columns'
    a regex is provided and level is None.
    """
    with pytest.raises(ValueError):
        _select_column_names(re.compile("A"), df_tuple)


def test_boolean_list_dtypes(df):
    """
    Raise ValueError if `columns_to_select` is a list of booleans
    and the length is unequal to the number of columns
    in the dataframe.
    """
    with pytest.raises(ValueError):
        _select_column_names([True, False], df)
    with pytest.raises(ValueError):
        _select_column_names(
            [True, True, True, False, False, False, True, True, True, False],
            df,
        )


def test_callable(df_numbers):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and at lease one Series has a wrong data type
    that makes the callable unapplicable.
    """
    with pytest.raises(
        TypeError,
        match="The output of the applied callable "
        "should be a boolean array.",
    ):
        _select_column_names(lambda df: df + 3, df_numbers)


@pytest.mark.xfail(reason="Indexing in Pandas is possible with a Series.")
def test_callable_returns_series(df):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and returns a Series.
    """
    with pytest.raises(ValueError):
        _select_column_names(lambda x: x + 1, df)


def test_callable_no_match(df):
    """
    Test if `columns_to_select` is a callable,
    and no value is returned.
    """
    assert _select_column_names(pd.api.types.is_float_dtype, df).empty

    assert _select_column_names(lambda x: "Date" in x.name, df).empty


def test_tuple_presence(df_tuple):
    """
    Raise KeyError if `columns_to_select` is a tuple
    and no match is returned.
    """
    with pytest.raises(KeyError, match="No match was returned.+"):
        _select_column_names(("A", "C"), df_tuple)


def test_callable_data_type(df1):
    """
    Test _select_column_names function on callables,
    specifically for data type checks.
    """
    assert_index_equal(
        _select_column_names(pd.api.types.is_integer_dtype, df1),
        df1.select_dtypes(int).columns,
    )

    assert_index_equal(
        _select_column_names(pd.api.types.is_float_dtype, df1),
        df1.select_dtypes(float).columns,
    )

    assert_index_equal(
        _select_column_names(pd.api.types.is_numeric_dtype, df1),
        df1.select_dtypes("number").columns,
    )

    assert_index_equal(
        _select_column_names(pd.api.types.is_categorical_dtype, df1),
        df1.select_dtypes("category").columns,
    )

    assert_index_equal(
        _select_column_names(pd.api.types.is_datetime64_dtype, df1),
        df1.select_dtypes(np.datetime64).columns,
    )

    assert_index_equal(
        _select_column_names(pd.api.types.is_object_dtype, df1),
        df1.select_dtypes("object").columns,
    )


def test_callable_string_methods(df1):
    """
    Test _select_column_names function on callables,
    specifically for column name checks.
    """
    assert_index_equal(
        _select_column_names(lambda x: x.name.startswith("type"), df1),
        df1.filter(like="type").columns,
    )

    assert_index_equal(
        _select_column_names(lambda x: x.name.endswith(("1", "2", "3")), df1),
        df1.filter(regex=r"\d$").columns,
    )

    assert_index_equal(
        _select_column_names(lambda x: "d" in x.name, df1),
        df1.filter(regex="d").columns,
    )

    assert_index_equal(
        _select_column_names(
            lambda x: x.name.startswith("code") and x.name.endswith("1"), df1
        ),
        df1.filter(regex=r"code.*1$").columns,
    )

    assert_index_equal(
        _select_column_names(
            lambda x: x.name.startswith("code") or x.name.endswith("1"), df1
        ),
        df1.filter(regex=r"^code.*|.*1$").columns,
    )


def test_callable_computations(df1):
    """
    Test _select_column_names function on callables,
    specifically for computations.
    """
    assert_index_equal(
        _select_column_names(lambda x: x.isna().any(), df1),
        df1.columns[df1.isna().any().array],
    )


def test_list_various(df1):
    """Test _select_column_names function on list type."""

    assert _select_column_names(["id", "Name"], df1) == ["id", "Name"]
    assert _select_column_names(["id", "code*"], df1) == list(
        df1.filter(regex="^id|^code").columns
    )
    assert [
        *_select_column_names(["id", "code*", slice("code", "code2")], df1)
    ] == df1.filter(regex="^(id|code)").columns.tolist()
    assert _select_column_names(["id", "Name"], df1) == ["id", "Name"]


def test_list_boolean(df):
    """Test _select_column_names function on list of booleans."""
    booleans = [True, True, True, False, False, False, True, True, True]
    assert_index_equal(
        _select_column_names(booleans, df), df.columns[booleans]
    )
