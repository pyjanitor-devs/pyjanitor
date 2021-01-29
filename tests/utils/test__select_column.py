import datetime
import re

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal

from janitor import patterns
from janitor.utils import _select_columns


@pytest.fixture
def df():
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


def test_type(df):
    """Raise TypeError if `columns_to_select` is the wrong type."""
    with pytest.raises(TypeError):
        _select_columns(2.5, df)
    with pytest.raises(TypeError):
        _select_columns(1, df)
    with pytest.raises(TypeError):
        _select_columns([3, "id"], df)


def test_strings_do_not_exist(df):
    """
    Raise KeyError if `columns_to_select` is a string
    and does not exist in the dataframe's columns.
    """
    with pytest.raises(KeyError):
        _select_columns("word", df)
    with pytest.raises(KeyError):
        _select_columns("*starter", df)


def test_slice_dtypes(df):
    """
    Raise ValueError if `columns_to_select` is a slice instance
    and either the start value or the stop value is not a string,
    or the step value is not an integer.
    """
    with pytest.raises(ValueError):
        _select_columns(slice(1, "M_end_date_2"), df)
    with pytest.raises(ValueError):
        _select_columns(slice("id", 2), df)
    with pytest.raises(ValueError):
        _select_columns(slice("id", "M_end_date_2", "3"), df)


def test_slice_presence(df):
    """
    Raise ValueError if `columns_to_select` is a slice instance
    and either the start value or the end value is not present
    in the dataframe.
    """
    with pytest.raises(ValueError):
        _select_columns(slice("Id", "M_start_date_1"), df)
    with pytest.raises(ValueError):
        _select_columns(slice("id", "M_end_date"), df)


def test_callable(df):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and at lease one Series has a wrong data type
    that makes the callable unapplicable.
    """
    with pytest.raises(TypeError):
        _select_columns(object, df)


def test_callable_returns_Series(df):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and returns a Series.
    """
    with pytest.raises(ValueError):
        _select_columns(lambda x: x + 1, df)


def test_callable_no_match(df):
    """
    Raise ValueError if `columns_to_select` is a callable, and
    no boolean results are returned, when the callable is
    applied to each series in the dataframe.
    """
    with pytest.raises(ValueError):
        _select_columns(pd.api.types.is_float_dtype, df)

    with pytest.raises(ValueError):
        _select_columns(lambda x: "Date" in x.name, df)


def test_regex_presence(df):
    """
    Raise KeyError if `columns_to_select` is a regex
    and none of the column names match.
    """
    with pytest.raises(KeyError):
        _select_columns(re.compile(r"^\d+"), df)


def test_strings(df1):
    """Test _select_columns function on strings."""
    assert _select_columns("id", df1) == ["id"]
    assert _select_columns("*type*", df1) == [
        "type",
        "type1",
        "type2",
        "type3",
    ]


def test_slice(df1):
    """Test _select_columns function on slices."""
    assert_index_equal(
        _select_columns(slice("code", "code2"), df1),
        df1.loc[:, slice("code", "code2")].columns,
    )
    assert_index_equal(
        _select_columns(slice("code2", None), df1),
        df1.loc[:, slice("code2", None)].columns,
    )
    assert_index_equal(
        _select_columns(slice(None, "code2"), df1),
        df1.loc[:, slice(None, "code2")].columns,
    )
    assert_index_equal(_select_columns(slice(None, None), df1), df1.columns)
    assert_index_equal(
        _select_columns(slice(None, None, 2), df1),
        df1.loc[:, slice(None, None, 2)].columns,
    )


def test_callable_data_type(df1):
    """
    Test _select_columns function on callables,
    specifically for data type checks.
    """
    assert_index_equal(
        _select_columns(pd.api.types.is_integer_dtype, df1),
        df1.select_dtypes(int).columns,
    )
    assert_index_equal(
        _select_columns(pd.api.types.is_float_dtype, df1),
        df1.select_dtypes(float).columns,
    )
    assert_index_equal(
        _select_columns(pd.api.types.is_numeric_dtype, df1),
        df1.select_dtypes("number").columns,
    )
    assert_index_equal(
        _select_columns(pd.api.types.is_categorical_dtype, df1),
        df1.select_dtypes("category").columns,
    )
    assert_index_equal(
        _select_columns(pd.api.types.is_datetime64_dtype, df1),
        df1.select_dtypes(np.datetime64).columns,
    )
    assert_index_equal(
        _select_columns(pd.api.types.is_object_dtype, df1),
        df1.select_dtypes("object").columns,
    )


def test_callable_string_methods(df1):
    """
    Test _select_columns function on callables,
    specifically for column name checks.
    """
    assert_index_equal(
        _select_columns(lambda x: x.name.startswith("type"), df1),
        df1.filter(like="type").columns,
    )
    assert_index_equal(
        _select_columns(lambda x: x.name.endswith(("1", "2", "3")), df1),
        df1.filter(regex=r"\d$").columns,
    )
    assert_index_equal(
        _select_columns(lambda x: "d" in x.name, df1),
        df1.filter(regex="d").columns,
    )
    assert_index_equal(
        _select_columns(
            lambda x: x.name.startswith("code") and x.name.endswith("1"), df1
        ),
        df1.filter(regex=r"code.*1$").columns,
    )
    assert_index_equal(
        _select_columns(
            lambda x: x.name.startswith("code") or x.name.endswith("1"), df1
        ),
        df1.filter(regex=r"^code.*|.*1$").columns,
    )


def test_callable_computations(df1):
    """
    Test _select_columns function on callables,
    specifically for computations.
    """
    assert_index_equal(
        _select_columns(lambda x: x.isna().any(), df1),
        df1.columns[df1.isna().any().array],
    )


def test_regex(df1):
    """Test _select_columns function on regular expressions."""
    assert _select_columns(re.compile(r"\d$"), df1) == list(
        df1.filter(regex=r"\d$").columns
    )
    assert _select_columns(patterns(r"\d$"), df1) == list(
        df1.filter(regex=r"\d$").columns
    )


def test_list_various(df1):
    """Test _select_columns function on list type."""

    assert _select_columns(["id", "Name"], df1) == ["id", "Name"]
    assert _select_columns(["id", "code*"], df1) == list(
        df1.filter(regex="^id|^code").columns
    )
    assert_index_equal(
        pd.Index(
            _select_columns(["id", "code*", slice("code", "code2")], df1)
        ),
        df1.filter(regex="^(id|code)").columns,
    )
