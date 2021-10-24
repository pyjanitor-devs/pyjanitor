import datetime
import re

import numpy as np
import pandas as pd
import pytest

from janitor import patterns
from janitor.functions.utils import _select_column_names


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


@pytest.fixture
def df_tuple():
    frame = pd.DataFrame(
        {
            "A": {0: "a", 1: "b", 2: "c"},
            "B": {0: 1, 1: 3, 2: 5},
            "C": {0: 2, 1: 4, 2: 6},
        }
    )
    frame.columns = [list("ABC"), list("DEF")]
    return frame


def test_type(df):
    """Raise TypeError if `columns_to_select` is the wrong type."""
    with pytest.raises(TypeError):
        _select_column_names(2.5, df)
    with pytest.raises(TypeError):
        _select_column_names(1, df)
    with pytest.raises(TypeError):
        _select_column_names([3, "id"], df)


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


def test_strings_do_not_exist(df):
    """
    Raise KeyError if `columns_to_select` is a string
    and does not exist in the dataframe's columns.
    """
    with pytest.raises(KeyError):
        _select_column_names("word", df)
    with pytest.raises(KeyError):
        _select_column_names("*starter", df)


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


def test_callable(df):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and at lease one Series has a wrong data type
    that makes the callable unapplicable.
    """
    with pytest.raises(TypeError):
        _select_column_names(object, df)


@pytest.mark.xfail(reason="Indexing in Pandas is possible with a Series.")
def test_callable_returns_Series(df):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and returns a Series.
    """
    with pytest.raises(ValueError):
        _select_column_names(lambda x: x + 1, df)


def test_callable_no_match(df):
    """
    Raise ValueError if `columns_to_select` is a callable,
    and no value is returned.
    """
    with pytest.raises(ValueError):
        _select_column_names(pd.api.types.is_float_dtype, df)

    with pytest.raises(ValueError):
        _select_column_names(lambda x: "Date" in x.name, df)


def test_regex_presence(df):
    """
    Raise KeyError if `columns_to_select` is a regex
    and none of the column names match.
    """
    with pytest.raises(KeyError):
        _select_column_names(re.compile(r"^\d+"), df)


def test_tuple_presence(df_tuple):
    """
    Raise KeyError if `columns_to_select` is a tuple
    and no match is returned.
    """
    with pytest.raises(KeyError):
        _select_column_names(("A", "C"), df_tuple)


def test_strings(df1):
    """Test _select_column_names function on strings."""
    assert _select_column_names("id", df1) == ["id"]
    assert _select_column_names("*type*", df1) == [
        "type",
        "type1",
        "type2",
        "type3",
    ]


def test_slice(df1):
    """Test _select_column_names function on slices."""
    assert (
        _select_column_names(slice("code", "code2"), df1)
        == df1.loc[:, slice("code", "code2")].columns.tolist()
    )

    assert (
        _select_column_names(slice("code2", None), df1)
        == df1.loc[:, slice("code2", None)].columns.tolist()
    )

    assert (
        _select_column_names(slice(None, "code2"), df1)
        == df1.loc[:, slice(None, "code2")].columns.tolist()
    )

    assert _select_column_names(slice(None, None), df1) == df1.columns.tolist()
    assert (
        _select_column_names(slice(None, None, 2), df1)
        == df1.loc[:, slice(None, None, 2)].columns.tolist()
    )


def test_callable_data_type(df1):
    """
    Test _select_column_names function on callables,
    specifically for data type checks.
    """
    assert (
        _select_column_names(pd.api.types.is_integer_dtype, df1)
        == df1.select_dtypes(int).columns.tolist()
    )

    assert (
        _select_column_names(pd.api.types.is_float_dtype, df1)
        == df1.select_dtypes(float).columns.tolist()
    )

    assert (
        _select_column_names(pd.api.types.is_numeric_dtype, df1)
        == df1.select_dtypes("number").columns.tolist()
    )

    assert (
        _select_column_names(pd.api.types.is_categorical_dtype, df1)
        == df1.select_dtypes("category").columns.tolist()
    )

    assert (
        _select_column_names(pd.api.types.is_datetime64_dtype, df1)
        == df1.select_dtypes(np.datetime64).columns.tolist()
    )

    assert (
        _select_column_names(pd.api.types.is_object_dtype, df1)
        == df1.select_dtypes("object").columns.tolist()
    )


def test_callable_string_methods(df1):
    """
    Test _select_column_names function on callables,
    specifically for column name checks.
    """
    assert _select_column_names(
        lambda x: x.name.startswith("type"), df1
    ) == list(df1.filter(like="type").columns)

    assert _select_column_names(
        lambda x: x.name.endswith(("1", "2", "3")), df1
    ) == list(df1.filter(regex=r"\d$").columns)

    assert _select_column_names(lambda x: "d" in x.name, df1) == list(
        df1.filter(regex="d").columns
    )

    assert _select_column_names(
        lambda x: x.name.startswith("code") and x.name.endswith("1"), df1
    ) == list(df1.filter(regex=r"code.*1$").columns)

    assert _select_column_names(
        lambda x: x.name.startswith("code") or x.name.endswith("1"), df1
    ) == list(df1.filter(regex=r"^code.*|.*1$").columns)


def test_callable_computations(df1):
    """
    Test _select_column_names function on callables,
    specifically for computations.
    """
    assert _select_column_names(lambda x: x.isna().any(), df1) == list(
        df1.columns[df1.isna().any().array]
    )


def test_regex(df1):
    """Test _select_column_names function on regular expressions."""
    assert _select_column_names(re.compile(r"\d$"), df1) == list(
        df1.filter(regex=r"\d$").columns
    )
    assert _select_column_names(patterns(r"\d$"), df1) == list(
        df1.filter(regex=r"\d$").columns
    )


def test_tuple(df_tuple):
    """Test _select_column_names function on tuple."""
    assert _select_column_names(("A", "D"), df_tuple) == ("A", "D")


def test_list_various(df1):
    """Test _select_column_names function on list type."""

    assert _select_column_names(["id", "Name"], df1) == ["id", "Name"]
    assert _select_column_names(["id", "code*"], df1) == list(
        df1.filter(regex="^id|^code").columns
    )
    assert (
        _select_column_names(["id", "code*", slice("code", "code2")], df1)
        == df1.filter(regex="^(id|code)").columns.tolist()
    )
    assert _select_column_names(["id", "Name"], df1) == ["id", "Name"]


def test_list_boolean(df):
    """Test _select_column_names function on list of booleans."""
    booleans = [True, True, True, False, False, False, True, True, True]
    assert _select_column_names(booleans, df) == list(df.columns[booleans])
