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
        _select_columns(df, 2.5)
    with pytest.raises(TypeError):
        _select_columns(df, 1)
    with pytest.raises(TypeError):
        _select_columns(df, ("id", "M_start_date_1"))
    with pytest.raises(TypeError):
        _select_columns(df, [3, "id"])


def test_strings(df):
    """
    Raise NameError if `columns_to_select` is a string
    and does not exist in the dataframe's columns.
    """
    with pytest.raises(NameError):
        _select_columns(df, "word")
    with pytest.raises(NameError):
        _select_columns(df, "*starter")


def test_slice_dtypes(df):
    """
    Raise ValueError if `columns_to_select` is a slice instance
    and either the start value or the stop value is not a string,
    or the step value is not an integer.
    """
    with pytest.raises(ValueError):
        _select_columns(df, slice(1, "M_end_date_2"))
    with pytest.raises(ValueError):
        _select_columns(df, slice("id", 2))
    with pytest.raises(ValueError):
        _select_columns(df, slice("id", "M_end_date_2", "3"))


def test_slice_presence(df):
    """
    Raise ValueError if `columns_to_select` is a slice instance
    and either the start value or the end value is not present
    in the dataframe.
    """
    with pytest.raises(ValueError):
        _select_columns(df, slice("Id", "M_start_date_1"))
    with pytest.raises(ValueError):
        _select_columns(df, slice("id", "M_end_date"))


def test_callable(df):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and at lease one Series has a wrong data type
    that makes the callable unapplicable.
    """
    with pytest.raises(TypeError):
        _select_columns(df, object)


def test_callable_returns_Series(df):
    """
    Check that error is raised if `columns_to_select` is a
    callable, and returns a Series.
    """
    with pytest.raises(ValueError):
        _select_columns(df, lambda x: x + 1)


def test_callable_no_match(df):
    """
    Raise ValueError if `columns_to_select` is a callable, and
    no boolean results are returned, when the callable is
    applied to each series in the dataframe.
    """
    with pytest.raises(ValueError):
        _select_columns(df, pd.api.types.is_float_dtype)

    with pytest.raises(ValueError):
        _select_columns(df, lambda x: "Date" in x.name)


def test_regex_presence(df):
    """
    Raise ValueError if `columns_to_select` is a regex
    and none of the column names match.
    """
    with pytest.raises(ValueError):
        _select_columns(df, re.compile(r"^\d+"))


class Test_Columns_not_List_Various_Inputs:
    @pytest.fixture(autouse=True)
    def test_columns(self, df1):
        self.df = df1

    def test_strings(self):
        assert _select_columns(self.df, "id") == ["id"]
        assert _select_columns(self.df, "*type*") == [
            "type",
            "type1",
            "type2",
            "type3",
        ]

    def test_slice(self):
        assert_index_equal(
            _select_columns(self.df, slice("code", "code2")),
            self.df.loc[:, slice("code", "code2")].columns,
        )
        assert_index_equal(
            _select_columns(self.df, slice("code2", None)),
            self.df.loc[:, slice("code2", None)].columns,
        )
        assert_index_equal(
            _select_columns(self.df, slice(None, "code2")),
            self.df.loc[:, slice(None, "code2")].columns,
        )
        assert_index_equal(
            _select_columns(self.df, slice(None, None)), self.df.columns
        )
        assert_index_equal(
            _select_columns(self.df, slice(None, None, 2)),
            self.df.loc[:, slice(None, None, 2)].columns,
        )

    def test_callable_data_type(self):
        assert_index_equal(
            _select_columns(self.df, pd.api.types.is_integer_dtype),
            self.df.select_dtypes(int).columns,
        )
        assert_index_equal(
            _select_columns(self.df, pd.api.types.is_float_dtype),
            self.df.select_dtypes(float).columns,
        )
        assert_index_equal(
            _select_columns(self.df, pd.api.types.is_numeric_dtype),
            self.df.select_dtypes("number").columns,
        )
        assert_index_equal(
            _select_columns(self.df, pd.api.types.is_categorical_dtype),
            self.df.select_dtypes("category").columns,
        )
        assert_index_equal(
            _select_columns(self.df, pd.api.types.is_datetime64_dtype),
            self.df.select_dtypes(np.datetime64).columns,
        )
        assert_index_equal(
            _select_columns(self.df, pd.api.types.is_object_dtype),
            self.df.select_dtypes("object").columns,
        )

    def test_callable_string_methods(self):
        assert_index_equal(
            _select_columns(self.df, lambda x: x.name.startswith("type")),
            self.df.filter(like="type").columns,
        )
        assert_index_equal(
            _select_columns(
                self.df, lambda x: x.name.endswith(("1", "2", "3"))
            ),
            self.df.filter(regex=r"\d$").columns,
        )
        assert_index_equal(
            _select_columns(self.df, lambda x: "d" in x.name),
            self.df.filter(regex="d").columns,
        )
        assert_index_equal(
            _select_columns(
                self.df,
                lambda x: x.name.startswith("code") and x.name.endswith("1"),
            ),
            self.df.filter(regex=r"code.*1$").columns,
        )
        assert_index_equal(
            _select_columns(
                self.df,
                lambda x: x.name.startswith("code") or x.name.endswith("1"),
            ),
            self.df.filter(regex=r"^code.*|.*1$").columns,
        )

    def test_callable_computations(self):
        assert_index_equal(
            _select_columns(self.df, lambda x: x.isna().any()),
            self.df.columns[self.df.isna().any().array],
        )

    def test_regex(self):
        assert _select_columns(self.df, re.compile(r"\d$")) == list(
            self.df.filter(regex=r"\d$").columns
        )
        assert _select_columns(self.df, patterns(r"\d$")) == list(
            self.df.filter(regex=r"\d$").columns
        )


class Test_Columns_in_List_Various_Inputs:
    @pytest.fixture(autouse=True)
    def test_columns_in_list(self, df1):
        self.df = df1

    def test_list_various(self):
        assert _select_columns(self.df, ["id", "Name"]) == ["id", "Name"]
        assert _select_columns(self.df, ["id", "code*"]) == list(
            self.df.filter(regex="^id|^code").columns
        )
        assert_index_equal(
            pd.Index(
                _select_columns(
                    self.df, ["id", "code*", slice("code", "code2")]
                )
            ),
            self.df.filter(regex="^(id|code)").columns,
        )


df = pd.DataFrame(
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


result = _select_columns(["id", "code*", slice("code", "code2"), lambda x: x.name.startswith("ty")], df)
print(df, end="\n\n")
print(result)