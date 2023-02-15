import numpy as np
import pandas as pd
import datetime
import pytest
from hypothesis import given
from hypothesis import settings
from pandas.testing import assert_frame_equal

from janitor.testing_utils.strategies import (
    categoricaldf_strategy,
    df_strategy,
)


@pytest.fixture
def df_checks():
    """pytest fixture"""
    return pd.DataFrame(
        [
            {"region": "Pacific", "2007": 1039, "2009": 2587},
            {"region": "Southwest", "2007": 51, "2009": 176},
            {"region": "Rocky Mountains and Plains", "2007": 200, "2009": 338},
        ]
    )


@pytest.mark.functions
@given(df=categoricaldf_strategy())
def test_encode_categorical(df):
    df = df.encode_categorical("names")
    assert df["names"].dtypes == "category"


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_encode_categorical_missing_column(df):
    """
    Raise KeyError for missing columns
    when only one arguments is provided to
    `column_names`.
    """
    with pytest.raises(KeyError):
        df.encode_categorical("aloha")


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_encode_categorical_missing_columns(df):
    """
    Raise KeyError for missing columns
    when the number of arguments to `column_names`
    is more than one.
    """
    with pytest.raises(KeyError):
        df.encode_categorical(["animals@#$%^", "cities", "aloha"])


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_encode_categorical_multiple_column_names(df):
    """
    Test output when more than one column is provided
    to `column_names`.
    """
    result = df.astype({"a": "category", "cities": "category"})
    assert_frame_equal(
        df.encode_categorical(column_names=["a", "cities"]),
        result,
    )


def test_both_column_names_kwargs(df_checks):
    """
    Raise Error if both `column_names`
    and kwargs are provided.
    """
    with pytest.raises(ValueError):
        df_checks.encode_categorical(column_names="region", region="sort")


def test_check_presence_column_names_in_kwargs(df_checks):
    """
    Raise ValueError if column names in `kwargs`
    do not exist in the dataframe.
    """
    with pytest.raises(ValueError):
        df_checks.encode_categorical(regon=None)


def test_categories_type_in_kwargs(df_checks):
    """
    Raise TypeError if the value provided is not array-like or a string.
    """
    with pytest.raises(TypeError):
        df_checks.encode_categorical(region=datetime.datetime(2017, 1, 1))


def test_categories_ndim_array_gt_1_in_kwargs(df_checks):
    """
    Raise ValueError if categories is provided, but is not a 1D array.
    """
    arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
    with pytest.raises(ValueError):
        df_checks.encode_categorical(region=arrays)


def test_categories_ndim_MultiIndex_gt_1_in_kwargs(df_checks):
    """
    Raise ValueError if categories is provided, but is not a 1D array.
    """
    arrays = [[1, 1, 2, 2], ["red", "blue", "red", "blue"]]
    arrays = pd.MultiIndex.from_arrays(arrays, names=("number", "color"))
    with pytest.raises(ValueError):
        df_checks.encode_categorical(region=arrays)


def test_categories_ndim_DataFrame_gt_1_in_kwargs(df_checks):
    """
    Raise ValueError if categories is provided, but is not a 1D array.
    """
    arrays = {"name": [1, 1, 2, 2], "number": ["red", "blue", "red", "blue"]}
    arrays = pd.DataFrame(arrays)
    with pytest.raises(ValueError):
        df_checks.encode_categorical(region=arrays)


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_categories_null_in_categories(df):
    """
    Raise ValueError if categories is provided, but has nulls.
    """
    with pytest.raises(ValueError):
        df.encode_categorical(a=[None, 2, 3])


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_non_unique_cat(df):
    """Raise ValueError if categories is provided, but is not unique."""
    with pytest.raises(ValueError):
        df.encode_categorical(a=[1, 2, 3, 3])


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_empty_cat(df):
    """Raise ValueError if empty categories is provided."""
    with pytest.raises(ValueError):
        df.encode_categorical(a=[])


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_empty_col(df):
    """
    Raise ValueError if categories is provided,
    but the relevant column is all nulls.
    """
    with pytest.raises(ValueError):
        df["col1"] = np.nan
        df.encode_categorical(col1=[1, 2, 3])


@pytest.mark.functions
@given(df=categoricaldf_strategy())
def test_warnings(df):
    """
    Test that warnings are raised if categories is provided, and
    the categories do not match the unique values in the column, or
    some values in the column are missing in the categories provided.
    """
    with pytest.warns(UserWarning):
        df.encode_categorical(
            numbers=[4, 5, 6], names=["John", "Mark", "Luke"]
        )


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_order_wrong_option_in_kwargs(df):
    """
    Raise ValueError if a string is provided, but is not
    one of None, 'sort', or 'appearance'.
    """
    with pytest.raises(ValueError):
        df.encode_categorical(a="sorted")


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_empty_col_sort(df):
    """
    Raise ValueError if a string is provided,
    but the relevant column is all nulls.
    """
    with pytest.raises(ValueError):
        df["col1"] = np.nan
        df.encode_categorical(col1="sort")


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_empty_col_appearance(df):
    """
    Raise ValueError if a string is provided,
    but the relevant column is all nulls.
    """
    with pytest.raises(ValueError):
        df["col1"] = np.nan
        df.encode_categorical(col1="appearance")


# directly comparing columns is safe -
# if the columns have differing categories
# (especially for ordered True) it will fail.
# if both categories are unordered, then the
# order is not considered.
# comparing with assert_frame_equal fails
# for unordered categoricals, as internally
# the order of the categories are compared
# which is irrelevant for unordered categoricals


@pytest.mark.functions
@given(df=categoricaldf_strategy())
@settings(deadline=None)
def test_all_None(df):
    """
    Test output where value is None.
    """
    result = df.encode_categorical(names=None)

    expected = df.astype({"names": "category"})
    assert expected["names"].equals(result["names"])


@pytest.mark.functions
@given(df=categoricaldf_strategy())
@settings(deadline=None)
def test_all_cat_None_1(df):
    """
    Test output where a string is provided.
    """
    result = df.encode_categorical(names="sort")
    categories = pd.CategoricalDtype(
        categories=df.names.factorize(sort=True)[-1], ordered=True
    )
    expected = df.astype({"names": categories})
    assert expected["names"].equals(result["names"])


@pytest.mark.functions
@given(df=categoricaldf_strategy())
def test_all_cat_None_2(df):
    """
    Test output where a string is provided.
    """
    result = df.encode_categorical(names="appearance")
    categories = pd.CategoricalDtype(
        categories=df.names.factorize(sort=False)[-1], ordered=True
    )
    expected = df.astype({"names": categories})
    assert expected["names"].equals(result["names"])


@pytest.mark.functions
@given(df=categoricaldf_strategy())
@settings(deadline=None)
def test_all_cat_not_None(df):
    """
    Test output where categories is provided.
    """
    result = df.encode_categorical(numbers=np.array([3, 1, 2]))
    categories = pd.CategoricalDtype(categories=[3, 1, 2], ordered=True)
    expected = df.astype({"numbers": categories})
    assert expected["numbers"].equals(result["numbers"])
