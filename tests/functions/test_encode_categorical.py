import numpy as np
import pandas as pd
import pytest
from hypothesis import given
from pandas.api.types import CategoricalDtype
from pandas.testing import assert_frame_equal

from janitor import AsCategorical
from janitor.errors import JanitorError
from janitor.testing_utils.strategies import (
    categoricaldf_strategy,
    df_strategy,
)


@pytest.mark.functions
@given(df=categoricaldf_strategy())
def test_encode_categorical(df):
    df = df.encode_categorical("names")
    assert df["names"].dtypes == "category"


@pytest.mark.functions
@given(df=df_strategy())
def test_encode_categorical_missing_column(df):
    """
    Raise JanitorError for missing columns
    when only one arguments is provided to
    `column_names`.
    """
    with pytest.raises(JanitorError):
        df.encode_categorical("aloha")


@pytest.mark.functions
@given(df=df_strategy())
def test_encode_categorical_missing_columns(df):
    """
    Raise JanitorError for missing columns
    when the number of arguments to `column_names`
    is more than one.
    """
    with pytest.raises(JanitorError):
        df.encode_categorical(["animals@#$%^", "cities", "aloha"])


@pytest.mark.functions
@given(df=df_strategy())
def test_encode_categorical_invalid_input(df):
    """
    Raise JanitorError for wrong input type
    for `column_names`.
    """
    with pytest.raises(JanitorError):
        df.encode_categorical(1)


@pytest.fixture
def df_categorical():
    return pd.DataFrame(
        {
            "col1": [2, 1, 3, 1],
            "col2": ["a", "b", "c", "d"],
            "col3": pd.date_range("1/1/2020", periods=4),
        }
    )


def test_encode_categorical_multiple_column_names(df_categorical):
    """
    Test output when more than one column is provided
    to `column_names`.
    """
    result = df_categorical.astype({"col1": "category", "col2": "category"})
    assert_frame_equal(
        df_categorical.encode_categorical(column_names=["col1", "col2"]),
        result,
    )


def test_both_column_names_kwargs(df_categorical):
    """
    Raise Error if both `column_names`
    and kwargs are provided.
    """
    with pytest.raises(ValueError):
        df_categorical.encode_categorical(
            column_names=["col1", "col2"], col1=(None, "sort")
        )


def test_check_presence_column_names_in_kwargs(df_categorical):
    """
    Raise ValueError if column names in `kwargs`
    do not exist in the dataframe.
    """
    with pytest.raises(ValueError):
        df_categorical.encode_categorical(col_1=(None, "sort"))


def test_check_type_tuple_in_kwargs(df_categorical):
    """
    Raise TypeError if the categories, order pairing
    in `kwargs` is not a tuple.
    """
    with pytest.raises(TypeError):
        df_categorical.encode_categorical(col1=[None, "sort"])


def test_categories_type_in_kwargs(df_categorical):
    """
    Raise TypeError if the wrong argument is supplied to
    the `categories` parameter in kwargs.
    """
    with pytest.raises(TypeError):
        df_categorical.encode_categorical(col1=({1: 2, 3: 3}, None))


def test_order_type_in_kwargs(df_categorical):
    """
    Raise TypeError if the wrong argument is supplied to
    the `order` parameter in kwargs.
    """
    with pytest.raises(TypeError):
        df_categorical.encode_categorical(col1=({1, 2, 3, 3}, {"sort"}))


def test_order_wrong_option_in_kwargs(df_categorical):
    """
    Raise ValueError if the value supplied to the `order`
    parameter in kwargs is not one of None, 'sort', or 'appearance'.
    """
    with pytest.raises(ValueError):
        df_categorical.encode_categorical(col1=({1, 2, 3, 3}, "sorted"))


as_categorical_tuple_length = [
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ),
        (None,),
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ),
        (None, "sort", "appearance"),
    ),
]


@pytest.mark.parametrize(
    "df,as_categorical_tuple", as_categorical_tuple_length
)
def test_tuple_length_in_kwargs(df, as_categorical_tuple):
    """
    Raise ValueError if the length of the tuple
    in kwargs is not equal to 2.
    """
    with pytest.raises(ValueError):
        df.encode_categorical(col1=as_categorical_tuple)


test_various_df = [
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ),
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ).astype({"col1": "category", "col2": "category"}),
        {
            "col1": AsCategorical(categories=None, order=None),
            "col2": AsCategorical(order=None, categories=None),
        },
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ),
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ).astype({"col1": "category", "col2": "category"}),
        {"col1": (None, None), "col2": (None, None)},
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ),
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ).astype(
            {
                "col3": "category",
                "col1": CategoricalDtype(categories=[1, 2, 3], ordered=True),
            }
        ),
        {"col3": (None, None), "col1": (None, "sort")},
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, np.nan],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ),
        pd.DataFrame(
            {
                "col1": [2.0, 1, 3, np.nan],
                "col2": ["a", "b", "c", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ).astype(
            {
                "col3": "category",
                "col1": CategoricalDtype(categories=[1, 2.0, 3], ordered=True),
            }
        ),
        {"col3": (None, None), "col1": (None, "sort")},
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, np.nan],
                "col2": ["b", "c", "a", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ),
        pd.DataFrame(
            {
                "col1": [2.0, 1, 3, np.nan],
                "col2": ["b", "c", "a", "d"],
                "col3": pd.date_range("1/1/2020", periods=4),
            }
        ).astype(
            {
                "col2": CategoricalDtype(
                    categories=["b", "c", "a", "d"], ordered=True
                ),
                "col1": CategoricalDtype(categories=[2.0, 1, 3], ordered=True),
            }
        ),
        {"col2": (None, "appearance"), "col1": (None, "appearance")},
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1, np.nan],
                "col2": ["a", "b", "c", "d", "a"],
                "col3": pd.date_range("1/1/2020", periods=5),
            }
        ),
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1, np.nan],
                "col2": ["a", "b", "c", "d", "a"],
                "col3": pd.date_range("1/1/2020", periods=5),
            }
        ).astype(
            {
                "col1": CategoricalDtype(
                    categories=[2.0, 1.0, 3.0], ordered=True
                ),
                "col2": CategoricalDtype(
                    categories=["a", "b", "c", "d"], ordered=False
                ),
            }
        ),
        {
            "col2": (["a", "b", "c", "d"], None),
            "col1": ([2.0, 1.0, 3.0], "appearance"),
        },
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1, np.nan],
                "col2": ["a", "b", "c", "d", "a"],
                "col3": pd.date_range("1/1/2020", periods=5),
            }
        ),
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1, np.nan],
                "col2": ["a", "b", "c", "d", "a"],
                "col3": pd.date_range("1/1/2020", periods=5),
            }
        ).astype(
            {
                "col1": CategoricalDtype(
                    categories=[2.0, 1.0, 3.0], ordered=True
                ),
                "col2": CategoricalDtype(
                    categories=["a", "b", "c", "d"], ordered=True
                ),
            }
        ),
        {
            "col2": (["a", "b", "c", "d"], "sort"),
            "col1": ([2.0, 1.0, 3.0], "appearance"),
        },
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1, np.nan],
                "col2": ["a", "b", "c", "d", "a"],
                "col3": pd.date_range("1/1/2020", periods=5),
            }
        ),
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1, np.nan],
                "col2": ["a", "b", "c", "d", "a"],
                "col3": pd.date_range("1/1/2020", periods=5),
            }
        ).astype(
            {
                "col1": CategoricalDtype(
                    categories=[1.0, 2.0, 3.0], ordered=True
                ),
                "col2": CategoricalDtype(
                    categories=["a", "b", "c", "d"], ordered=True
                ),
                "col3": CategoricalDtype(
                    categories=pd.to_datetime(
                        [
                            "2020-01-01",
                            "2020-01-02",
                            "2020-01-03",
                            "2020-01-04",
                            "2020-01-05",
                        ]
                    ),
                    ordered=True,
                ),
            }
        ),
        {
            "col3": (None, "sort"),
            "col1": (None, "sort"),
            "col2": (None, "appearance"),
        },
    ),
]


@pytest.mark.parametrize("df_in, df_out, kwargs", test_various_df)
def test_various(df_in, df_out, kwargs):
    """
    Test output for various combinations.
    """
    result = df_in.encode_categorical(**kwargs,)
    assert_frame_equal(result, df_out)


df_warnings = [
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1, np.nan],
                "col2": ["a", "b", "c", "d", "a"],
                "col3": pd.date_range("1/1/2020", periods=5),
            }
        ),
        {
            "col1": ([4, 5, 6], "appearance"),
            "col2": (["a", "b", "c", "d"], "sort"),
        },
    ),
    (
        pd.DataFrame(
            {
                "col1": [2, 1, 3, 1, np.nan],
                "col2": ["a", "b", "c", "d", "a"],
                "col3": pd.date_range("1/1/2020", periods=5),
            }
        ),
        {
            "col1": ([2.0, 1.0, 3.0], "appearance"),
            "col2": (["a", "b", "c"], "sort"),
        },
    ),
]


@pytest.mark.parametrize("df_in,kwargs", df_warnings)
def test_warnings(df_in, kwargs):
    """
    Test that warnings are raised if `categories` is provided, and
    the categories do not match the unique values in the column, or
    some values in the column are missing in `categories`.
    """

    with pytest.warns(UserWarning):
        df_in.encode_categorical(**kwargs)


df = pd.DataFrame(
    {
        "col1": [2, 1, 3, 1, np.nan],
        "col2": ["a", "b", "c", "d", "a"],
        "col3": pd.date_range("1/1/2020", periods=5),
    }
)

result = df.encode_categorical(
    col1=([2.0, 1.0, 3.0], "appearance"), col2=(["a", "b", "c"], "sort")
)

print(df, end="\n\n")

print(result)
