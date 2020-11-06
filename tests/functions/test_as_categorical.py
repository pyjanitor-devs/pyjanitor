import pandas as pd
import numpy as np
import pytest

from pandas.api.types import CategoricalDtype
from pandas.testing import assert_frame_equal


def test_check_type_column_names():
    "Raise TypeError if `column_names` is not a list type."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(TypeError):
        df.as_categorical(column_names=1)


def test_check_type_categories():
    "Raise TypeError if `categories` is not a list type."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(TypeError):
        df.as_categorical(categories=1)


def test_check_type_ordered():
    "Raise TypeError if `ordered` is not a string/list type."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(TypeError):
        df.as_categorical(ordered=1)


def test_check_ordered_not_appearance_or_sort():
    "Raise ValueError if `ordered` is not either 'appearance' or 'sort'."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(ValueError):
        df.as_categorical(ordered="rar")


def test_check_presence_column_names():
    "Raise ValueError if `column_names` is not present in dataframe."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(ValueError):
        df.as_categorical(column_names="col")


def test_check_categories_if_column_names_None():
    "Raise ValueError if `column_names` is None and `categories` is a list."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(ValueError):
        df.as_categorical(categories=[70, 60])


def test_check_ordered_if_column_names_None():
    "Raise ValueError if `column_names` is None and `ordered` is a list."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(ValueError):
        df.as_categorical(ordered=["sorted"])


def test_check_categories_if_column_names_gt_1():
    """
    Raise ValueError if `column_names` is a list, more than one,
    and not all subs in `categories` are lists.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        df.as_categorical(column_names=["col1", "col2"], categories=[70, 60])


def test_check_categories_length_if_column_names_gt_1():
    """
    Raise ValueError if `column_names` is a list, more than one,
    and the length of `categories` does not match the length of
    `column_names`.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        df.as_categorical(column_names=["col1", "col2"], categories=[[70, 60]])


def test_check_ordered_if_column_names_gt_1():
    """
    Raise ValueError if `column_names` is a list, more than one,
    and `ordered` is not a list.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        df.as_categorical(
            column_names=["col1", "col2"],
            categories=[[70, 60], [80, 90]],
            ordered="sorted",
        )


def test_check_ordered_length_if_column_names_gt_1():
    """
    Raise ValueError if `column_names` is a list, more than one,
    and the length of `ordered` does not match `column_names` length.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        df.as_categorical(
            column_names=["col1", "col2"],
            categories=[[70, 60], [80, 90]],
            ordered=["sorted"],
        )


def test_check_categories_if_column_names_eq_1():
    """
    Raise ValueError if a single `column_names` is provided,
    and there are sub-lists within `categories`.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        df.as_categorical(column_names="col1", categories=[[70, 60]])


def test_check_ordered_if_column_names_eq_1():
    """
    Raise ValueError if a single `column_names` is provided,
    and `ordered` is a list.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        df.as_categorical(
            column_names="col1",
            categories=[70, 60],
            ordered=["sorted", "appearance"],
        )


test_df = [
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
        ).astype("category"),
        None,
        None,
        None,
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
                "col1": CategoricalDtype(categories=[2, 1, 3], ordered=True),
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
                        ]
                    ),
                    ordered=True,
                ),
            }
        ),
        None,
        None,
        "appearance",
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
                "col1": CategoricalDtype(categories=[1, 2, 3], ordered=True),
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
                        ]
                    ),
                    ordered=True,
                ),
            }
        ),
        None,
        None,
        "sort",
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
                "col1": CategoricalDtype(categories=[1.0, 2.0, 3.0                                                      ], ordered=True),
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
        None,
        None,
        "sort",
    ),
]


@pytest.mark.parametrize(
    "df_in,df_out,column_names,categories,ordered", test_df
)
def test_various(df_in, df_out, column_names, categories, ordered):
    """
    Test output for various combinations.
    """
    result = df_in.as_categorical(
        column_names=column_names, categories=categories, ordered=ordered,
    )
    assert_frame_equal(result, df_out)


import janitor

df = pd.DataFrame(
    {
        "col1": [2, 1, 3, 1, np.nan],
        "col2": ["a", "b", "c", "d", "a"],
        "col3": pd.date_range("1/1/2020", periods=5),
    }
)


print(df)


print(df.as_categorical(ordered="sort").dtypes)

