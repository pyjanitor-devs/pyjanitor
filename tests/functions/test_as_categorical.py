import pandas as pd
import pytest

from janitor.functions import as_categorical


def test_check_type_dataframe():
    "Raise TypeError if `df` is not a dataframe."
    df = pd.Series([60, 70])
    with pytest.raises(TypeError):
        as_categorical(df)


def test_check_type_column_names():
    "Raise TypeError if `column_names` is not a list type."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(TypeError):
        as_categorical(df=df, column_names=1)


def test_check_type_categories():
    "Raise TypeError if `categories` is not a list type."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(TypeError):
        as_categorical(df=df, categories=1)


def test_check_type_ordered():
    "Raise TypeError if `ordered` is not a string/list type."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(TypeError):
        as_categorical(df=df, ordered=1)


def test_check_presence_column_names():
    "Raise ValueError if `column_names` is not present in dataframe."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(ValueError):
        as_categorical(df=df, column_names="col")


def test_check_categories_if_column_names_None():
    "Raise ValueError if `column_names` is None and `categories` is a list."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(ValueError):
        as_categorical(df=df, categories=[70, 60])


def test_check_ordered_if_column_names_None():
    "Raise ValueError if `column_names` is None and `ordered` is a list."
    df = pd.DataFrame({"col1": [60, 70]})
    with pytest.raises(ValueError):
        as_categorical(df=df, ordered=["sorted"])


def test_check_categories_if_column_names_gt_1():
    """
    Raise ValueError if `column_names` is a list, more than one,
    and not all subs in `categories` are lists.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        as_categorical(
            df=df, column_names=["col1", "col2"], categories=[70, 60]
        )


def test_check_categories_length_if_column_names_gt_1():
    """
    Raise ValueError if `column_names` is a list, more than one,
    and the length of `categories` does not match the length of
    `column_names`.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        as_categorical(
            df=df, column_names=["col1", "col2"], categories=[[70, 60]]
        )


def test_check_ordered_if_column_names_gt_1():
    """
    Raise ValueError if `column_names` is a list, more than one,
    and `ordered` is not a list.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        as_categorical(
            df=df,
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
        as_categorical(
            df=df,
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
        as_categorical(df=df, column_names="col1", categories=[[70, 60]])


def test_check_ordered_if_column_names_eq_1():
    """
    Raise ValueError if a single `column_names` is provided,
    and `ordered` is a list.
    """
    df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})
    with pytest.raises(ValueError):
        as_categorical(
            df=df,
            column_names="col1",
            categories=[70, 60],
            ordered=["sorted", "appearance"],
        )
