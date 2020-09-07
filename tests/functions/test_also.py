from unittest.mock import Mock
import pytest


def remove_first_two_letters_from_col_names(df):
    col_names = df.columns
    col_names = [name[2:] for name in col_names]
    df.columns = col_names
    return df


def remove_rows_3_and_4(df):
    df = df.drop(3, axis=0)
    df = df.drop(4, axis=0)
    return df


def drop_inplace(df):
    df.drop(columns=[df.columns[0]], inplace=True)


@pytest.mark.functions
def test_also_column_manipulation_no_change(dataframe):
    cols = tuple(dataframe.columns)
    df = dataframe.also(remove_first_two_letters_from_col_names)
    assert dataframe is df
    assert cols == tuple(df.columns)


@pytest.mark.functions
def test_also_remove_rows_no_change(dataframe):
    df = dataframe.also(remove_rows_3_and_4)
    rows = tuple(df.index)
    assert rows == (0, 1, 2, 3, 4, 5, 6, 7, 8)


@pytest.mark.functions
def test_also_runs_function(dataframe):
    method = Mock(return_value=None)
    df = dataframe.also(method)
    assert id(df) == id(dataframe)
    assert method.call_count == 1


@pytest.mark.functions
def test_also_drop_inplace(dataframe):
    cols = tuple(dataframe.columns)
    df = dataframe.also(drop_inplace)
    assert tuple(df.columns) == cols
