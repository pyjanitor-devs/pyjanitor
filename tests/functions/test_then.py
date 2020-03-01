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


@pytest.mark.functions
def test_then_column_names(dataframe):
    df = dataframe.then(remove_first_two_letters_from_col_names)
    cols = tuple(df.columns)
    assert cols == ("", "ll__Chart", "corated-elephant", "imals@#$%^", "ties")


@pytest.mark.functions
def test_then_remove_rows(dataframe):
    df = dataframe.then(remove_rows_3_and_4)
    rows = tuple(df.index)
    assert rows == (0, 1, 2, 5, 6, 7, 8)
