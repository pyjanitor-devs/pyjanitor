import pytest
from hypothesis import given

from janitor.utils import check_column


@pytest.mark.functions
def test_rename_columns(dataframe):
    df = dataframe.clean_names().rename_columns(
        {"a": "index", "bell_chart": "chart"}
    )
    assert set(df.columns) == set(
        ["index", "chart", "decorated_elephant", "animals@#$%^", "cities"]
    )
    assert "a" not in set(df.columns)


@pytest.mark.functions
def test_rename_columns_absent_column(dataframe):
    """
    rename_column should raise an error if the column is absent.
    """
    df = dataframe.copy()
    with pytest.raises(ValueError):
        df.clean_names().rename_columns({"a": "index", "bb": "chart"})

    assert set(df.columns) == set(dataframe.columns)


@pytest.mark.functions
def test_check_column(dataframe):
    """
    rename_column should return true if column exist
    """
    assert check_column(dataframe, ["a"]) is None


@pytest.mark.functions
def test_check_column_absent_column(dataframe):
    """
    rename_column should raise an error if the column is absent.
    """
    with pytest.raises(ValueError):
        check_column(dataframe, ["b"])
