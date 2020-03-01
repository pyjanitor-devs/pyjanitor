import pytest
from hypothesis import given  # noqa: F401


@pytest.mark.functions
def test_rename_column(dataframe):
    df = dataframe.clean_names().rename_column("a", "index")
    assert set(df.columns) == set(
        ["index", "bell_chart", "decorated_elephant", "animals@#$%^", "cities"]
    )
    assert "a" not in set(df.columns)


@pytest.mark.functions
def test_rename_column_absent_column(dataframe):
    """
    rename_column should raise an error if the column is absent.
    """
    with pytest.raises(ValueError):
        dataframe.clean_names().rename_column("bb", "index")
