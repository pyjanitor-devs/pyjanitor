import pytest
import pandas as pd


@pytest.mark.functions
def test_fill_empty(null_df):
    df = null_df.fill_empty(column_names=["2"], value=3)
    assert set(df.loc[:, "2"]) == set([3])


@pytest.mark.functions
def test_fill_empty_column_string(null_df):
    df = null_df.fill_empty(column_names="2", value=3)
    assert set(df.loc[:, "2"]) == set([3])


@pytest.mark.functions
@pytest.mark.parametrize(
    "column_names",
    [
        (0, 1, "2", "3"),  # tuple
        [0, 1, "2", "3"],  # list
        {0, 1, "2", "3"},  # set
        ({0: 0, 1: 1, "2": "2", "3": "3"}).keys(),  # dict key
        ({0: 0, 1: 1, "2": "2", "3": "3"}).values(),  # dict value
        pd.Index([0, 1, "2", "3"]),  # Index
    ],
)
def test_column_names_iterable_type(null_df, column_names):
    result = null_df.fill_empty(column_names=column_names, value=3)
    excepted = null_df.fillna(3)

    assert result.equals(excepted)
