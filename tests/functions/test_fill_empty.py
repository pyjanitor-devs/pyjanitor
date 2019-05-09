import pytest


@pytest.mark.functions
def test_fill_empty(null_df):
    df = null_df.fill_empty(column_names=["2"], value=3)
    assert set(df.loc[:, "2"]) == set([3])


@pytest.mark.functions
def test_fill_empty_column_string(null_df):
    df = null_df.fill_empty(column_names="2", value=3)
    assert set(df.loc[:, "2"]) == set([3])
