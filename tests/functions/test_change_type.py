import pytest


@pytest.mark.functions
def test_change_type(dataframe):
    df = dataframe.change_type(column="a", dtype=float)
    assert df["a"].dtype == float
