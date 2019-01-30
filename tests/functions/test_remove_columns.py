import pytest

from janitor.testing_utils.fixtures import dataframe


@pytest.mark.functions
def test_remove_columns(dataframe):
    df = dataframe.remove_columns(columns=["a"])
    assert len(df.columns) == 4
