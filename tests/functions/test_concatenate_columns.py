import pytest


@pytest.mark.functions
def test_concatenate_columns(dataframe):
    df = dataframe.concatenate_columns(
        columns=["a", "decorated-elephant"], sep="-", new_column_name="index"
    )
    assert "index" in df.columns
