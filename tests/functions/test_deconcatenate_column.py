import pytest


@pytest.mark.functions
def test_deconcatenate_column(dataframe):
    df = dataframe.concatenate_columns(
        columns=["a", "decorated-elephant"], sep="-", new_column_name="index"
    )
    df = df.deconcatenate_column(
        column="index", new_column_names=["A", "B"], sep="-"
    )
    assert "A" in df.columns
    assert "B" in df.columns
