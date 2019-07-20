import pytest


@pytest.mark.functions
def test_deconcatenate_column(dataframe):
    df_orig = dataframe.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    )
    df = df_orig.deconcatenate_column(
        column_name="index", new_column_names=["A", "B"], sep="-"
    )
    assert "A" in df.columns
    assert "B" in df.columns
