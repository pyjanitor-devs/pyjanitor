import pytest


@pytest.mark.functions
def test_deconcatenate_column_preserve_position(dataframe):
    df_original = dataframe.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    )
    index_original = list(df_original.columns).index("index")
    df = df_original.deconcatenate_column(
        column_name="index",
        new_column_names=["col1", "col2"],
        sep="-",
        preserve_position=True,
    )
    assert "index" not in df.columns, "column_name not dropped"
    assert "col1" in df.columns, "new column not present"
    assert "col2" in df.columns, "new column not present"
    assert (
        len(df_original.columns) + 1 == len(df.columns)
    ), 'Number of columns inconsistent'
    assert (
        list(df.columns).index("col1") == index_original
    ), "Position not preserved"
    assert (
        list(df.columns).index("col2") == index_original + 1
    ), "Position not preserved"
