import pytest


@pytest.mark.functions
def test_deconcatenate_column(dataframe):
    df_orig = dataframe.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    )
    index_original = list(df_orig.columns).index("index")
    index_next = index_original + 1
    df = df_orig.deconcatenate_column(
        column_name="index",
        new_column_names=["col1", "col2"],
        sep="-",
        preserve_position=False,
    )
    assert "col1" in df.columns
    assert "col2" in df.columns
    # Test for `preserve_position` kwarg
    df = df_orig.deconcatenate_column(
        column_name="index",
        new_column_names=["col1", "col2"],
        sep="-",
        preserve_position=True,
    )
    assert "index" not in df.columns, "column_name not dropped"
    assert (
        list(df.columns).index("col1") == index_original
    ), "Position not preserved"
    assert (
        list(df.columns).index("col2") == index_next
    ), "Position not preserved"
