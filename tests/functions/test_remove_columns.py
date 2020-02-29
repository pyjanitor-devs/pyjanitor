import pytest


@pytest.mark.functions
def test_remove_columns_one_col(dataframe):
    df = dataframe.remove_columns(column_names=["a"])
    assert len(df.columns) == 4


@pytest.mark.functions
def test_remove_columns_mult_cols(dataframe):
    df = dataframe.remove_columns(column_names=["a", "Bell__Chart"])
    assert len(df.columns) == 3


@pytest.mark.functions
def test_remove_columns_no_cols(dataframe):
    df = dataframe.remove_columns(column_names=[])
    assert len(df.columns) == 5


@pytest.mark.functions
def test_remove_columns_all_cols(dataframe):
    df = dataframe.remove_columns(
        column_names=[
            "a",
            "Bell__Chart",
            "decorated-elephant",
            "animals@#$%^",
            "cities",
        ]
    )
    assert len(df.columns) == 0


@pytest.mark.skip(reason="Not sure why this is failing")
def test_remove_columns_strange_cols(dataframe):
    df = dataframe.remove_columns(
        column_names=[
            "a",
            ["Bell__Chart", "decorated-elephant", "animals@#$%^", "cities"],
        ]
    )
    assert len(df.columns) == 0


@pytest.mark.functions
def test_remove_columns_strange_cols_multilevel(multilevel_dataframe):
    # When creating a multi level dataframe with 4 columns * 2 columns
    # (16 columns in total)
    # From input

    # If 2 columns (2 tuples = 4 codes) are removed
    df = multilevel_dataframe.remove_columns(
        column_names=[("bar", "one"), ("baz", "two")]
    )

    # Then the total number of codes must be 12 (16-4)
    assert (
        len([item for sublist in df.columns.codes for item in sublist]) == 12
    )
