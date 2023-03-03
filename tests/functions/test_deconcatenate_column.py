import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor  # noqa: F401


@pytest.mark.functions
def test_deconcatenate_column_collection(dataframe: pd.DataFrame):
    column_names = ["a", "decorated-elephant", "cities"]

    lists = [dataframe[column_name] for column_name in column_names]

    elements = tuple(zip(*lists))

    concat_df = (
        dataframe.copy()
        .add_column("concatenated", elements)
        .remove_columns(column_names)
    )

    deconcat_df = (
        concat_df.deconcatenate_column(
            "concatenated", new_column_names=column_names
        )
        .remove_columns("concatenated")
        .reorder_columns(dataframe.columns)
    )

    assert_frame_equal(dataframe, deconcat_df)

    deconcat_df = concat_df.deconcatenate_column(
        "concatenated", new_column_names=column_names, preserve_position=True
    ).reorder_columns(dataframe.columns)

    assert_frame_equal(dataframe, deconcat_df)


@pytest.mark.functions
def test_deconcatenate_column_string_no_sep(dataframe):
    with pytest.raises(ValueError):
        df_orig = dataframe.concatenate_columns(
            column_names=["a", "decorated-elephant"],
            sep="-",
            new_column_name="index",
        )
        df = df_orig.deconcatenate_column(  # noqa: F841
            column_name="index", new_column_names=["A", "B"]
        )


@pytest.mark.functions
def test_deconcatenate_column_string(dataframe):
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


@pytest.mark.functions
def test_deconcatenate_column_preserve_position_string(dataframe):
    df_original = dataframe.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    )
    index_original = list(df_original.columns).index("index")
    new_column_names = ["col1", "col2"]
    df = df_original.deconcatenate_column(
        column_name="index",
        new_column_names=new_column_names,
        sep="-",
        preserve_position=True,
    )
    assert "index" not in df.columns, "column_name not dropped"
    assert "col1" in df.columns, "new column not present"
    assert "col2" in df.columns, "new column not present"
    assert len(df_original.columns) + 1 == len(
        df.columns
    ), "Number of columns inconsistent"
    assert (
        list(df.columns).index("col1") == index_original
    ), "Position not preserved"
    assert (
        list(df.columns).index("col2") == index_original + 1
    ), "Position not preserved"
    assert len(df.columns) == (
        len(df_original.columns) + len(new_column_names) - 1
    ), "Number of columns after deconcatenation is incorrect"


def test_deconcatenate_column_autoname(dataframe):
    df_original = dataframe.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    ).remove_columns(["a", "decorated-elephant"])

    df = df_original.deconcatenate_column(
        "index",
        sep="-",
        new_column_names=["a", "decorated-elephant"],
        autoname="col",
    )

    assert "col1" in df.columns
    assert "col2" in df.columns
    assert "a" not in df.columns
    assert "decorated-elephant" not in df.columns


data = {
    "a": [1, 2, 3] * 3,
    "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
    "decorated-elephant": [1, 2, 3] * 3,
    "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
    "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
}
