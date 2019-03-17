import numpy as np
import pandas as pd
import pytest


@pytest.mark.functions
def test_transform_columns(dataframe):
    # replacing the data of the original column

    df = (
        dataframe.add_column("another", 10)
        .add_column("column", 100)
        .transform_columns(["another", "column"], np.log10)
    )
    expected = pd.DataFrame(
        {"another": np.ones(len(df)), "column": np.ones(len(df)) * 2}
    )
    pd.testing.assert_frame_equal(df[["another", "column"]], expected)


@pytest.mark.functions
def test_transform_column_with_suffix(dataframe):
    # creating a new destination column

    df = (
        dataframe.add_column("another", 10)
        .add_column("column", 100)
        .transform_columns(["another", "column"], np.log10, suffix="_log")
    )

    assert "another_log" in df.columns
    assert "column_log" in df.columns
    assert "another" in df.columns
    assert "column" in df.columns


@pytest.mark.functions
def test_transform_column_with_new_names(dataframe):
    # creating a new destination column
    df = (
        dataframe.add_column("another", 10)
        .add_column("column", 100)
        .transform_columns(
            ["another", "column"],
            np.log10,
            new_names={"another": "hello", "column": "world"},
        )
    )

    assert "hello" in df.columns
    assert "world" in df.columns
    assert "another" in df.columns
    assert "column" in df.columns


@pytest.mark.functions
def test_suffix_newname_validation(dataframe):
    with pytest.raises(ValueError):
        df = (
            dataframe.add_column("another", 10)
            .add_column("column", 100)
            .transform_columns(
                ["another", "column"],
                np.log10,
                new_names={"another": "hello", "column": "world"},
                suffix="_log",
            )
        )
