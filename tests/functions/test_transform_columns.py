"""Tests for transform_columns."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_transform_columns(dataframe):
    """Checks in-place transformation of multiple columns is as expected."""
    df = (
        dataframe.add_column("another", 10)
        .add_column("column", 100)
        .transform_columns(["another", "column"], np.log10)
    )
    expected = pd.DataFrame(
        {"another": np.ones(len(df)), "column": np.ones(len(df)) * 2}
    )
    assert_frame_equal(df[["another", "column"]], expected)


@pytest.mark.functions
def test_transform_column_with_suffix(dataframe):
    """Checks `suffix` creates new columns as expected."""
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
    """Checks `new_column_names` creates new columns as expected."""
    df = (
        dataframe.add_column("another", 10)
        .add_column("column", 100)
        .transform_columns(
            ["another", "column"],
            np.log10,
            new_column_names={"another": "hello", "column": "world"},
        )
    )

    assert "hello" in df.columns
    assert "world" in df.columns
    assert "another" in df.columns
    assert "column" in df.columns


@pytest.mark.functions
def test_transform_column_with_incomplete_new_names(dataframe):
    """Use of `new_column_names` with additional columns (not in `column_names`
    should passthrough silently. Related to bug #1063.
    """
    df = (
        dataframe.add_column("another", 10)
        .add_column("column", 100)
        .transform_columns(
            ["another", "column"],
            np.log10,
            new_column_names={
                "another": "hello",
                "fakecol": "world",
            },
        )
    )

    assert "another" in df.columns
    assert "column" in df.columns
    assert "hello" in df.columns
    assert "world" not in df.columns


@pytest.mark.functions
def test_suffix_newname_validation(dataframe):
    """Check ValueError is raised when both suffix and new_column_names are
    provided."""
    with pytest.raises(
        ValueError,
        match="Only one of `suffix` or `new_column_names` should be specified",
    ):
        _ = (
            dataframe.add_column("another", 10)
            .add_column("column", 100)
            .transform_columns(
                ["another", "column"],
                np.log10,
                new_column_names={"another": "hello", "column": "world"},
                suffix="_log",
            )
        )
