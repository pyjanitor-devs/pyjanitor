"""Tests for concatenate_columns."""

import pandas as pd
import pytest

from janitor.errors import JanitorError


@pytest.mark.functions
def test_concatenate_columns(dataframe):
    """Basic test for concatenate_columns."""
    df = dataframe.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    )
    assert "index" in df.columns


@pytest.mark.functions
def test_concatenate_columns_null_values(missingdata_df):
    """Test for concatenating columns with null values.

    Null values ought to show up as "nan" in strings
    in the concatenated column.
    """
    df = missingdata_df.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
        ignore_empty=True,
    )
    expected_values = ["1.0-1", "2.0-2", "3"] * 3
    assert expected_values == df["index"].tolist()


@pytest.mark.functions
@pytest.mark.parametrize(
    "sep, ignore_empty, expected",
    [
        ("-", True, ["A--B-X", "A-B-X", "-Y--Z", ""]),
        ("-", False, ["A--B--X", "A-B--X", "-Y--Z-", "--"]),
        ("", True, ["A--BX", "A-BX", "-Y-Z", ""]),
    ],
)
def test_concatenate_preserves_field_contents(sep, ignore_empty, expected):
    """Embedded separators belong to the field even when empty fields are skipped."""
    df = pd.DataFrame(
        {
            "a": ["A--B", "A-B", "-Y-", ""],
            "b": ["", "", "Z", ""],
            "c": ["X", "X", "", ""],
        }
    )
    result = df.concatenate_columns(["a", "b", "c"], "id", sep, ignore_empty)
    assert result["id"].tolist() == expected


@pytest.mark.functions
@pytest.mark.parametrize("column_names", [["a"], []])
def test_concatenate_columns_errors(dataframe, column_names):
    """
    Test that an error is raised when less than two columns are specified.
    """
    with pytest.raises(JanitorError, match="At least two columns must be specified"):
        dataframe.concatenate_columns(
            column_names=column_names, new_column_name="index"
        )
