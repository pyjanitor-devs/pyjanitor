import pandas as pd
import pytest

from janitor.functions.adorn import adorn_totals

# Sample data
data = {
    "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
    "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
    "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

df = pd.DataFrame(data)


@pytest.mark.functions
def test_adorn_totals_row():
    """
    Test that adorn_totals correctly adds a 'Total' row to the crosstab.
    """
    result = adorn_totals(df, "Category", "Subcategory", axis=0)

    assert (
        "Total" in result.index
    ), "The 'Total' row must be present in the crosstab."
    assert (
        result.loc["Total"].sum() == df["Value"].count()
    ), "The sum of the 'Total' row must match the total count of the values."


@pytest.mark.functions
def test_adorn_totals_column():
    """
    Test that adorn_totals correctly adds a 'Total' column to the crosstab.
    """
    result = adorn_totals(df, "Category", "Subcategory", axis=1)

    assert (
        "Total" in result.columns
    ), "The 'Total' column must be present in the crosstab."
    assert (
        result["Total"].sum() == df["Value"].count()
    ), "The sum of the 'Total' column must match the total count of the values."


@pytest.mark.functions
def test_adorn_totals_empty_df():
    """
    Test that adorn_totals works correctly with an empty DataFrame.
    """
    empty_df = pd.DataFrame(columns=["Category", "Subcategory", "Value"])
    result_row = adorn_totals(empty_df, "Category", "Subcategory", axis=0)
    result_col = adorn_totals(empty_df, "Category", "Subcategory", axis=1)

    assert (
        result_row.empty
    ), "The crosstab must be empty when an empty DataFrame is used."
    assert (
        result_col.empty
    ), "The crosstab must be empty when an empty DataFrame is used."


@pytest.mark.functions
def test_adorn_totals_invalid_axis():
    """
    Test that adorn_totals raises an error when an invalid axis is provided.
    """
    data = {
        "Category": ["A", "B", "C"],
        "Subcategory": ["X", "Y", "Z"],
        "Value": [1, 2, 3],
    }
    df = pd.DataFrame(data)

    with pytest.raises(ValueError, match="The 'axis' argument must be 0 .* 1"):
        adorn_totals(df, "Category", "Subcategory", axis=2)  # Invalid axis


@pytest.mark.functions
def test_adorn_totals_large_data():
    """
    Test that adorn_totals works correctly with a larger DataFrame.
    """
    large_data = {
        "Category": ["A"] * 1000 + ["B"] * 1000,
        "Subcategory": ["X"] * 500 + ["Y"] * 500 + ["X"] * 500 + ["Y"] * 500,
        "Value": list(range(2000)),
    }
    large_df = pd.DataFrame(large_data)
    result = adorn_totals(large_df, "Category", "Subcategory", axis=0)

    assert (
        "Total" in result.index
    ), "The 'Total' row must be present in the crosstab for a large DataFrame."
    assert result.loc["Total"].sum() == len(
        large_data["Value"]
    ), "The sum of the 'Total' row must match the total count of the values."
