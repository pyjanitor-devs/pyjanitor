import pandas as pd
import pytest

from janitor.functions.adorn import tabyl


@pytest.mark.functions
def test_tabyl_basic_counts():
    """
    Test that tabyl correctly generates a crosstab with raw counts.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Region": [
            "North",
            "South",
            "East",
            "West",
            "North",
            "South",
            "East",
            "West",
            "North",
            "East",
        ],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = tabyl(
        df, "Category", "Subcategory", "Region", show_percentages=False
    )

    assert (
        result.shape[1] >= 5
    ), f"Expected at least 5 columns, got {result.shape[1]}"
    assert (
        result.iloc[:, 1:].sum().sum() == 10
    )  # The sum of the counts should be equal to 10


@pytest.mark.functions
def test_tabyl_with_percentages_row():
    """
    Test that tabyl correctly calculates percentages by row.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Region": [
            "North",
            "South",
            "East",
            "West",
            "North",
            "South",
            "East",
            "West",
            "North",
            "East",
        ],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = tabyl(
        df,
        "Category",
        "Subcategory",
        "Region",
        show_counts=False,
        show_percentages=True,
        percentage_axis="row",
    )

    result_numeric = result.applymap(
        lambda x: (
            float(x.strip("%")) / 100 if isinstance(x, str) and "%" in x else x
        )
    )
    assert (
        result_numeric.select_dtypes(include=["float", "int"]).min().min() >= 0
    )
    assert (
        result_numeric.select_dtypes(include=["float", "int"]).max().max() <= 1
    )


@pytest.mark.functions
def test_tabyl_with_percentages_col():
    """
    Test that tabyl correctly calculates percentages by column.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Region": [
            "North",
            "South",
            "East",
            "West",
            "North",
            "South",
            "East",
            "West",
            "North",
            "East",
        ],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = tabyl(
        df,
        "Category",
        "Subcategory",
        "Region",
        show_counts=False,
        show_percentages=True,
        percentage_axis="col",
    )

    result_numeric = result.applymap(
        lambda x: (
            float(x.strip("%")) / 100 if isinstance(x, str) and "%" in x else x
        )
    )
    assert (
        result_numeric.select_dtypes(include=["float", "int"]).min().min() >= 0
    )
    assert (
        result_numeric.select_dtypes(include=["float", "int"]).max().max() <= 1
    )


@pytest.mark.functions
def test_tabyl_with_percentages_all():
    """
    Test that tabyl correctly calculates total percentages.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Region": [
            "North",
            "South",
            "East",
            "West",
            "North",
            "South",
            "East",
            "West",
            "North",
            "East",
        ],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = tabyl(
        df,
        "Category",
        "Subcategory",
        "Region",
        show_counts=False,
        show_percentages=True,
        percentage_axis="all",
    )

    result_numeric = result.applymap(
        lambda x: (
            float(x.strip("%")) / 100 if isinstance(x, str) and "%" in x else x
        )
    )
    assert (
        result_numeric.select_dtypes(include=["float", "int"]).min().min() >= 0
    )
    assert (
        result_numeric.select_dtypes(include=["float", "int"]).max().max() <= 1
    )


@pytest.mark.functions
def test_tabyl_missing_col1():
    """
    Test that tabyl raises an error if col1 is missing from the DataFrame.
    """
    data = {"Category": ["A", "B"], "Subcategory": ["X", "Y"]}
    df = pd.DataFrame(data)

    with pytest.raises(
        ValueError, match="Column 'Region' is not in the DataFrame."
    ):
        tabyl(df, "Region")


@pytest.mark.functions
def test_tabyl_missing_col2():
    """
    Test that tabyl raises an error if col2 is missing from the DataFrame.
    """
    data = {"Category": ["A", "B"], "Subcategory": ["X", "Y"]}
    df = pd.DataFrame(data)

    with pytest.raises(
        ValueError, match="Column 'Value' is not in the DataFrame."
    ):
        tabyl(df, "Category", "Value")


@pytest.mark.functions
def test_tabyl_missing_col3():
    """
    Test that tabyl raises an error if col3 is missing from the DataFrame.
    """
    data = {"Category": ["A", "B"], "Subcategory": ["X", "Y"]}
    df = pd.DataFrame(data)

    with pytest.raises(
        ValueError, match="Column 'Region' is not in the DataFrame."
    ):
        tabyl(df, "Category", "Subcategory", "Region")


@pytest.mark.functions
def test_tabyl_single_column():
    """
    Test that tabyl works correctly with only col1 specified.
    """
    data = {"Category": ["A", "B", "A", "C", "B", "A", "C"]}
    df = pd.DataFrame(data)

    result = tabyl(df, "Category")
    assert result.shape[0] == 3  # Three unique values in 'Category'
    assert (
        result["count"].sum() == 7
    )  # Total count should match the number of rows


@pytest.mark.functions
def test_tabyl_invalid_percentage_axis():
    """
    Test that tabyl raises an error for invalid percentage_axis values.
    """
    data = {"Category": ["A", "B"], "Subcategory": ["X", "Y"]}
    df = pd.DataFrame(data)

    with pytest.raises(
        ValueError,
        match="`percentage_axis` must be one of 'row', 'col', or 'all'.",
    ):
        tabyl(
            df,
            "Category",
            "Subcategory",
            show_percentages=True,
            percentage_axis="invalid",
        )
