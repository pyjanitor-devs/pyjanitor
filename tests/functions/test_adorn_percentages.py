import pandas as pd
import pytest

from janitor.functions.adorn import adorn_percentages


@pytest.mark.functions
def test_adorn_percentages_row():
    """
    Test that adorn_percentages correctly calculates row percentages.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = adorn_percentages(
        df, "Category", "Subcategory", axis="row", fmt=True, include_ns=False
    )

    assert result.shape[0] == 3
    # 3 unique categories
    assert result.shape[1] > 1
    # Should have more than one column (including percentages)
    assert "%" in result.iloc[0, 1]
    # Check that the result contains percentages


@pytest.mark.functions
def test_adorn_percentages_col():
    """
    Test that adorn_percentages correctly calculates column percentages.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = adorn_percentages(
        df, "Category", "Subcategory", axis="col", fmt=True, include_ns=False
    )

    assert result.shape[0] == 3
    # 3 unique categories
    assert result.shape[1] > 1
    # Should have more than one column (including percentages)
    assert "%" in result.iloc[0, 1]
    # Check that the result contains percentages


@pytest.mark.functions
def test_adorn_percentages_all():
    """
    Test that adorn_percentages correctly calculates total (global) percentages.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = adorn_percentages(
        df, "Category", "Subcategory", axis="all", fmt=True, include_ns=False
    )

    assert result.shape[0] == 3
    # 3 unique categories
    assert result.shape[1] > 1
    # Should have more than one column (including percentages)
    assert "%" in result.iloc[0, 1]
    # Check that the result contains percentages


@pytest.mark.functions
def test_adorn_percentages_with_ns_row():
    """
    Test that adorn_percentages correctly calculates row percentages
    with raw counts.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = adorn_percentages(
        df, "Category", "Subcategory", axis="row", fmt=True, include_ns=True
    )

    assert result.shape[0] == 3
    # 3 unique categories
    assert result.shape[1] > 1
    # Should have more than one column (including percentages and raw counts)
    assert "(" in result.iloc[0, 1]
    # Check that raw counts are included


@pytest.mark.functions
def test_adorn_percentages_with_ns_col():
    """
    Test that adorn_percentages correctly calculates column percentages
    with raw counts.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = adorn_percentages(
        df, "Category", "Subcategory", axis="col", fmt=True, include_ns=True
    )

    assert result.shape[0] == 3
    # 3 unique categories
    assert result.shape[1] > 1
    # Should have more than one column (including percentages and raw counts)
    assert "(" in result.iloc[0, 1]
    # Check that raw counts are included


@pytest.mark.functions
def test_adorn_percentages_with_ns_all():
    """
    Test that adorn_percentages correctly calculates total (global) percentages
    with raw counts.
    """
    data = {
        "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
        "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
        "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    }
    df = pd.DataFrame(data)

    result = adorn_percentages(
        df, "Category", "Subcategory", axis="all", fmt=True, include_ns=True
    )

    assert result.shape[0] == 3
    # 3 unique categories
    assert result.shape[1] > 1
    # Should have more than one column (including percentages and raw counts)
    assert "(" in result.iloc[0, 1]
    # Check that raw counts are included


@pytest.mark.functions
def test_adorn_percentages_empty_pivot():
    """
    Test that adorn_percentages returns an empty DataFrame if the pivot is empty.
    """
    # DataFrame sans colonnes valides pour le pivot
    data = {"NonExistentColumn": [], "AnotherColumn": [], "Value": []}
    df = pd.DataFrame(data)

    # Appel de la fonction avec des colonnes inexistantes
    result = adorn_percentages(df, "NonExistentColumn", "AnotherColumn")

    # Vérifie que le résultat est un DataFrame vide
    assert result.empty, "Expected an empty DataFrame when pivot is empty."


@pytest.mark.functions
def test_adorn_percentages_invalid_axis():
    """
    Test that adorn_percentages raises a ValueError for an invalid axis argument.
    """
    data = {
        "Category": ["A", "B"],
        "Subcategory": ["X", "Y"],
        "Value": [10, 20],
    }
    df = pd.DataFrame(data)

    with pytest.raises(
        ValueError, match="The 'axis' argument must be 'row', 'col', or 'all'."
    ):
        adorn_percentages(df, "Category", "Subcategory", axis="invalid")
