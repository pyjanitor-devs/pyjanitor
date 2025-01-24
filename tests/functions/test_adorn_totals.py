import pandas as pd
import pytest

from janitor.functions.adorn import adorn_totals

# Données d'exemple
data = {
    "Category": ["A", "A", "B", "B", "C", "C", "A", "B", "C", "A"],
    "Subcategory": ["X", "Y", "X", "Y", "X", "Y", "X", "Y", "X", "X"],
    "Value": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
}

df = pd.DataFrame(data)


@pytest.mark.functions
def test_adorn_totals_row():
    """
    Test que adorn_totals ajoute correctement une ligne 'Total' au tableau croisé.
    """
    result = adorn_totals(df, "Category", "Subcategory", axis=0)

    assert (
        "Total" in result.index
    ), "La ligne 'Total' doit être présente dans le tableau."
    assert (
        result.loc["Total"].sum() == df["Value"].count()
    ), "La somme de la ligne 'Total' doit correspondre au total des comptes."


@pytest.mark.functions
def test_adorn_totals_column():
    """
    Test que adorn_totals ajoute correctement une colonne 'Total' au tableau croisé.
    """
    result = adorn_totals(df, "Category", "Subcategory", axis=1)

    assert (
        "Total" in result.columns
    ), "La colonne 'Total' doit être présente dans le tableau."
    assert (
        result["Total"].sum() == df["Value"].count()
    ), "La somme de la colonne 'Total' doit correspondre au total des comptes."


@pytest.mark.functions
def test_adorn_totals_empty_df():
    """
    Test que adorn_totals fonctionne correctement avec un DataFrame vide.
    """
    empty_df = pd.DataFrame(columns=["Category", "Subcategory", "Value"])
    result_row = adorn_totals(empty_df, "Category", "Subcategory", axis=0)
    result_col = adorn_totals(empty_df, "Category", "Subcategory", axis=1)

    assert (
        result_row.empty
    ), "Le tableau croisé doit être vide lorsqu'un DataFrame vide est utilisé."
    assert (
        result_col.empty
    ), "Le tableau croisé doit être vide lorsqu'un DataFrame vide est utilisé."


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
    Test que adorn_totals fonctionne correctement avec un DataFrame plus grand.
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
    ), "La ligne 'Total' doit être présente dans le tableau pour un grand DataFrame."
    assert result.loc["Total"].sum() == len(
        large_data["Value"]
    ), "La somme de la ligne 'Total' doit correspondre au total des comptes."
