"""Tests for the tabyl function."""

import pandas as pd
import pytest

import janitor  # noqa: F401 - needed to register DataFrame methods


@pytest.fixture
def basic_df():
    """Create a basic DataFrame for tabyl tests."""
    return pd.DataFrame(
        {
            "cyl": [4, 6, 8, 4, 6, 8, 4, 6],
            "gear": [3, 3, 3, 4, 4, 4, 5, 5],
            "am": [0, 0, 0, 1, 1, 1, 1, 1],
        }
    )


@pytest.fixture
def df_with_na():
    """Create a DataFrame with NA values."""
    return pd.DataFrame(
        {
            "category": ["A", "B", "A", None, "B", "A"],
            "value": [1, 2, 1, 1, 2, None],
        }
    )


@pytest.fixture
def categorical_df():
    """Create a DataFrame with categorical column."""
    df = pd.DataFrame(
        {
            "grade": pd.Categorical(
                ["A", "B", "A", "C"],
                categories=["A", "B", "C", "D"],  # D is unused
            ),
            "subject": pd.Categorical(
                ["Math", "Science", "Math", "Science"],
                categories=["Math", "Science", "History"],  # History is unused
            ),
        }
    )
    return df


# =============================================================================
# 1-way tabyl tests
# =============================================================================


@pytest.mark.functions
def test_tabyl_one_way_basic(basic_df):
    """Test 1-way tabyl with basic DataFrame."""
    result = basic_df.tabyl("cyl")

    assert list(result.columns) == ["cyl", "n", "percent"]
    assert result.shape == (3, 3)
    assert result["n"].sum() == 8
    assert result["percent"].sum() == pytest.approx(1.0)


@pytest.mark.functions
def test_tabyl_one_way_values(basic_df):
    """Test 1-way tabyl returns correct values."""
    result = basic_df.tabyl("cyl")

    # cyl=4 appears 3 times, cyl=6 appears 3 times, cyl=8 appears 2 times
    expected_n = [3, 3, 2]
    assert list(result["n"]) == expected_n
    assert list(result["percent"]) == [0.375, 0.375, 0.25]


@pytest.mark.functions
def test_tabyl_one_way_with_na_show_na_true(df_with_na):
    """Test 1-way tabyl with NA values and show_na=True."""
    result = df_with_na.tabyl("category", show_na=True)

    # Should have 3 rows: A, B, and None
    assert result.shape[0] == 3
    assert "valid_percent" in result.columns


@pytest.mark.functions
def test_tabyl_one_way_with_na_show_na_false(df_with_na):
    """Test 1-way tabyl with NA values and show_na=False."""
    result = df_with_na.tabyl("category", show_na=False)

    # Should have 2 rows: A and B (no None)
    assert result.shape[0] == 2
    assert "valid_percent" not in result.columns


@pytest.mark.functions
def test_tabyl_one_way_categorical_missing_levels(categorical_df):
    """Test 1-way tabyl shows missing categorical levels."""
    result = categorical_df.tabyl("grade", show_missing_levels=True)

    # Should have 4 rows: A, B, C, D (D has count 0)
    assert result.shape[0] == 4
    assert 0 in result["n"].values  # D should have count 0


@pytest.mark.functions
def test_tabyl_one_way_categorical_no_missing_levels(categorical_df):
    """Test 1-way tabyl hides missing categorical levels."""
    result = categorical_df.tabyl("grade", show_missing_levels=False)

    # Should have 3 rows: A, B, C (D is excluded)
    assert result.shape[0] == 3


# =============================================================================
# 2-way tabyl tests
# =============================================================================


@pytest.mark.functions
def test_tabyl_two_way_basic(basic_df):
    """Test 2-way tabyl with basic DataFrame."""
    result = basic_df.tabyl("cyl", "gear")

    # First column is cyl, rest are gear values (3, 4, 5)
    assert result.columns[0] == "cyl"
    assert set(result.columns[1:]) == {3, 4, 5}
    assert result.shape[0] == 3  # 3 unique cyl values


@pytest.mark.functions
def test_tabyl_two_way_values(basic_df):
    """Test 2-way tabyl returns correct crosstab values."""
    result = basic_df.tabyl("cyl", "gear")

    # Check specific cell values
    # cyl=4, gear=3 -> 1, cyl=4, gear=4 -> 1, cyl=4, gear=5 -> 1
    row_4 = result[result["cyl"] == 4].iloc[0]
    assert row_4[3] == 1
    assert row_4[4] == 1
    assert row_4[5] == 1

    # cyl=8, gear=5 -> 0
    row_8 = result[result["cyl"] == 8].iloc[0]
    assert row_8[5] == 0


@pytest.mark.functions
def test_tabyl_two_way_with_na(df_with_na):
    """Test 2-way tabyl with NA values."""
    result = df_with_na.tabyl("category", "value", show_na=True)

    # Should include NA in the crosstab
    assert result.shape[0] >= 2  # At least A and B (possibly None)


@pytest.mark.functions
def test_tabyl_two_way_categorical_missing_levels(categorical_df):
    """Test 2-way tabyl shows missing categorical levels."""
    result = categorical_df.tabyl("grade", "subject", show_missing_levels=True)

    # Should have 4 rows (A, B, C, D) and columns for Math, Science, History
    assert result.shape[0] == 4
    assert "History" in result.columns or len(result.columns) == 4


# =============================================================================
# 3-way tabyl tests
# =============================================================================


@pytest.mark.functions
def test_tabyl_three_way_basic(basic_df):
    """Test 3-way tabyl with basic DataFrame."""
    result = basic_df.tabyl("cyl", "gear", "am")

    # Should return a dictionary
    assert isinstance(result, dict)
    # Keys should be unique values of am: 0 and 1
    assert set(result.keys()) == {0, 1}


@pytest.mark.functions
def test_tabyl_three_way_values(basic_df):
    """Test 3-way tabyl returns correct nested crosstab values."""
    result = basic_df.tabyl("cyl", "gear", "am")

    # For am=0, there are 3 rows (one for each cyl value)
    # cyl=4, gear=3, am=0 -> 1
    am_0 = result[0]
    assert am_0.shape[0] == 3
    row_4_am0 = am_0[am_0["cyl"] == 4].iloc[0]
    assert row_4_am0[3] == 1  # cyl=4, gear=3, am=0 -> 1

    # For am=1, check that we have the expected structure
    am_1 = result[1]
    assert am_1.shape[0] == 3  # 3 cyl values
    # cyl=4, gear=4, am=1 -> 1
    row_4_am1 = am_1[am_1["cyl"] == 4].iloc[0]
    assert row_4_am1[4] == 1  # cyl=4, gear=4, am=1 -> 1


@pytest.mark.functions
def test_tabyl_three_way_each_is_dataframe(basic_df):
    """Test 3-way tabyl returns DataFrames for each split value."""
    result = basic_df.tabyl("cyl", "gear", "am")

    for key, df in result.items():
        assert isinstance(df, pd.DataFrame)
        assert "cyl" in df.columns


# =============================================================================
# Error handling tests
# =============================================================================


@pytest.mark.functions
def test_tabyl_no_columns_raises_error(basic_df):
    """Test tabyl raises ValueError when no columns provided."""
    with pytest.raises(ValueError, match="At least one column name"):
        basic_df.tabyl()


@pytest.mark.functions
def test_tabyl_too_many_columns_raises_error(basic_df):
    """Test tabyl raises ValueError when more than 3 columns provided."""
    with pytest.raises(ValueError, match="At most three column names"):
        basic_df.tabyl("cyl", "gear", "am", "extra")


@pytest.mark.functions
def test_tabyl_invalid_column_raises_error(basic_df):
    """Test tabyl raises KeyError when column not found."""
    with pytest.raises(KeyError, match="Column 'invalid' not found"):
        basic_df.tabyl("invalid")


# =============================================================================
# Method chaining tests
# =============================================================================


@pytest.mark.functions
def test_tabyl_method_chaining():
    """Test tabyl works with method chaining."""
    result = pd.DataFrame({"a": [1, 1, 2, 2, 2], "b": ["x", "y", "x", "x", "y"]}).tabyl(
        "a"
    )

    assert list(result.columns) == ["a", "n", "percent"]
    assert result.shape == (2, 3)


@pytest.mark.functions
def test_tabyl_does_not_mutate_original(basic_df):
    """Test that tabyl does not mutate the original DataFrame."""
    original_shape = basic_df.shape
    original_columns = list(basic_df.columns)

    basic_df.tabyl("cyl")

    assert basic_df.shape == original_shape
    assert list(basic_df.columns) == original_columns


# =============================================================================
# Integration tests with adorn_* functions
# =============================================================================


@pytest.mark.functions
def test_tabyl_two_way_with_adorn_totals(basic_df):
    """Test that 2-way tabyl works with adorn_totals."""
    result = basic_df.tabyl("cyl", "gear").adorn_totals("both")

    # Should have totals row
    assert len(result) == 4  # 3 cyl values + Total row
    assert result.iloc[-1]["cyl"] == "Total"
    # Should have totals column
    assert "Total" in result.columns


@pytest.mark.functions
def test_tabyl_two_way_with_adorn_percentages(basic_df):
    """Test that 2-way tabyl works with adorn_percentages."""
    result = basic_df.tabyl("cyl", "gear").adorn_percentages("row")

    # Each row should sum to 1.0
    numeric_cols = [c for c in result.columns if c != "cyl"]
    for idx in result.index:
        row_sum = result.loc[idx, numeric_cols].sum()
        assert abs(row_sum - 1.0) < 0.001 or row_sum == 0


@pytest.mark.functions
def test_tabyl_two_way_full_adorn_pipeline(basic_df):
    """Test full adorn pipeline with tabyl output.

    Tests: tabyl -> adorn_totals -> adorn_percentages ->
    adorn_pct_formatting -> adorn_ns.
    """
    result = (
        basic_df.tabyl("cyl", "gear")
        .adorn_totals("both")
        .adorn_percentages("row")
        .adorn_pct_formatting(digits=1)
        .adorn_ns()
    )

    # Should have totals row and column
    assert len(result) == 4
    assert "Total" in result.columns

    # Should have formatted percentages with N counts
    # Check a cell in the first data row (not totals)
    first_data_cell = result.iloc[0][3]  # First numeric column value
    assert "%" in str(first_data_cell)
    assert "(" in str(first_data_cell)


@pytest.mark.functions
def test_tabyl_one_way_with_adorn_totals():
    """Test that 1-way tabyl works with adorn_totals."""
    df = pd.DataFrame({"category": ["A", "B", "A", "C", "B", "A"]})
    result = df.tabyl("category").adorn_totals("row")

    # Should have totals row
    assert len(result) == 4  # 3 categories + Total row
    assert result.iloc[-1]["category"] == "Total"
    # Total n should be sum of all n values
    assert result.iloc[-1]["n"] == 6


@pytest.mark.functions
def test_tabyl_preserves_numeric_types_for_adorn():
    """Test that tabyl output has numeric types that work with adorn functions."""
    df = pd.DataFrame(
        {
            "group": ["A", "A", "B", "B", "B"],
            "status": ["yes", "no", "yes", "yes", "no"],
        }
    )
    result = df.tabyl("group", "status")

    # The count columns should be numeric
    for col in ["no", "yes"]:
        if col in result.columns:
            assert pd.api.types.is_numeric_dtype(result[col])

    # Should work with adorn_percentages
    pct_result = result.adorn_percentages("row")
    for col in ["no", "yes"]:
        if col in pct_result.columns:
            assert pd.api.types.is_numeric_dtype(pct_result[col])

@pytest.mark.functions
def test_adorn_functions_preserve_numeric_first_column():
    """Test that adorn functions preserve numeric values in column 0."""
    df_counts = pd.DataFrame(
        {
            "cyl": [4, 6, 8],
            "gear_3": [1, 2, 1],
            "gear_4": [2, 2, 3],
        }
    )

    df_pct = pd.DataFrame(
        {
            "cyl": [4, 6, 8],
            "gear_3": ["33.3%", "50.0%", "25.0%"],
            "gear_4": ["66.7%", "50.0%", "75.0%"],
        }
    )

    df_unrounded = pd.DataFrame(
        {
            "cyl": [4, 6, 8],
            "gear_3": [33.333333, 50.000000, 25.000000],
            "gear_4": [66.666667, 50.000000, 75.000000],
        }
    )

    # 1. Test adorn_pct_formatting
    raw_pct_df = pd.DataFrame(
        {
            "cyl": [4, 6, 8],
            "gear_3": [0.333333, 0.500000, 0.250000],
            "gear_4": [0.666667, 0.500000, 0.750000],
        }
    )
    result_pct = raw_pct_df.adorn_pct_formatting(digits=1)
    assert list(result_pct["cyl"]) == [4, 6, 8]

    # 2. Test adorn_ns
    result_ns = df_pct.adorn_ns(ns=df_counts)
    assert list(result_ns["cyl"]) == [4, 6, 8]
    assert result_ns.loc[0, "gear_3"] == "33.3% (1)"

    # 3. Test adorn_rounding
    result_rounding = df_unrounded.adorn_rounding(digits=1)
    assert list(result_rounding["cyl"]) == [4, 6, 8]
    assert result_rounding.loc[0, "gear_3"] == 33.3
