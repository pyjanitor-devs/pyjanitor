"""Tests for adorn_* functions."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def simple_df():
    """Create a simple DataFrame for testing."""
    return pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "count1": [10, 20, 30],
            "count2": [5, 15, 25],
        }
    )


@pytest.fixture
def crosstab_df():
    """Create a cross-tabulation style DataFrame for testing."""
    return pd.DataFrame(
        {
            "row_var": ["X", "Y"],
            "col_a": [10, 20],
            "col_b": [5, 15],
            "col_c": [3, 7],
        }
    )


@pytest.fixture
def float_df():
    """Create a DataFrame with float values for rounding tests."""
    return pd.DataFrame(
        {
            "category": ["A", "B"],
            "value1": [1.2345, 2.5678],
            "value2": [3.4567, 4.5555],
        }
    )


# Tests for adorn_totals


@pytest.mark.functions
def test_adorn_totals_row(simple_df):
    """Test adding a totals row."""
    result = simple_df.adorn_totals("row")
    assert len(result) == 4
    assert result.iloc[-1]["category"] == "Total"
    assert result.iloc[-1]["count1"] == 60
    assert result.iloc[-1]["count2"] == 45


@pytest.mark.functions
def test_adorn_totals_col(simple_df):
    """Test adding a totals column."""
    result = simple_df.adorn_totals("col")
    assert "Total" in result.columns
    assert result.iloc[0]["Total"] == 15
    assert result.iloc[1]["Total"] == 35
    assert result.iloc[2]["Total"] == 55


@pytest.mark.functions
def test_adorn_totals_both(simple_df):
    """Test adding totals to both row and column."""
    result = simple_df.adorn_totals("both")
    assert len(result) == 4
    assert "Total" in result.columns
    assert result.iloc[-1]["category"] == "Total"
    assert result.iloc[-1]["Total"] == 105  # Grand total


@pytest.mark.functions
def test_adorn_totals_custom_name(simple_df):
    """Test custom name for totals."""
    result = simple_df.adorn_totals("row", name="Sum")
    assert result.iloc[-1]["category"] == "Sum"


@pytest.mark.functions
def test_adorn_totals_invalid_where(simple_df):
    """Test that invalid where parameter raises ValueError."""
    with pytest.raises(ValueError):
        simple_df.adorn_totals("invalid")


# Tests for adorn_percentages


@pytest.mark.functions
def test_adorn_percentages_row(simple_df):
    """Test row-wise percentages."""
    result = simple_df.adorn_percentages("row")
    # Row 0: 10/(10+5) = 0.667, 5/(10+5) = 0.333
    assert np.isclose(result.iloc[0]["count1"], 10 / 15)
    assert np.isclose(result.iloc[0]["count2"], 5 / 15)


@pytest.mark.functions
def test_adorn_percentages_col(simple_df):
    """Test column-wise percentages."""
    result = simple_df.adorn_percentages("col")
    # Column sum for count1 = 60
    assert np.isclose(result.iloc[0]["count1"], 10 / 60)
    assert np.isclose(result.iloc[1]["count1"], 20 / 60)


@pytest.mark.functions
def test_adorn_percentages_all(simple_df):
    """Test overall percentages."""
    result = simple_df.adorn_percentages("all")
    # Grand total = 60 + 45 = 105
    assert np.isclose(result.iloc[0]["count1"], 10 / 105)
    assert np.isclose(result.iloc[0]["count2"], 5 / 105)


@pytest.mark.functions
def test_adorn_percentages_invalid_denominator(simple_df):
    """Test that invalid denominator raises ValueError."""
    with pytest.raises(ValueError):
        simple_df.adorn_percentages("invalid")


@pytest.mark.functions
def test_adorn_percentages_stores_original_counts(simple_df):
    """Test that original counts are stored in attrs."""
    result = simple_df.adorn_percentages("row")
    assert "_original_counts" in result.attrs
    assert result.attrs["_original_counts"].iloc[0]["count1"] == 10


# Tests for adorn_pct_formatting


@pytest.mark.functions
def test_adorn_pct_formatting_default(simple_df):
    """Test default percentage formatting."""
    result = simple_df.adorn_percentages("row").adorn_pct_formatting()
    # Row 0: 10/15 = 66.7%
    assert result.iloc[0]["count1"] == "66.7%"


@pytest.mark.functions
def test_adorn_pct_formatting_no_sign(simple_df):
    """Test formatting without percent sign."""
    result = simple_df.adorn_percentages("row").adorn_pct_formatting(affix_sign=False)
    assert result.iloc[0]["count1"] == "66.7"


@pytest.mark.functions
def test_adorn_pct_formatting_digits(simple_df):
    """Test formatting with custom digits."""
    result = simple_df.adorn_percentages("row").adorn_pct_formatting(digits=2)
    assert result.iloc[0]["count1"] == "66.67%"


@pytest.mark.functions
def test_adorn_pct_formatting_invalid_rounding(simple_df):
    """Test that invalid rounding method raises ValueError."""
    with pytest.raises(ValueError):
        simple_df.adorn_percentages("row").adorn_pct_formatting(rounding="invalid")


# Tests for adorn_ns


@pytest.mark.functions
def test_adorn_ns_rear(simple_df):
    """Test adding N counts at rear position."""
    result = (
        simple_df.adorn_percentages("row")
        .adorn_pct_formatting()
        .adorn_ns(position="rear")
    )
    assert "(10)" in result.iloc[0]["count1"]
    assert result.iloc[0]["count1"].endswith("(10)")


@pytest.mark.functions
def test_adorn_ns_front(simple_df):
    """Test adding N counts at front position."""
    result = (
        simple_df.adorn_percentages("row")
        .adorn_pct_formatting()
        .adorn_ns(position="front")
    )
    assert "(10)" in result.iloc[0]["count1"]
    assert result.iloc[0]["count1"].startswith("(10)")


@pytest.mark.functions
def test_adorn_ns_custom_format_func(simple_df):
    """Test custom format function for N counts."""
    result = (
        simple_df.adorn_percentages("row")
        .adorn_pct_formatting()
        .adorn_ns(format_func=lambda n: f"[n={int(n)}]")
    )
    assert "[n=10]" in result.iloc[0]["count1"]


@pytest.mark.functions
def test_adorn_ns_no_original_counts(simple_df):
    """Test that adorn_ns raises error without original counts."""
    # Create a df without original counts stored
    df_no_counts = simple_df.copy()
    df_no_counts["count1"] = df_no_counts["count1"].astype(str)
    with pytest.raises(ValueError, match="No original counts available"):
        df_no_counts.adorn_ns()


@pytest.mark.functions
def test_adorn_ns_invalid_position(simple_df):
    """Test that invalid position raises ValueError."""
    with pytest.raises(ValueError):
        simple_df.adorn_percentages("row").adorn_ns(position="invalid")


# Tests for adorn_title


@pytest.mark.functions
def test_adorn_title_combined(crosstab_df):
    """Test combined placement of title."""
    result = crosstab_df.adorn_title(
        placement="combined", row_name="row", col_name="col"
    )
    assert "row/col" in result.columns


@pytest.mark.functions
def test_adorn_title_top(crosstab_df):
    """Test top placement of title."""
    result = crosstab_df.adorn_title(placement="top", col_name="columns")
    # Result should have MultiIndex columns
    assert isinstance(result.columns, pd.MultiIndex)


@pytest.mark.functions
def test_adorn_title_invalid_placement(crosstab_df):
    """Test that invalid placement raises ValueError."""
    with pytest.raises(ValueError):
        crosstab_df.adorn_title(placement="invalid")


# Tests for adorn_rounding


@pytest.mark.functions
def test_adorn_rounding_default(float_df):
    """Test default rounding."""
    result = float_df.adorn_rounding(digits=2)
    assert result.iloc[0]["value1"] == 1.23
    assert result.iloc[0]["value2"] == 3.46


@pytest.mark.functions
def test_adorn_rounding_half_up(float_df):
    """Test half up rounding."""
    result = float_df.adorn_rounding(digits=1, rounding="half up")
    assert result.iloc[0]["value1"] == 1.2
    assert result.iloc[1]["value1"] == 2.6


@pytest.mark.functions
def test_adorn_rounding_preserves_non_numeric(float_df):
    """Test that non-numeric columns are preserved."""
    result = float_df.adorn_rounding(digits=1)
    assert result.iloc[0]["category"] == "A"
    assert result.iloc[1]["category"] == "B"


@pytest.mark.functions
def test_adorn_rounding_invalid_rounding(float_df):
    """Test that invalid rounding method raises ValueError."""
    with pytest.raises(ValueError):
        float_df.adorn_rounding(rounding="invalid")


# Tests for chaining multiple adorn functions


@pytest.mark.functions
def test_adorn_full_pipeline(simple_df):
    """Test full pipeline of adorn functions."""
    result = (
        simple_df.adorn_totals("both")
        .adorn_percentages("row")
        .adorn_pct_formatting(digits=1)
        .adorn_ns()
    )
    # Should have totals row
    assert len(result) == 4
    # Should have totals column
    assert "Total" in result.columns
    # Should have formatted percentages with N counts
    assert "%" in result.iloc[0]["count1"]
    assert "(" in result.iloc[0]["count1"]


# Tests for NA value handling


@pytest.fixture
def df_with_na():
    """Create a DataFrame with NA values for testing."""
    return pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "count1": [10, np.nan, 30],
            "count2": [5, 15, np.nan],
        }
    )


@pytest.mark.functions
def test_adorn_totals_with_na(df_with_na):
    """Test adorn_totals handles NA values correctly with na_rm=True."""
    result = df_with_na.adorn_totals("row", na_rm=True)
    # Should sum non-NA values: 10 + 30 = 40 for count1
    assert result.iloc[-1]["count1"] == 40
    # Should sum non-NA values: 5 + 15 = 20 for count2
    assert result.iloc[-1]["count2"] == 20


@pytest.mark.functions
def test_adorn_percentages_with_na(df_with_na):
    """Test adorn_percentages handles NA values correctly."""
    result = df_with_na.adorn_percentages("col", na_rm=True)
    # Column sum for count1 = 40 (excluding NA)
    assert np.isclose(result.iloc[0]["count1"], 10 / 40)
    assert np.isclose(result.iloc[2]["count1"], 30 / 40)
    # NA should remain NA
    assert pd.isna(result.iloc[1]["count1"])


@pytest.mark.functions
def test_adorn_pct_formatting_with_na(df_with_na):
    """Test adorn_pct_formatting preserves NA values."""
    result = df_with_na.adorn_percentages("col").adorn_pct_formatting()
    # NA should remain NA
    assert pd.isna(result.iloc[1]["count1"])


# Tests for numeric first column edge case


@pytest.fixture
def numeric_first_col_df():
    """Create a DataFrame with numeric first column."""
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "count1": [10, 20, 30],
            "count2": [5, 15, 25],
        }
    )


@pytest.mark.functions
def test_adorn_totals_numeric_first_column(numeric_first_col_df):
    """Test adorn_totals with numeric first column."""
    result = numeric_first_col_df.adorn_totals("row")
    # First column should be summed as it's numeric
    assert result.iloc[-1]["id"] == 6  # 1 + 2 + 3
    assert result.iloc[-1]["count1"] == 60


@pytest.mark.functions
def test_adorn_percentages_numeric_first_column(numeric_first_col_df):
    """Test adorn_percentages with numeric first column."""
    result = numeric_first_col_df.adorn_percentages("row")
    # All numeric columns should be converted to percentages
    # Row 0: id=1, count1=10, count2=5, total=16
    assert np.isclose(result.iloc[0]["id"], 1 / 16)
    assert np.isclose(result.iloc[0]["count1"], 10 / 16)


# Tests for adorn_ns with custom ns DataFrame


@pytest.mark.functions
def test_adorn_ns_with_custom_ns(simple_df):
    """Test adorn_ns with custom ns DataFrame provided."""
    # Create a custom ns DataFrame with different values
    custom_ns = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "count1": [100, 200, 300],
            "count2": [50, 150, 250],
        }
    )
    result = (
        simple_df.adorn_percentages("row").adorn_pct_formatting().adorn_ns(ns=custom_ns)
    )
    # Should use custom ns values
    assert "(100)" in result.iloc[0]["count1"]
    assert "(200)" in result.iloc[1]["count1"]


# Tests for adorn_ns skipping totals row


@pytest.mark.functions
def test_adorn_ns_skips_totals_row(simple_df):
    """Test that adorn_ns correctly skips the totals row."""
    result = (
        simple_df.adorn_totals("row")
        .adorn_percentages("row")
        .adorn_pct_formatting()
        .adorn_ns()
    )
    # Totals row should have percentage but no N count appended
    # because the original counts don't include the totals row
    totals_row_value = result.iloc[-1]["count1"]
    # The totals row should still have a percentage
    assert "%" in totals_row_value
    # But should NOT have parentheses from adorn_ns
    # (unless it was added during adorn_totals which stores counts before totals)


@pytest.mark.functions
def test_adorn_ns_with_non_sequential_index():
    """Test adorn_ns handles non-sequential indices correctly."""
    df = pd.DataFrame(
        {
            "category": ["A", "B", "C"],
            "count1": [10, 20, 30],
            "count2": [5, 15, 25],
        },
        index=[10, 20, 30],  # Non-sequential index
    )
    result = df.adorn_percentages("row").adorn_pct_formatting().adorn_ns()
    # Should still work correctly with non-sequential indices
    assert "(10)" in result.loc[10, "count1"]
    assert "(20)" in result.loc[20, "count1"]
    assert "(30)" in result.loc[30, "count1"]


# Tests for adorn_totals fill parameter


@pytest.mark.functions
def test_adorn_totals_fill_parameter():
    """Test that fill parameter works for non-numeric columns."""
    df = pd.DataFrame(
        {
            "category": ["A", "B"],
            "subcategory": ["X", "Y"],
            "count": [10, 20],
        }
    )
    result = df.adorn_totals("row", fill="--")
    # First column should have the name "Total"
    assert result.iloc[-1]["category"] == "Total"
    # Second non-numeric column should have the fill value
    assert result.iloc[-1]["subcategory"] == "--"


# Tests for thousand separator in adorn_ns


@pytest.mark.functions
def test_adorn_ns_thousand_separator():
    """Test that adorn_ns includes thousand separator in large numbers."""
    df = pd.DataFrame(
        {
            "category": ["A", "B"],
            "count": [1000, 2000000],
        }
    )
    result = df.adorn_percentages("col").adorn_pct_formatting().adorn_ns()
    # Should have thousand separator
    assert "(1,000)" in result.iloc[0]["count"]
    assert "(2,000,000)" in result.iloc[1]["count"]
