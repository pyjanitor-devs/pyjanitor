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


class TestAdornTotals:
    """Tests for adorn_totals function."""

    @pytest.mark.functions
    def test_adorn_totals_row(self, simple_df):
        """Test adding a totals row."""
        result = simple_df.adorn_totals("row")
        assert len(result) == 4
        assert result.iloc[-1]["category"] == "Total"
        assert result.iloc[-1]["count1"] == 60
        assert result.iloc[-1]["count2"] == 45

    @pytest.mark.functions
    def test_adorn_totals_col(self, simple_df):
        """Test adding a totals column."""
        result = simple_df.adorn_totals("col")
        assert "Total" in result.columns
        assert result.iloc[0]["Total"] == 15
        assert result.iloc[1]["Total"] == 35
        assert result.iloc[2]["Total"] == 55

    @pytest.mark.functions
    def test_adorn_totals_both(self, simple_df):
        """Test adding totals to both row and column."""
        result = simple_df.adorn_totals("both")
        assert len(result) == 4
        assert "Total" in result.columns
        assert result.iloc[-1]["category"] == "Total"
        assert result.iloc[-1]["Total"] == 105  # Grand total

    @pytest.mark.functions
    def test_adorn_totals_custom_name(self, simple_df):
        """Test custom name for totals."""
        result = simple_df.adorn_totals("row", name="Sum")
        assert result.iloc[-1]["category"] == "Sum"

    @pytest.mark.functions
    def test_adorn_totals_invalid_where(self, simple_df):
        """Test that invalid where parameter raises ValueError."""
        with pytest.raises(ValueError):
            simple_df.adorn_totals("invalid")


class TestAdornPercentages:
    """Tests for adorn_percentages function."""

    @pytest.mark.functions
    def test_adorn_percentages_row(self, simple_df):
        """Test row-wise percentages."""
        result = simple_df.adorn_percentages("row")
        # Row 0: 10/(10+5) = 0.667, 5/(10+5) = 0.333
        assert np.isclose(result.iloc[0]["count1"], 10 / 15)
        assert np.isclose(result.iloc[0]["count2"], 5 / 15)

    @pytest.mark.functions
    def test_adorn_percentages_col(self, simple_df):
        """Test column-wise percentages."""
        result = simple_df.adorn_percentages("col")
        # Column sum for count1 = 60
        assert np.isclose(result.iloc[0]["count1"], 10 / 60)
        assert np.isclose(result.iloc[1]["count1"], 20 / 60)

    @pytest.mark.functions
    def test_adorn_percentages_all(self, simple_df):
        """Test overall percentages."""
        result = simple_df.adorn_percentages("all")
        # Grand total = 60 + 45 = 105
        assert np.isclose(result.iloc[0]["count1"], 10 / 105)
        assert np.isclose(result.iloc[0]["count2"], 5 / 105)

    @pytest.mark.functions
    def test_adorn_percentages_invalid_denominator(self, simple_df):
        """Test that invalid denominator raises ValueError."""
        with pytest.raises(ValueError):
            simple_df.adorn_percentages("invalid")

    @pytest.mark.functions
    def test_adorn_percentages_stores_original_counts(self, simple_df):
        """Test that original counts are stored in attrs."""
        result = simple_df.adorn_percentages("row")
        assert "_original_counts" in result.attrs
        assert result.attrs["_original_counts"].iloc[0]["count1"] == 10


class TestAdornPctFormatting:
    """Tests for adorn_pct_formatting function."""

    @pytest.mark.functions
    def test_adorn_pct_formatting_default(self, simple_df):
        """Test default percentage formatting."""
        result = simple_df.adorn_percentages("row").adorn_pct_formatting()
        # Row 0: 10/15 = 66.7%
        assert result.iloc[0]["count1"] == "66.7%"

    @pytest.mark.functions
    def test_adorn_pct_formatting_no_sign(self, simple_df):
        """Test formatting without percent sign."""
        result = simple_df.adorn_percentages("row").adorn_pct_formatting(
            affix_sign=False
        )
        assert result.iloc[0]["count1"] == "66.7"

    @pytest.mark.functions
    def test_adorn_pct_formatting_digits(self, simple_df):
        """Test formatting with custom digits."""
        result = simple_df.adorn_percentages("row").adorn_pct_formatting(digits=2)
        assert result.iloc[0]["count1"] == "66.67%"

    @pytest.mark.functions
    def test_adorn_pct_formatting_invalid_rounding(self, simple_df):
        """Test that invalid rounding method raises ValueError."""
        with pytest.raises(ValueError):
            simple_df.adorn_percentages("row").adorn_pct_formatting(rounding="invalid")


class TestAdornNs:
    """Tests for adorn_ns function."""

    @pytest.mark.functions
    def test_adorn_ns_rear(self, simple_df):
        """Test adding N counts at rear position."""
        result = (
            simple_df.adorn_percentages("row")
            .adorn_pct_formatting()
            .adorn_ns(position="rear")
        )
        assert "(10)" in result.iloc[0]["count1"]
        assert result.iloc[0]["count1"].endswith("(10)")

    @pytest.mark.functions
    def test_adorn_ns_front(self, simple_df):
        """Test adding N counts at front position."""
        result = (
            simple_df.adorn_percentages("row")
            .adorn_pct_formatting()
            .adorn_ns(position="front")
        )
        assert "(10)" in result.iloc[0]["count1"]
        assert result.iloc[0]["count1"].startswith("(10)")

    @pytest.mark.functions
    def test_adorn_ns_custom_format_func(self, simple_df):
        """Test custom format function for N counts."""
        result = (
            simple_df.adorn_percentages("row")
            .adorn_pct_formatting()
            .adorn_ns(format_func=lambda n: f"[n={int(n)}]")
        )
        assert "[n=10]" in result.iloc[0]["count1"]

    @pytest.mark.functions
    def test_adorn_ns_no_original_counts(self, simple_df):
        """Test that adorn_ns raises error without original counts."""
        # Create a df without original counts stored
        df_no_counts = simple_df.copy()
        df_no_counts["count1"] = df_no_counts["count1"].astype(str)
        with pytest.raises(ValueError, match="No original counts available"):
            df_no_counts.adorn_ns()

    @pytest.mark.functions
    def test_adorn_ns_invalid_position(self, simple_df):
        """Test that invalid position raises ValueError."""
        with pytest.raises(ValueError):
            simple_df.adorn_percentages("row").adorn_ns(position="invalid")


class TestAdornTitle:
    """Tests for adorn_title function."""

    @pytest.mark.functions
    def test_adorn_title_combined(self, crosstab_df):
        """Test combined placement of title."""
        result = crosstab_df.adorn_title(
            placement="combined", row_name="row", col_name="col"
        )
        assert "row/col" in result.columns

    @pytest.mark.functions
    def test_adorn_title_top(self, crosstab_df):
        """Test top placement of title."""
        result = crosstab_df.adorn_title(placement="top", col_name="columns")
        # Result should have MultiIndex columns
        assert isinstance(result.columns, pd.MultiIndex)

    @pytest.mark.functions
    def test_adorn_title_invalid_placement(self, crosstab_df):
        """Test that invalid placement raises ValueError."""
        with pytest.raises(ValueError):
            crosstab_df.adorn_title(placement="invalid")


class TestAdornRounding:
    """Tests for adorn_rounding function."""

    @pytest.fixture
    def float_df(self):
        """Create a DataFrame with float values for rounding tests."""
        return pd.DataFrame(
            {
                "category": ["A", "B"],
                "value1": [1.2345, 2.5678],
                "value2": [3.4567, 4.5555],
            }
        )

    @pytest.mark.functions
    def test_adorn_rounding_default(self, float_df):
        """Test default rounding."""
        result = float_df.adorn_rounding(digits=2)
        assert result.iloc[0]["value1"] == 1.23
        assert result.iloc[0]["value2"] == 3.46

    @pytest.mark.functions
    def test_adorn_rounding_half_up(self, float_df):
        """Test half up rounding."""
        result = float_df.adorn_rounding(digits=1, rounding="half up")
        assert result.iloc[0]["value1"] == 1.2
        assert result.iloc[1]["value1"] == 2.6

    @pytest.mark.functions
    def test_adorn_rounding_preserves_non_numeric(self, float_df):
        """Test that non-numeric columns are preserved."""
        result = float_df.adorn_rounding(digits=1)
        assert result.iloc[0]["category"] == "A"
        assert result.iloc[1]["category"] == "B"

    @pytest.mark.functions
    def test_adorn_rounding_invalid_rounding(self, float_df):
        """Test that invalid rounding method raises ValueError."""
        with pytest.raises(ValueError):
            float_df.adorn_rounding(rounding="invalid")


class TestAdornChaining:
    """Tests for chaining multiple adorn functions."""

    @pytest.mark.functions
    def test_full_pipeline(self, simple_df):
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
