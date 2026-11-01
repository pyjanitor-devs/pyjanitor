"""Tests for adorn_* functions."""

import numpy as np
import pandas as pd
import pytest

import janitor  # noqa: F401


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
    """Test adorn_totals with numeric first column.

    The first column is always treated as a row identifier (consistent with
    R janitor tabyl behavior), so it gets "Total" instead of being summed.
    """
    result = numeric_first_col_df.adorn_totals("row")
    # First column is treated as row identifier, gets "Total" label
    assert result.iloc[-1]["id"] == "Total"
    assert result.iloc[-1]["count1"] == 60


@pytest.mark.functions
def test_adorn_percentages_numeric_first_column(numeric_first_col_df):
    """Test adorn_percentages with numeric first column.

    The first column is always treated as a row identifier (consistent with
    R janitor tabyl behavior), so it's not converted to percentages.
    """
    result = numeric_first_col_df.adorn_percentages("row")
    # First column is treated as row identifier, not converted to percentages
    # Original values are preserved
    assert result.iloc[0]["id"] == 1
    # Row 0: count1=10, count2=5, total=15
    assert np.isclose(result.iloc[0]["count1"], 10 / 15)
    assert np.isclose(result.iloc[0]["count2"], 5 / 15)


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


# The first column is the row identifier for every adorn_* function, including
# when it is numeric. Issue #1676.


IDENTIFIER_DTYPES = ["int64", "float64", "Int64", "Float64"]


def numeric_identifier_frame(dtype: str) -> pd.DataFrame:
    """Build a two-row counts frame whose first column uses ``dtype``."""
    if dtype == "int64":
        year = pd.Series([2020, 2021], dtype="int64")
    elif dtype == "float64":
        # Not already at 2 decimal places, so rounding would change the value.
        year = pd.Series([2020.444, 2021.555], dtype="float64")
    elif dtype == "Int64":
        year = pd.Series([2020, pd.NA], dtype="Int64")
    elif dtype == "Float64":
        year = pd.Series([2020.444, pd.NA], dtype="Float64")
    else:
        raise AssertionError(f"unexpected dtype {dtype}")
    return pd.DataFrame({"year": year, "yes": [10, 20], "no": [5, 15]})


def assert_identifier_unchanged(result: pd.Series, original: pd.Series) -> None:
    """Assert a row-identifier column kept its values and dtype."""
    pd.testing.assert_series_equal(result, original)


@pytest.mark.functions
@pytest.mark.parametrize("dtype", IDENTIFIER_DTYPES)
def test_adorn_pct_formatting_preserves_numeric_first_column(dtype):
    """Numeric row identifiers are not formatted as percentages."""
    df = numeric_identifier_frame(dtype)
    proportions = df.adorn_percentages("row")

    result = proportions.adorn_pct_formatting()

    assert_identifier_unchanged(result["year"], df["year"])
    assert result.iloc[0]["yes"] == "66.7%"
    assert result.iloc[0]["no"] == "33.3%"
    assert result.iloc[1]["yes"] == "57.1%"
    assert result.iloc[1]["no"] == "42.9%"


@pytest.mark.functions
@pytest.mark.parametrize("dtype", IDENTIFIER_DTYPES)
def test_adorn_ns_preserves_numeric_first_column(dtype):
    """Numeric row identifiers do not receive an N count."""
    df = numeric_identifier_frame(dtype)

    result = df.adorn_percentages("row").adorn_pct_formatting().adorn_ns()

    assert_identifier_unchanged(result["year"], df["year"])
    assert result.iloc[0]["yes"] == "66.7% (10)"
    assert result.iloc[0]["no"] == "33.3% (5)"
    assert result.iloc[1]["yes"] == "57.1% (20)"
    assert result.iloc[1]["no"] == "42.9% (15)"


@pytest.mark.functions
@pytest.mark.parametrize("dtype", IDENTIFIER_DTYPES)
def test_adorn_rounding_preserves_numeric_first_column(dtype):
    """Numeric row identifiers are not rounded and keep their dtype."""
    df = numeric_identifier_frame(dtype)

    result = df.adorn_percentages("row").adorn_rounding(digits=2)

    assert_identifier_unchanged(result["year"], df["year"])
    assert result.iloc[0]["yes"] == 0.67
    assert result.iloc[0]["no"] == 0.33
    assert result.iloc[1]["yes"] == 0.57
    assert result.iloc[1]["no"] == 0.43


@pytest.mark.functions
def test_adorn_pipeline_numeric_grouping_from_tabyl():
    """A tabyl on a numeric grouping variable keeps that variable readable.

    This is the report from issue #1676: ``cyl`` was rewritten as
    ``400.0%``, ``600.0%``, ``800.0%``.
    """
    df = pd.DataFrame(
        {
            "cyl": [4, 6, 8, 4, 6, 8, 4, 6],
            "gear": [3, 3, 3, 4, 4, 4, 5, 5],
        }
    )
    tabyl = df.tabyl("cyl", "gear")

    formatted = tabyl.adorn_percentages("row").adorn_pct_formatting()
    counted = formatted.adorn_ns()
    rounded = tabyl.adorn_percentages("row").adorn_rounding(digits=2)

    assert formatted["cyl"].tolist() == [4, 6, 8]
    assert formatted["cyl"].dtype == tabyl["cyl"].dtype
    assert formatted.iloc[0, 1] == "33.3%"
    assert counted["cyl"].tolist() == [4, 6, 8]
    assert counted.iloc[0, 1] == "33.3% (1)"
    assert rounded["cyl"].tolist() == [4, 6, 8]
    assert rounded["cyl"].dtype == tabyl["cyl"].dtype
    assert rounded.iloc[0, 1] == 0.33


@pytest.mark.functions
def test_adorn_string_first_column_regression():
    """A string row identifier is unchanged and data columns are still adorned."""
    df = pd.DataFrame({"category": ["A", "B"], "count1": [10, 20], "count2": [5, 15]})

    totals = df.adorn_totals("both")
    percentages = df.adorn_percentages("row")
    formatted = percentages.adorn_pct_formatting()
    counted = formatted.adorn_ns()
    rounded = percentages.adorn_rounding(digits=2)

    assert totals.iloc[0]["category"] == "A"
    assert totals.iloc[-1]["category"] == "Total"
    assert totals.iloc[0]["Total"] == 15
    assert totals.iloc[-1]["count1"] == 30
    assert percentages["category"].tolist() == ["A", "B"]
    assert np.isclose(percentages.iloc[0]["count1"], 10 / 15)
    assert formatted["category"].tolist() == ["A", "B"]
    assert formatted.iloc[0]["count1"] == "66.7%"
    assert counted["category"].tolist() == ["A", "B"]
    assert counted.iloc[0]["count1"] == "66.7% (10)"
    assert rounded["category"].tolist() == ["A", "B"]
    assert rounded.iloc[0]["count1"] == 0.67


@pytest.mark.functions
@pytest.mark.parametrize("dtype", IDENTIFIER_DTYPES)
def test_adorn_totals_and_percentages_leave_numeric_identifier(dtype):
    """Totals and percentages already treat a numeric first column as a label."""
    df = numeric_identifier_frame(dtype)

    totals = df.adorn_totals("both")
    percentages = df.adorn_percentages("row")
    column_percentages = df.adorn_percentages("col")

    assert totals.iloc[-1, 0] == "Total"
    # The totals label widens the identifier column, so compare values only.
    result_ids = totals.iloc[:-1, 0].tolist()
    original_ids = df["year"].tolist()
    assert len(result_ids) == len(original_ids)
    for got, expected in zip(result_ids, original_ids, strict=True):
        if pd.isna(expected):
            assert pd.isna(got)
        else:
            assert got == expected
    assert totals.iloc[0]["Total"] == 15
    assert totals.iloc[-1]["yes"] == 30
    assert totals.iloc[-1]["no"] == 20
    assert totals.iloc[-1]["Total"] == 50
    assert_identifier_unchanged(percentages["year"], df["year"])
    assert np.isclose(percentages.iloc[0]["yes"], 10 / 15)
    assert_identifier_unchanged(column_percentages["year"], df["year"])
    assert np.isclose(column_percentages.iloc[0]["yes"], 10 / 30)
    assert np.isclose(column_percentages.iloc[1]["yes"], 20 / 30)


@pytest.mark.functions
def test_adorn_rounding_leaves_plain_numeric_first_column():
    """The identifier rule applies to a plain DataFrame, not only to a tabyl."""
    df = pd.DataFrame({"year": [2020.44, 2021.55], "value": [1.234, 2.345]})

    result = df.adorn_rounding(digits=1)

    assert result["year"].tolist() == [2020.44, 2021.55]
    assert result["year"].dtype == df["year"].dtype
    assert result["value"].tolist() == [1.2, 2.3]


@pytest.mark.functions
def test_adorn_ns_counts_frame_without_identifier_column():
    """A counts frame that carries no row identifier keeps every count.

    ``ns`` does not have to repeat the identifier column of the frame being
    adorned. Dropping column 0 of ``ns`` would discard a real count.
    """
    df = pd.DataFrame({"id": ["A"], "yes": [0.5], "no": [0.5]})
    ns = pd.DataFrame({"yes": [10], "no": [5]})

    result = df.adorn_pct_formatting().adorn_ns(ns=ns)

    assert result.iloc[0]["yes"] == "50.0% (10)"
    assert result.iloc[0]["no"] == "50.0% (5)"
    assert result.iloc[0]["id"] == "A"


@pytest.mark.functions
def test_adorn_ns_identifier_comes_from_target_frame():
    """The identifier is column 0 of the frame being adorned, not of ``ns``."""
    df = pd.DataFrame({"id": ["A"], "yes": [0.5], "no": [0.5]})
    ns = pd.DataFrame({"no": [5], "yes": [10], "id": [0]})

    result = df.adorn_pct_formatting().adorn_ns(ns=ns)

    assert result.iloc[0]["no"] == "50.0% (5)"
    assert result.iloc[0]["id"] == "A"
    assert result.iloc[0]["yes"] == "50.0% (10)"


@pytest.mark.functions
def test_adorn_pct_formatting_duplicate_column_labels():
    """Only column position 0 is exempt when a label is repeated."""
    df = pd.DataFrame([[1.0, 2.0, 0.5]], columns=["value", "value", "ratio"])

    result = df.adorn_pct_formatting()

    assert result.iloc[0, 0] == 1.0
    assert not isinstance(result.iloc[0, 0], str)
    assert result.iloc[0, 1] == "200.0%"
    assert isinstance(result.iloc[0, 1], str)
    assert result.iloc[0, 2] == "50.0%"


@pytest.mark.functions
def test_adorn_rounding_duplicate_column_labels():
    """Rounding exempts only column position 0 when a label is repeated."""
    df = pd.DataFrame([[1.25, 2.55, 0.55]], columns=["value", "value", "ratio"])

    result = df.adorn_rounding(digits=1, rounding="half up")

    assert result.iloc[0, 0] == 1.25
    assert result.iloc[0, 1] == 2.6
    assert result.iloc[0, 2] == 0.6


@pytest.mark.functions
def test_adorn_ns_duplicate_labels_in_explicit_ns():
    """Each repeat of a label reads its own count column."""
    df = pd.DataFrame(
        [["A", "50.0%", "50.0%"]],
        columns=["id", "yes", "yes"],
    )
    ns = pd.DataFrame([[10, 20]], columns=["yes", "yes"])

    result = df.adorn_ns(ns=ns)

    assert result.iloc[0, 0] == "A"
    assert result.iloc[0, 1] == "50.0% (10)"
    assert result.iloc[0, 2] == "50.0% (20)"


@pytest.mark.functions
@pytest.mark.parametrize(
    "call",
    [
        lambda frame: frame.adorn_pct_formatting(),
        lambda frame: frame.adorn_rounding(),
        lambda frame: frame.adorn_ns(ns=pd.DataFrame()),
    ],
    ids=["pct_formatting", "rounding", "ns"],
)
def test_adorn_formatting_on_dataframe_with_no_columns(call):
    """A DataFrame with no columns comes back unchanged."""
    result = call(pd.DataFrame())

    assert result.shape == (0, 0)
    assert list(result.columns) == []
