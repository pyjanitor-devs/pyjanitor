import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def messy_spreadsheet():
    """Return a DataFrame mimicking a dirty spreadsheet export."""
    return pd.DataFrame(
        [
            ["Quarterly Sales Report", np.nan, np.nan],
            ["Generated on 2026 01 01", np.nan, np.nan],
            ["Customer ID", "Region", "Revenue"],
            ["C101", "North", 1500],
            ["C102", "South", 2300],
        ]
    )


@pytest.mark.functions
def test_find_header_default(messy_spreadsheet):
    """Detect first row without missing values."""
    row_idx = messy_spreadsheet.find_header()
    assert row_idx == 2


@pytest.mark.functions
def test_find_header_first_row_complete():
    """Detect header at row zero when top row is complete."""
    df = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 5, 6],
        }
    )
    assert df.find_header() == 0


@pytest.mark.functions
def test_find_header_target_first_column(messy_spreadsheet):
    """Search for target label in first column."""
    row_idx = messy_spreadsheet.find_header("Customer ID")
    assert row_idx == 2


@pytest.mark.functions
def test_find_header_target_by_column_index(messy_spreadsheet):
    """Search for target label by column integer position."""
    row_idx = messy_spreadsheet.find_header("Revenue", column=2)
    assert row_idx == 2


@pytest.mark.functions
def test_find_header_target_by_column_name():
    """Search for target label by existing column label."""
    df = pd.DataFrame(
        {
            "col_x": [np.nan, "ID", "1"],
            "col_y": [np.nan, "Amount", "500"],
        }
    )
    row_idx = df.find_header(target="Amount", column="col_y")
    assert row_idx == 1


@pytest.mark.functions
def test_find_header_target_integer_column_with_named_columns():
    """Search for target label by column integer position on named columns."""
    df = pd.DataFrame(
        {
            "alpha": ["Report", "ID", "1"],
            "beta": [np.nan, "Revenue", "200"],
        }
    )
    row_idx = df.find_header("Revenue", column=1)
    assert row_idx == 1


@pytest.mark.functions
def test_find_header_target_integer_column_not_found():
    """Target missing on column integer position raises ValueError."""
    df = pd.DataFrame(
        {
            "alpha": ["Report", "ID", "1"],
            "beta": [np.nan, "Revenue", "200"],
        }
    )
    with pytest.raises(ValueError, match="column position 1"):
        df.find_header("Missing", column=1)


@pytest.mark.functions
def test_find_header_non_dataframe_raises():
    """Non DataFrame input raises TypeError."""
    from janitor.functions.find_header import find_header

    with pytest.raises(TypeError):
        find_header("not a dataframe")


@pytest.mark.functions
def test_find_header_kwargs_syntax(messy_spreadsheet):
    """Support dictionary keyword argument matching R janitor syntax."""
    row_idx = messy_spreadsheet.find_header(**{"Region": 1})
    assert row_idx == 2


@pytest.mark.functions
def test_find_header_empty_df_raises():
    """Empty dataframe raises ValueError."""
    df = pd.DataFrame()
    with pytest.raises(ValueError, match="empty DataFrame"):
        df.find_header()


@pytest.mark.functions
def test_find_header_no_complete_row_raises():
    """Dataframe with all rows having missing values raises ValueError."""
    df = pd.DataFrame(
        [
            [np.nan, 1],
            [2, np.nan],
        ]
    )
    with pytest.raises(ValueError, match="No complete row found"):
        df.find_header()


@pytest.mark.functions
def test_find_header_target_not_found_raises(messy_spreadsheet):
    """Target missing in target column raises ValueError."""
    with pytest.raises(ValueError, match="not found"):
        messy_spreadsheet.find_header("NonexistentHeader")


@pytest.mark.functions
def test_find_header_invalid_column_raises(messy_spreadsheet):
    """Invalid column position or name raises ValueError."""
    with pytest.raises(ValueError, match="Column .* not found"):
        messy_spreadsheet.find_header("Revenue", column=99)


@pytest.mark.functions
def test_row_to_names_with_find_header(messy_spreadsheet):
    """Verify row to names integration when passing find header string."""
    cleaned = messy_spreadsheet.row_to_names(
        row_numbers="find_header",
        remove_rows=True,
        remove_rows_above=True,
        reset_index=True,
    )
    assert list(cleaned.columns) == ["Customer ID", "Region", "Revenue"]
    assert len(cleaned) == 2
    assert cleaned.iloc[0, 0] == "C101"
    assert cleaned.iloc[1, 2] == 2300


@pytest.mark.functions
def test_row_to_names_invalid_string_raises(messy_spreadsheet):
    """Verify row to names raises ValueError for unsupported string."""
    with pytest.raises(ValueError, match="Unsupported string option"):
        messy_spreadsheet.row_to_names("unsupported_option")


@pytest.mark.functions
def test_row_to_names_invalid_type_raises(messy_spreadsheet):
    """Verify row to names raises TypeError for unsupported type."""
    with pytest.raises(TypeError, match="row_numbers should be either"):
        messy_spreadsheet.row_to_names(3.14)
