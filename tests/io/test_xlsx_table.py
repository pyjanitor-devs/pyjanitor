import os
import pandas as pd
import pytest

from janitor import io
from pandas.testing import assert_frame_equal


TEST_DATA_DIR = "tests/test_data"
filename = os.path.join(TEST_DATA_DIR, "016-MSPTDA-Excel.xlsx")
no_headers = os.path.join(TEST_DATA_DIR, "excel_without_headers.xlsx")


def test_check_sheetname():
    """Test sheetname is a string."""
    with pytest.raises(TypeError):
        io.xlsx_table(filename, 1, None)


def test_check_filename():
    """Raise error if file does not exist."""
    with pytest.raises(FileNotFoundError):
        io.xlsx_table("excel.xlsx", 1, None)


def test_table_exists():
    """Raise error if there is no table in the sheet."""
    with pytest.raises(ValueError):
        io.xlsx_table(filename, "Cover")


def test_table_name():
    """
    Raise error if `table` is not None,
    and the table name cannot be found.
    """
    with pytest.raises(ValueError):
        io.xlsx_table(filename, "Tables", table="fake")


def test_table_str():
    """Test output for single table."""
    expected = io.xlsx_table(filename, "Tables", "dSupplier")
    actual = (
        pd.read_excel(
            filename, engine="openpyxl", sheet_name="Tables", usecols="N:R"
        )
        .rename(columns={"SupplierID.1": "SupplierID"})
        .dropna()
    )
    assert_frame_equal(expected, actual)


def test_table_no_header():
    """Test output for single table, without header."""
    expected = io.xlsx_table(no_headers, "Tables", "dSalesReps")
    actual = pd.read_excel(
        no_headers,
        engine="openpyxl",
        sheet_name="Tables",
        usecols="A:C",
        names=["C0", "C1", "C2"],
    )
    assert_frame_equal(expected, actual)


def test_tables():
    """Test output for multiple tables."""
    expected = io.xlsx_table(filename, "Tables", ("dSalesReps", "dSupplier"))
    actual = {
        "dSalesReps": pd.read_excel(
            filename, engine="openpyxl", sheet_name="Tables", usecols="A:C"
        ),
        "dSupplier": pd.read_excel(
            filename, engine="openpyxl", sheet_name="Tables", usecols="N:R"
        )
        .rename(columns={"SupplierID.1": "SupplierID"})
        .dropna(),
    }
    for key, value in expected.items():
        assert_frame_equal(value, actual[key])


def test_tables_None():
    """Test output for multiple tables."""
    expected = io.xlsx_table(filename, "Tables")
    actual = {
        "dSalesReps": pd.read_excel(
            filename, engine="openpyxl", sheet_name="Tables", usecols="A:C"
        ),
        "dSupplier": pd.read_excel(
            filename, engine="openpyxl", sheet_name="Tables", usecols="N:R"
        )
        .rename(columns={"SupplierID.1": "SupplierID"})
        .dropna(),
        "dProduct": pd.read_excel(
            filename, engine="openpyxl", sheet_name="Tables", usecols="E:I"
        )
        .dropna()
        .astype({"ProductID": int, "CategoryID": int}),
        "dCategory": pd.read_excel(
            filename, engine="openpyxl", sheet_name="Tables", usecols="K:L"
        )
        .rename(columns={"CategoryID.1": "CategoryID"})
        .dropna()
        .astype({"CategoryID": int}),
    }
    for key, value in expected.items():
        assert_frame_equal(value, actual[key])
