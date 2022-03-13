import pytest

from janitor import io
from pathlib import Path
from openpyxl import load_workbook


TEST_DATA_DIR = "tests/test_data"
filename = Path(TEST_DATA_DIR).joinpath("worked-examples.xlsx").resolve()


wb = load_workbook(filename)


def test_check_sheetname():
    """Raise error if sheet name does not exist."""
    with pytest.raises(KeyError):
        io.xlsx_cells(filename, 5)


def test_check_filename():
    """Raise error if file does not exist."""
    with pytest.raises(FileNotFoundError):
        io.xlsx_cells("excel.xlsx", "clean")


def test_check_start_none():
    """
    Raise error if start_point is None,
    and end_point is not None.
    """
    with pytest.raises(TypeError):
        io.xlsx_cells(filename, "clean", start_point=None, end_point="B5")


def test_check_end_none():
    """
    Raise error if start_point is not None,
    and end_point is None.
    """
    with pytest.raises(TypeError):
        io.xlsx_cells(wb, "clean", start_point="A1", end_point=None)


def test_default_values():
    """
    Test output for default values.
    """
    output = io.xlsx_cells(wb, "clean").squeeze().tolist()

    assert "Matilda" in output


def test_output_position_info():
    """
    Test output for position/data type.
    """
    output = io.xlsx_cells(
        wb, "clean", start_point="A1", end_point="B5", data_type=True
    )["data_type"].tolist()

    assert "n" in output


def test_output_font_info():
    """
    Test output for font.
    """
    output = io.xlsx_cells(wb, "clean", font_color=True)["font_color"].tolist()

    assert "FF000000" in output
