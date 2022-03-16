import pytest

from janitor import io
from pathlib import Path
from openpyxl import load_workbook


TEST_DATA_DIR = "tests/test_data"
filename = Path(TEST_DATA_DIR).joinpath("worked-examples.xlsx").resolve()


wb = load_workbook(filename)
wb_c = load_workbook(filename, read_only=True)


def test_check_sheetname():
    """Raise error if sheet name does not exist."""
    with pytest.raises(KeyError):
        io.xlsx_cells(filename, 5)


def test_comment_read_only():
    """Raise error if comments and read_only is True."""
    with pytest.raises(ValueError):
        io.xlsx_cells("excel.xlsx", "clean", comments=True)


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


def test_output_kwargs_defaults():
    """
    Raise AttributeError if kwargs is provided
    and the key is already part of the default attributes
    that are returned as columns.
    """
    with pytest.raises(ValueError):
        io.xlsx_cells(wb, "clean", internal_value=True)


def test_output_kwargs_defaults_read_only_true():
    """
    Raise AttributeError if kwargs is provided
    and the key is already part of the default attributes
    that are returned as columns.
    """
    with pytest.raises(ValueError):
        io.xlsx_cells(wb_c, "clean", internal_value=True)


def test_output_wrong_kwargs():
    """
    Raise AttributeError if kwargs is provided
    and the key is not a valid openpyxl cell attribute.
    """
    with pytest.raises(AttributeError):
        io.xlsx_cells(filename, "clean", font_color=True)


def test_output_wrong_kwargs_read_only_false():
    """
    Raise AttributeError if kwargs is provided
    and the key is not a valid openpyxl cell attribute.
    """
    with pytest.raises(AttributeError):
        io.xlsx_cells(filename, "clean", font_color=True, read_only=False)


def test_default_values():
    """
    Test output for default values.
    """
    assert "Matilda" in io.xlsx_cells(wb, "clean")["value"].tolist()


def test_default_values_blank_cells_true():
    """
    Test output for default values if include_blank_cells.
    """
    assert None in io.xlsx_cells(filename, "pivot-notes")["value"].tolist()


def test_default_values_blank_cells_false():
    """
    Test output for default values if include_blank_cells is False.
    """
    assert (
        None
        not in io.xlsx_cells(
            wb,
            "pivot-notes",
            include_blank_cells=False,
            start_point="A1",
            end_point="G7",
        )["value"].tolist()
    )


wb_c.close()
