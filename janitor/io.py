import os
import subprocess
from glob import glob
from io import StringIO
from typing import Iterable, Union

import pandas as pd

from .errors import JanitorError
from .utils import deprecated_alias, check, import_message


@deprecated_alias(seperate_df="separate_df", filespath="files_path")
def read_csvs(
    files_path: Union[str, Iterable[str]], separate_df: bool = False, **kwargs
) -> Union[pd.DataFrame, dict]:
    """
    Read multiple CSV files and return a dictionary of DataFrames, or
    one concatenated DataFrame.

    :param files_path: The filepath pattern matching the CSV files.
        Accepts regular expressions, with or without `.csv` extension.
        Also accepts iterable of file paths.
    :param separate_df: If `False` (default), returns a single Dataframe
        with the concatenation of the csv files.
        If `True`, returns a dictionary of separate DataFrames
        for each CSV file.
    :param kwargs: Keyword arguments to pass into the
        original pandas `read_csv`.
    :returns: DataFrame of concatenated DataFrames or dictionary of DataFrames.
    :raises JanitorError: if `None` provided for `files_path`.
    :raises JanitorError: if length of `files_path` is `0`.
    :raises ValueError: if no CSV files exist in `files_path`.
    :raises ValueError: if columns in input CSV files do not match.
    """
    # Sanitize input
    if files_path is None:
        raise JanitorError("`None` provided for `files_path`")
    if not files_path:
        raise JanitorError("0 length `files_path` provided")

    # Read the csv files
    # String to file/folder or file pattern provided
    if isinstance(files_path, str):
        dfs_dict = {
            os.path.basename(f): pd.read_csv(f, **kwargs)
            for f in glob(files_path)
        }
    # Iterable of file paths provided
    else:
        dfs_dict = {
            os.path.basename(f): pd.read_csv(f, **kwargs) for f in files_path
        }
    # Check if dataframes have been read
    if not dfs_dict:
        raise ValueError("No CSV files to read with the given `files_path`")
    # Concatenate the dataframes if requested (default)
    col_names = list(dfs_dict.values())[0].columns  # noqa: PD011
    if not separate_df:
        # If columns do not match raise an error
        for df in dfs_dict.values():  # noqa: PD011
            if not all(df.columns == col_names):
                raise ValueError(
                    "Columns in input CSV files do not match."
                    "Files cannot be concatenated."
                )
        return pd.concat(
            list(dfs_dict.values()),
            ignore_index=True,
            sort=False,  # noqa: PD011
            copy=False,
        )
    return dfs_dict


def read_commandline(cmd: str, **kwargs) -> pd.DataFrame:
    """
    Read a CSV file based on a command-line command.

    For example, you may wish to run the following command on `sep-quarter.csv`
    before reading it into a pandas DataFrame:

    ```bash
    cat sep-quarter.csv | grep .SEA1AA
    ```

    In this case, you can use the following Python code to load the dataframe:

    ```python
    import janitor as jn
    df = jn.read_commandline("cat data.csv | grep .SEA1AA")
    ```

    This function assumes that your command line command will return
    an output that is parsable using `pandas.read_csv` and StringIO.
    We default to using `pd.read_csv` underneath the hood.
    Keyword arguments are passed through to read_csv.

    :param cmd: Shell command to preprocess a file on disk.
    :param kwargs: Keyword arguments that are passed through to
        `pd.read_csv()`.
    :returns: A pandas DataFrame parsed from the stdout of the underlying
        shell.
    """

    check("cmd", cmd, [str])
    # adding check=True ensures that an explicit, clear error
    # is raised, so that the user can see the reason for the failure
    outcome = subprocess.run(
        cmd, shell=True, capture_output=True, text=True, check=True
    )
    return pd.read_csv(StringIO(outcome.stdout), **kwargs)


def xlsx_table(
    path: str,
    sheetname: str,
    table: Union[str, list, tuple] = None,
) -> Union[pd.DataFrame, dict]:
    """
    Returns a DataFrame of values in a table in the Excel file.
    This applies to an Excel file, where the data range is explicitly
    specified as a Microsoft Excel table.

    If there is a single table in the sheet, or a string is provided
    as an argument to the `table` parameter, a pandas DataFrame is returned;
    if there is more than one table in the sheet,
    and the `table` argument is `None`, or a list/tuple of names,
    a dictionary of DataFrames is returned, where the keys of the dictionary
    are the table names.

    Example:

    ```python

    filename = "excel_table.xlsx"

    # single table
    jn.xlsx_table(filename, sheetname='Tables', table = 'dCategory')

        CategoryID      Category
    0           1       Beginner
    1           2       Advanced
    2           3      Freestyle
    3           4    Competition
    4           5  Long Distance

    # multiple tables:
    jn.xlsx_table(filename, sheetname = 'Tables', table = ['dCategory', 'dSupplier'])

    {'dCategory':    CategoryID       Category
                    0           1       Beginner
                    1           2       Advanced
                    2           3      Freestyle
                    3           4    Competition
                    4           5  Long Distance,
    'dSupplier':   SupplierID             Supplier        City State                         E-mail
                    0         GB       Gel Boomerangs     Oakland    CA          gel@gel-boomerang.com
                    1         CO  Colorado Boomerangs    Gunnison    CO  Pollock@coloradoboomerang.com
                    2         CC        Channel Craft    Richland    WA                    Dino@CC.com
                    3         DB        Darnell Booms  Burlington    VT            Darnell@Darnell.com}
    ```

    :param path: Path to the Excel File.
    :param sheetname: Name of the sheet from which the tables
        are to be extracted.
    :param table: Name of a table, or list of tables in the sheet.
    :returns: A pandas DataFrame, or a dictionary of DataFrames,
        if there are multiple arguments for the `table` parameter,
        or the argument to `table` is `None`.
    :raises ValueError: If there are no tables in the sheet.

    """  # noqa : E501

    try:
        from openpyxl import load_workbook
    except ImportError:
        import_message(
            submodule="io",
            package="openpyxl",
            conda_channel="conda-forge",
            pip_install=True,
        )
    wb = load_workbook(filename=path, read_only=False, keep_links=False)
    ws = wb[sheetname]

    contents = ws.tables
    if not contents:
        raise ValueError(f"There is no table in `{sheetname}` sheet.")

    if isinstance(table, str):
        table = [table]
    if table is not None:
        check("table", table, [list, tuple])
        for entry in table:
            if entry not in contents:
                raise ValueError(
                    f"{entry} is not a table in the {sheetname} sheet."
                )
        data = (
            (key, value) for key, value in contents.items() if key in table
        )
    else:
        data = contents.items()

    frame = {}
    for key, value in data:
        content = ((cell.value for cell in row) for row in ws[value])
        if contents[key].headerRowCount == 1:
            column_names = next(content)
            content = zip(*content)
            frame[key] = dict(zip(column_names, content))
        else:
            content = zip(*content)
            frame[key] = {f"C{num}": val for num, val in enumerate(content)}

    if len(frame) == 1:
        _, frame = frame.popitem()
        return pd.DataFrame(frame)
    return {key: pd.DataFrame(value) for key, value in frame.items()}


def xlsx_cells(
    path: str,
    sheetname: Union[str, list, tuple] = None,
    start_point: Union[str, int] = None,
    end_point: Union[str, int] = None,
    internal_value: bool = False,
    coordinate: bool = False,
    row: bool = False,
    column: bool = False,
    data_type: bool = False,
    is_date: bool = False,
    number_format: bool = False,
    fill_type: bool = False,
    fg_color: bool = False,
    bg_color: bool = False,
    font_name: bool = False,
    font_size: bool = False,
    font_bold: bool = False,
    font_italic: bool = False,
    font_vertalign: bool = False,
    font_underline: bool = False,
    font_strike: bool = False,
    font_color: bool = False,
    font_outline: bool = False,
    font_shadow: bool = False,
    font_condense: bool = False,
    font_extend: bool = False,
    font_charset: bool = False,
    font_family: bool = False,
    left_border_style: bool = False,
    right_border_style: bool = False,
    top_border_style: bool = False,
    bottom_border_style: bool = False,
    diagonal_border_style: bool = False,
    horizontal_alignment: bool = False,
    vertical_alignment: bool = False,
    text_rotation: bool = False,
    wrap_text: bool = False,
    shrink_to_fit: bool = False,
    indent: bool = False,
    locked: bool = False,
    hidden: bool = False,
    comments: bool = False,
) -> pd.DataFrame:
    """

    :param path: Path to the Excel File. Can also be an openpyxl Workbook.
    :param sheetname: Name of the sheet from which the cells
        are to be extracted.
    :param start_point: start coordinates of the Excel sheet. This is useful
        if the user is only interested in a subsection of the sheet.
    :param end_point: end coordinates of the Excel sheet. This is useful
        if the user is only interested in a subsection of the sheet.
    :param internal_value: internal value in cell.
    :param coordinate: The cell's position.
    :param row: Row position of the cell.
    :param column: Column position of the cell.
    :param data_type: Data type of the cell.
    :param is_date: Checks if the cell is a date.
    :param number_format: Number format, if the cell is a number.
    :param fill_type: fill type of the cell.
    :param fg_color: foreground color of the cell.
    :param bg_color: background color of the cell.
    :param font_name: name of the cell font.
    :param font_size: size of the cell font.
    :param font_bold: is the cell's font bold?
    :param font_italic: is the cell's font italicized?
    :param font_vertalign: superscript or subscript?.
    :param font_underline: if the cell has an underline.
    :param font_strike: if the cell has a strikethrough.
    :param font_color: color of the cell's font.
    :param font_outline: outline of the cell.
    :param font_shadow: shadow of the cell.
    :param font_condense: cell font characteristics.
    :param font_extend: cell font characteristics.
    :param font_charset: cell font charset.
    :param font_family: family to which the cell font belongs.
    :param left_border_style: cell border style.
    :param right_border_style: cell border style.
    :param top_border_style: cell border style.
    :param bottom_border_style: cell border style.
    :param diagonal_border_style: cell border style.
    :param horizontal_alignment: cell alignment.
    :param vertical_alignment: cell alignment.
    :param text_rotation: text position.
    :param wrap_text: cell wrap_text argument.
    :param shrink_to_fit: cell shrink to fit argument.
    :param indent: cell indented?
    :param locked: locked cell?
    :param hidden: hidden cell?
    :param comments: are comments in the cell?
    :returns: A pandas DataFrame.
    """  # noqa : E501

    try:
        from openpyxl import load_workbook

        # from openpyxl.cell.read_only import EmptyCell
        from openpyxl.workbook.workbook import Workbook
    except ImportError:
        import_message(
            submodule="io",
            package="openpyxl",
            conda_channel="conda-forge",
            pip_install=True,
        )
    from collections import defaultdict
    from itertools import chain, compress

    if isinstance(path, Workbook):
        ws = path[sheetname]
    else:
        # for efficiency, read_only is set to False
        # if comments is True, read_only has to be True,
        # as lazy loading is not enabled for comments
        wb = load_workbook(filename=path, read_only=comments, keep_links=False)
        ws = wb[sheetname]
    # start_range and end_range applies if the user is interested in
    # only a subset of the Excel File and knows the coordinates
    if start_point or end_point:
        check("start_point", start_point, [str, int])
        check("end_point", end_point, [str, int])
        ws = ws[start_point:end_point]
    ws = chain.from_iterable(ws)
    frame = defaultdict(list)
    for cell in ws:
        # default values
        value = getattr(cell, "value", None)
        frame["value"].append(value)

        ########### position and data type ####################   # noqa : E266
        cell_arguments = (
            internal_value,
            coordinate,
            row,
            column,
            data_type,
            is_date,
            number_format,
        )

        if any(cell_arguments):
            names = (
                "internal_value",
                "coordinate",
                "row",
                "column",
                "data_type",
                "is_date",
                "number_format",
            )
            cell_arguments = compress(names, cell_arguments)

            for entry in cell_arguments:
                value = getattr(cell, entry, None)
                frame[entry].append(value)

        ########### font info ####################   # noqa : E266
        cell_arguments = (
            font_name,
            font_size,
            font_bold,
            font_italic,
            font_vertalign,
            font_underline,
            font_strike,
            font_color,
            font_outline,
            font_shadow,
            font_condense,
            font_extend,
            font_charset,
            font_family,
        )

        if any(cell_arguments):
            names = [
                "name",
                "size",
                "bold",
                "italic",
                "vertAlign",
                "underline",
                "strike",
                "color",
                "outline",
                "shadow",
                "condense",
                "extend",
                "charset",
                "family",
            ]
            cell_arguments = compress(names, cell_arguments)

            for entry in cell_arguments:
                cell_format = getattr(cell, "font", None)
                if cell_format is None:
                    value = None
                else:
                    value = getattr(cell_format, entry, None)
                if (value is not None) and (entry == "color"):
                    value = value.rgb
                frame[f"font_{entry.lower()}"].append(value)

    return pd.DataFrame(frame, copy=False)
