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
    read_only: bool = True,
    include_blank_cells: bool = True,
    fill: bool = False,
    font: bool = False,
    alignment: bool = False,
    border: bool = False,
    protection: bool = False,
    comment: bool = False,
    **kwargs,
) -> pd.DataFrame:
    """

    :param path: Path to the Excel File. Can also be an openpyxl Workbook.
    :param sheetname: Name of the sheet from which the cells are to be extracted.
        If `None`, all the sheets in the file are extracted;
        if a string, or list or tuple, ,only the specified sheets are extracted.
    :param start_point: start coordinates of the Excel sheet. This is useful
        if the user is only interested in a subsection of the sheet.
    :param end_point: end coordinates of the Excel sheet. This is useful
        if the user is only interested in a subsection of the sheet.
    :param read_only: Determines if the entire file is loaded in memory,
        or streamed. For memory efficiency, read_only should be set to `False`.
        Some cell properties like `comments`, can only be accessed by
        setting `read_only` to `True`.
    :param include_blank_cells: Determines if empty cells should be included.
    :param fill: If `True`, return fill properties of the cell.
    :param font: If `True`, return font properties of the cell.
    :param alignment: If `True`, return alignment properties of the cell.
    :param border: If `True`, return border properties of the cell.
    :param protection: If `True`, return protection properties of the cell.
    :param comment: If `True`, return comments properties of the cell.
    :param kwargs: Any other attributes of the cell, that can be accessed from openpyxl.
    :raises ValueError: If kwargs is provided, and one of the keys is a default column.
    :raises AttributeError: If kwargs is provided and any of the keys
        is not a openpyxl cell attribute.
    :returns: A pandas DataFrame.
    """  # noqa : E501

    try:
        from openpyxl import load_workbook
        from openpyxl.cell.read_only import ReadOnlyCell
        from openpyxl.cell.cell import Cell
        from openpyxl.workbook.workbook import Workbook
        import inspect
    except ImportError:
        import_message(
            submodule="io",
            package="openpyxl",
            conda_channel="conda-forge",
            pip_install=True,
        )
    from collections import defaultdict
    from itertools import chain

    path_is_workbook = isinstance(path, Workbook)
    if path_is_workbook:
        ws = path[sheetname]
    else:
        # for memory efficiency, read_only is set to True
        # if comments is True, read_only has to be False,
        # as lazy loading is not enabled for comments
        if comment and read_only:
            raise ValueError(
                "To access comments, kindly set 'read_only' to False."
            )
        wb = load_workbook(
            filename=path, read_only=read_only, keep_links=False
        )
        ws = wb[sheetname]
    # start_point and end_point applies if the user is interested in
    # only a subset of the Excel File and knows the coordinates
    if start_point or end_point:
        check("start_point", start_point, [str, int])
        check("end_point", end_point, [str, int])
        ws = ws[start_point:end_point]
    ws = chain.from_iterable(ws)
    frame = defaultdict(list)
    defaults = (
        "value",
        "internal_value",
        "coordinate",
        "row",
        "column",
        "data_type",
        "is_date",
        "number_format",
    )
    parameters = {
        "fill": fill,
        "font": font,
        "alignment": alignment,
        "border": border,
        "protection": protection,
        "comment": comment,
    }
    if kwargs:
        if path_is_workbook:
            if path.read_only:
                _cell = ReadOnlyCell
            else:
                _cell = Cell
        else:
            if read_only:
                _cell = ReadOnlyCell
            else:
                _cell = Cell

        attrs = {
            attr
            for attr, _ in inspect.getmembers(_cell, not (inspect.isroutine))
            if not attr.startswith("_")
        }

        for key in kwargs:
            if key in defaults:
                raise ValueError(
                    f"{key} is part of the default attributes "
                    "returned as a column."
                )
            elif key not in attrs:
                raise AttributeError(
                    f"{key} is not a recognized attribute of {_cell}."
                )
        parameters.update(kwargs)
    for cell in ws:
        if (cell.value is None) and (not include_blank_cells):
            continue
        for value in defaults:
            frame[value].append(getattr(cell, value, None))
        for parent, boolean_value in parameters.items():
            check(f"The value for {parent}", boolean_value, [bool])
            if not boolean_value:
                continue
            boolean_value = object_to_dict(getattr(cell, parent, None))
            frame[parent].append(boolean_value)

    if (not path_is_workbook) and wb.read_only:
        wb.close()
    return pd.DataFrame(frame, copy=False)


def object_to_dict(obj):
    """
    Recursively get the attributes
    of a class as a dictionary.

    :param obj: Object whose attributes are to be extracted.
    :Returns: A dictionary or the object.
    """
    # https://stackoverflow.com/a/71366813/7175713
    data = {}
    if getattr(obj, "__dict__", None):
        for key, value in obj.__dict__.items():
            data[key] = object_to_dict(value)
        return data
    return obj
