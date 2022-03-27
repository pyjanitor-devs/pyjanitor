from __future__ import annotations
import os
import subprocess
from glob import glob
from io import StringIO
from typing import Iterable, Union, TYPE_CHECKING, NamedTuple


import pandas as pd
import inspect

from .errors import JanitorError
from .utils import deprecated_alias, check, import_message
from collections import defaultdict
from itertools import chain


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


if TYPE_CHECKING:
    from openpyxl import Workbook


def xlsx_table(
    path: Union[str, Workbook],
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

        >>> import pandas as pd
        >>> from janitor import xlsx_table
        >>> filename="../pyjanitor/tests/test_data/016-MSPTDA-Excel.xlsx"

        # single table
        >>> xlsx_table(filename, sheetname='Tables', table='dCategory')
           CategoryID       Category
        0           1       Beginner
        1           2       Advanced
        2           3      Freestyle
        3           4    Competition
        4           5  Long Distance

        # multiple tables:
        >>> out=xlsx_table(filename, sheetname="Tables", table=["dCategory", "dSalesReps"])
        >>> out["dCategory"]
           CategoryID       Category
        0           1       Beginner
        1           2       Advanced
        2           3      Freestyle
        3           4    Competition
        4           5  Long Distance
        >>> out["dSalesReps"].head(3)
           SalesRepID             SalesRep Region
        0           1  Sioux Radcoolinator     NW
        1           2        Tyrone Smithe     NE
        2           3         Chantel Zoya     SW

    :param path: Path to the Excel File. It can also be an openpyxl Workbook.
    :param sheetname: Name of the sheet from which the tables
        are to be extracted.
    :param table: Name of a table, or list of tables in the sheet.
    :returns: A pandas DataFrame, or a dictionary of DataFrames,
        if there are multiple arguments for the `table` parameter,
        or the argument to `table` is `None`.
    :raises AttributeError: If a workbook is provided, and is a ReadOnlyWorksheet.
    :raises ValueError: If there are no tables in the sheet.
    :raises KeyError: If the provided table does not exist in the sheet.


    """  # noqa : E501

    try:
        from openpyxl import load_workbook
        from openpyxl.workbook.workbook import Workbook
    except ImportError:
        import_message(
            submodule="io",
            package="openpyxl",
            conda_channel="conda-forge",
            pip_install=True,
        )
    if isinstance(path, Workbook):
        ws = path[sheetname]
    else:
        ws = load_workbook(
            filename=path, read_only=False, keep_links=False, data_only=True
        )
        ws = ws[sheetname]

    try:
        contents = ws.tables
    except AttributeError as error:
        raise AttributeError(
            "Accessing the tables is not supported for ReadOnlyWorksheet"
        ) from error

    if not contents:
        raise ValueError(f"There is no table in '{sheetname}' sheet.")

    class TableArgs(NamedTuple):
        """
        Named Tuple to easily index values
        from the tables in the sheet.
        """

        table_name: str
        ref: str
        headerRowCount: int

    if isinstance(table, str):
        table = [table]
    if table is not None:
        check("table", table, [str, list, tuple])
        try:
            data = []
            for key in table:
                outcome = TableArgs(
                    key, contents[key].ref, contents[key].headerRowCount
                )
                data.append(outcome)
        except KeyError as error:
            raise KeyError(
                f"Table {error} is not in the '{sheetname}' sheet."
            ) from error
    else:
        data = (
            TableArgs(key, contents[key].ref, contents[key].headerRowCount)
            for key in contents
        )

    frame = {}
    for table_arg in data:
        content = [[cell.value for cell in row] for row in ws[table_arg.ref]]

        if table_arg.headerRowCount:
            header, *content = content
        else:
            header = [f"C{num}" for num in range(len(content[0]))]
        frame[table_arg.table_name] = pd.DataFrame(
            content, columns=header, copy=False
        )

    if len(frame) == 1:
        _, frame = frame.popitem()
    return frame


def xlsx_cells(
    path: Union[str, Workbook],
    sheetnames: Union[str, list, tuple] = None,
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
) -> Union[dict, pd.DataFrame]:
    """
    Imports data from spreadsheet without coercing it into a rectangle.
    Each cell is represented by a row in a dataframe, and includes the
    cell's coordinates, the value, row and column position.
    The cell formatting (fill, font, border, etc) can also be accessed;
    usually this is returned as a dictionary in the cell, and the specific
    cell format attribute can be accessed using `pd.Series.str.get`.

    Inspiration for this comes from R's [tidyxl][link] package.
    [link]: https://nacnudus.github.io/tidyxl/reference/tidyxl.html

    Example:

        >>> import pandas as pd
        >>> from janitor import xlsx_cells
        >>> pd.set_option("display.max_columns", None)
        >>> pd.set_option("display.expand_frame_repr", False)
        >>> pd.set_option("max_colwidth", None)
        >>> filename = "../pyjanitor/tests/test_data/worked-examples.xlsx"

        # Each cell is returned as a row:
        >>> xlsx_cells(filename, sheetnames="highlights")
            value internal_value coordinate  row  column data_type  is_date number_format
        0     Age            Age         A1    1       1         s    False       General
        1  Height         Height         B1    1       2         s    False       General
        2       1              1         A2    2       1         n    False       General
        3       2              2         B2    2       2         n    False       General
        4       3              3         A3    3       1         n    False       General
        5       4              4         B3    3       2         n    False       General
        6       5              5         A4    4       1         n    False       General
        7       6              6         B4    4       2         n    False       General

        # Access cell formatting such as fill :
        >>> out=xlsx_cells(filename, sheetnames="highlights", fill=True).select_columns("value", "fill")
        >>> out
            value                                                                                                                                              fill
        0     Age     {'patternType': None, 'fgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}, 'bgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}}
        1  Height     {'patternType': None, 'fgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}, 'bgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}}
        2       1     {'patternType': None, 'fgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}, 'bgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}}
        3       2     {'patternType': None, 'fgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}, 'bgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}}
        4       3  {'patternType': 'solid', 'fgColor': {'rgb': 'FFFFFF00', 'type': 'rgb', 'tint': 0.0}, 'bgColor': {'rgb': 'FFFFFF00', 'type': 'rgb', 'tint': 0.0}}
        5       4  {'patternType': 'solid', 'fgColor': {'rgb': 'FFFFFF00', 'type': 'rgb', 'tint': 0.0}, 'bgColor': {'rgb': 'FFFFFF00', 'type': 'rgb', 'tint': 0.0}}
        6       5     {'patternType': None, 'fgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}, 'bgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}}
        7       6     {'patternType': None, 'fgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}, 'bgColor': {'rgb': '00000000', 'type': 'rgb', 'tint': 0.0}}

        # specific cell attributes can be accessed by using Pandas' series.str.get :
        >>> out.fill.str.get("fgColor").str.get("rgb")
        0    00000000
        1    00000000
        2    00000000
        3    00000000
        4    FFFFFF00
        5    FFFFFF00
        6    00000000
        7    00000000
        Name: fill, dtype: object

    :param path: Path to the Excel File. It can also be an openpyxl Workbook.
    :param sheetnames: Names of the sheets from which the cells are to be extracted.
        If `None`, all the sheets in the file are extracted;
        if it is a string, or list or tuple, only the specified sheets are extracted.
    :param start_point: start coordinates of the Excel sheet. This is useful
        if the user is only interested in a subsection of the sheet.
        If start_point is provided, end_point must be provided as well.
    :param end_point: end coordinates of the Excel sheet. This is useful
        if the user is only interested in a subsection of the sheet.
        If end_point is provided, start_point must be provided as well.
    :param read_only: Determines if the entire file is loaded in memory,
        or streamed. For memory efficiency, read_only should be set to `True`.
        Some cell properties like `comment`, can only be accessed by
        setting `read_only` to `False`.
    :param include_blank_cells: Determines if cells without a value should be included.
    :param fill: If `True`, return fill properties of the cell.
        It is usually returned as a dictionary.
    :param font: If `True`, return font properties of the cell.
        It is usually returned as a dictionary.
    :param alignment: If `True`, return alignment properties of the cell.
        It is usually returned as a dictionary.
    :param border: If `True`, return border properties of the cell.
        It is usually returned as a dictionary.
    :param protection: If `True`, return protection properties of the cell.
        It is usually returned as a dictionary.
    :param comment: If `True`, return comment properties of the cell.
        It is usually returned as a dictionary.
    :param kwargs: Any other attributes of the cell, that can be accessed from openpyxl.
    :raises ValueError: If kwargs is provided, and one of the keys is a default column.
    :raises AttributeError: If kwargs is provided and any of the keys
        is not a openpyxl cell attribute.
    :returns: A pandas DataFrame, or a dictionary of DataFrames.
    """  # noqa : E501

    try:
        from openpyxl import load_workbook
        from openpyxl.cell.read_only import ReadOnlyCell
        from openpyxl.cell.cell import Cell
        from openpyxl.workbook.workbook import Workbook
    except ImportError:
        import_message(
            submodule="io",
            package="openpyxl",
            conda_channel="conda-forge",
            pip_install=True,
        )

    path_is_workbook = isinstance(path, Workbook)
    if not path_is_workbook:
        # for memory efficiency, read_only is set to True
        # if comments is True, read_only has to be False,
        # as lazy loading is not enabled for comments
        if comment and read_only:
            raise ValueError(
                "To access comments, kindly set 'read_only' to False."
            )
        path = load_workbook(
            filename=path, read_only=read_only, keep_links=False
        )
    # start_point and end_point applies if the user is interested in
    # only a subset of the Excel File and knows the coordinates
    if start_point or end_point:
        check("start_point", start_point, [str, int])
        check("end_point", end_point, [str, int])

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

    if not sheetnames:
        sheetnames = path.sheetnames
    elif isinstance(sheetnames, str):
        sheetnames = [sheetnames]
    else:
        check("sheetnames", sheetnames, [str, list, tuple])

    out = {
        sheetname: _xlsx_cells(
            path[sheetname],
            defaults,
            parameters,
            start_point,
            end_point,
            include_blank_cells,
        )
        for sheetname in sheetnames
    }
    if len(out) == 1:
        _, out = out.popitem()

    if (not path_is_workbook) and path.read_only:
        path.close()

    return out


def _xlsx_cells(
    wb: Workbook,
    defaults: tuple,
    parameters: dict,
    start_point: Union[str, int],
    end_point: Union[str, int],
    include_blank_cells: bool,
):
    """
    Function to process a single sheet. Returns a DataFrame.

    :param wb: Openpyxl Workbook.
    :param defaults: Sequence of default cell attributes.
    :param parameters: Dictionary of cell attributes to be retrieved.
        that will always be returned as columns.
    :param start_point: start coordinates of the Excel sheet.
    :param end_point: end coordinates of the Excel sheet.
    :param include_blank_cells: Determines if empty cells should be included.
    :param path_is_workbook: True/False.
    :returns : A pandas DataFrame.
    """

    if start_point:
        wb = wb[start_point:end_point]
    wb = chain.from_iterable(wb)
    frame = defaultdict(list)

    for cell in wb:
        if (cell.value is None) and (not include_blank_cells):
            continue
        for value in defaults:
            frame[value].append(getattr(cell, value, None))
        for parent, boolean_value in parameters.items():
            check(f"The value for {parent}", boolean_value, [bool])
            if not boolean_value:
                continue
            boolean_value = _object_to_dict(getattr(cell, parent, None))
            frame[parent].append(boolean_value)

    return pd.DataFrame(frame, copy=False)


def _object_to_dict(obj):
    """
    Recursively get the attributes
    of an object as a dictionary.

    :param obj: Object whose attributes are to be extracted.
    :returns: A dictionary or the object.
    """
    # https://stackoverflow.com/a/71366813/7175713
    data = {}
    if getattr(obj, "__dict__", None):
        for key, value in obj.__dict__.items():
            data[key] = _object_to_dict(value)
        return data
    return obj
