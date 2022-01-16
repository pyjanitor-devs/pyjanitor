import os
import subprocess
from glob import glob
from io import StringIO
from openpyxl import load_workbook
from typing import Iterable, Union

import pandas as pd

from .errors import JanitorError
from .utils import deprecated_alias, check


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
    if len(files_path) == 0:
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
    if len(dfs_dict) == 0:
        raise ValueError("No CSV files to read with the given `files_path`")
    # Concatenate the dataframes if requested (default)
    col_names = list(dfs_dict.values())[0].columns  # noqa: PD011
    if not separate_df:
        # If columns do not match raise an error
        for df in dfs_dict.values():  # noqa: PD011
            if not all(df.columns == col_names):
                raise ValueError(
                    "Columns in input CSV files do not match."
                    "Files cannot be concatenated"
                )
        return pd.concat(
            list(dfs_dict.values()),
            ignore_index=True,
            sort=False,  # noqa: PD011
        )
    else:
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

    This function assumes that your command line command will return
    an output that is parsable using pandas.read_csv and StringIO.
    We default to using pd.read_csv underneath the hood.
    Keyword arguments are passed through to read_csv.
    ```

    :param cmd: Shell command to preprocess a file on disk.
    :param kwargs: Keyword arguments that are passed through to pd.read_csv().
    :raises JanitorError: If commandline command is malformed or invalid.
    :returns: A pandas DataFrame parsed from the stdout of the underlying
        shell.
    """

    check("cmd", cmd, [str])
    outcome = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if outcome.returncode != 0:
        raise JanitorError(outcome.stderr)
    else:
        outcome = outcome.stdout
    return pd.read_csv(StringIO(outcome), **kwargs)


def xlsx_table(
    path: str,
    sheetname: str,
    table: Union[str, list, tuple] = None,
    header: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame of values in a table in the Excel file.
    If the `table` argument is provided, a pandas DataFrame is returned;
    if the `table` argument is None, or a list/tuple of names,
    a dictionary of DataFrames is returned, where the keys of the dictionary
    are the table names.

    :param path: path to the Excel File.
    :param sheetname: Name of the sheet
        from which the tables are to be extracted.
    :param table: name of a table, or list of tables in the sheet.
    :raises ValueError: if there are not tables in the sheet.
    :param header: If the first row should be used as column names.
    :returns: A pandas DataFrame, or a dictionary of DataFrames.
    """

    wb = load_workbook(filename=path, read_only=False, keep_links=False)
    ws = wb[sheetname]

    contents = ws.tables
    if not contents:
        raise ValueError(f"There is no table in `{sheetname}` sheet.")
    contents = contents.items()

    if isinstance(table, str):
        table = [table]
    if table is not None:
        check("table", table, [list, tuple])

    if isinstance(table, (list, tuple)):
        for entry in table:
            if entry not in contents:
                raise ValueError(
                    f"""
                    {entry} is not a table
                    in the {sheetname} sheet.
                    """
                )
        contents = ((key, value) for key, value in contents if key in table)

    frame = {}
    for key, value in contents:
        content = ((cell.value for cell in row) for row in ws[value])
        if header:
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
