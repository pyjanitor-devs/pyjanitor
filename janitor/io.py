import os
import subprocess
from glob import glob
from typing import Iterable, Union

import pandas as pd

from .errors import JanitorError
from .utils import deprecated_alias
from io import StringIO


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
            list(dfs_dict.values()),  # noqa: PD011
            ignore_index=True,
            sort=False,
        )
    else:
        return dfs_dict


def read_commandline(cmd: str) -> pd.DataFrame:
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
    df = jn.read_commandline("cat sep-quarter.csv | grep .SEA1AA")
    ```

    :param cmd: Shell command to preprocess a file on disk.
    :returns: A pandas DataFrame parsed from the stdout of the underlying
        shell.
    :raises EmptyDataError: If there is no data to parse, this often happens
        because the cmd param is either an invalid bash command, thus
        nothing happens in the shell , or if cmd param is not a string,
        thus creating an invalid shell command.
    """
    # cmd = cmd.split(" ")
    try:
        outcome = subprocess.run(
            cmd, shell=True, capture_output=True, text=True
        )
        outcome = outcome.stdout
    except pd.EmptyDataError:
        msg = (
            "Empty Data Error: Be sure your parameter"
            " is both a valid shell command and a string"
        )
        raise pd.EmptyDataError(msg)

    return pd.read_csv(StringIO(outcome))
