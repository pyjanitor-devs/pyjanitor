import os
from glob import glob
from typing import Iterable, Union

import pandas as pd

from .errors import JanitorError
from .utils import deprecated_alias


@deprecated_alias(seperate_df="separate_df", filespath="files_path")
def read_csvs(
    files_path: Union[str, Iterable[str]], separate_df: bool = False, **kwargs
) -> Union[pd.DataFrame, dict]:
    """
    :param files_path: The filepath pattern matching the CSVs files.
        Accepts regular expressions, with or without csv extension.
        Also accepts iterable of file paths.
    :param separate_df: If False (default) returns a single Dataframe
        with the concatenation of the csv files.
        If True, returns a dictionary of separate dataframes
        for each CSV file.
    :param kwargs: Keyword arguments to pass into the
        original pandas `read_csv`.
    """
    # Sanitize input
    if files_path is None:
        raise JanitorError("`None` provided for `files_path`")
    if len(files_path) == 0:
        raise JanitorError("0 length `files_path` provided")

    # Read the csv files
    # String to file/folder or file pattern provided
    if isinstance(files_path, str):
        dfs = {
            os.path.basename(f): pd.read_csv(f, **kwargs)
            for f in glob(files_path)
        }
    # Iterable of file paths provided
    else:
        dfs = {
            os.path.basename(f): pd.read_csv(f, **kwargs) for f in files_path
        }
    # Check if dataframes have been read
    if len(dfs) == 0:
        raise ValueError("No CSV files to read with the given `files_path`")
    # Concatenate the dataframes if requested (default)
    col_names = list(dfs.values())[0].columns
    if not separate_df:
        # If columns do not match raise an error
        for df in dfs.values():
            if not all(df.columns == col_names):
                raise ValueError(
                    "Columns in input CSV files do not match."
                    "Files cannot be concatenated"
                )
        return pd.concat(list(dfs.values()), ignore_index=True, sort=False)
    else:
        return dfs
