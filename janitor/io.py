"""Convenience I/O functions."""
import os
from glob import glob

import pandas as pd


def read_csvs(filespath: str, seperate_df: bool = False, **kwargs):
    """
    Read multiple CSV files as a single DataFrame.

    :param filespath: The filepath pattern matching the CSVs files.
        Accepts regular expressions, with or without csv extension.
    :param seperate_df: If False (default) returns a single Dataframe
        with the concatenation of the csv files.
        If True, returns a dictionary of seperate dataframes
        for each CSV file.
    :param kwargs: Keyword arguments to pass into the
        original pandas `read_csv`.
    """
    # Sanitize input
    assert filespath is not None
    assert len(filespath) != 0

    # Read the csv files
    dfs = {
        os.path.basename(f): pd.read_csv(f, **kwargs) for f in glob(filespath)
    }
    # Check if dataframes have been read
    if len(dfs) == 0:
        raise ValueError("No CSV files to read with the given filespath")
    # Concatenate the dataframes if requested (default)
    col_names = list(dfs.values())[0].columns
    if not seperate_df:
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
