from glob import glob
import os
import pandas as pd


def read_csvs(filespath: str, seperate_df: bool = False, **kwargs):
    """
    :param filespath: The string pattern matching the CSVs files. Accepts regular expressions, with or without csv extension
    :param seperate_df: If False (default) returns a single Dataframe with the concatenation of the csv files-
        If True, returns a dictionary of seperate dataframes for each CSV file.
    :param kwargs: Keyword arguments to pass into the original pandas `read_csv`.
    """
    # Sanitize input
    assert filespath is not None
    assert len(filespath) != 0

    # Check if the original filespath contains .csv
    if not filespath.endswith(".csv"):
        filespath += ".csv"
    # Read the csv files
    dfs = {
        os.path.basename(f): pd.read_csv(f, **kwargs) for f in glob(filespath)
    }
    # Check if dataframes have been read
    if len(dfs) == 0:
        raise ValueError("No CSV files to read with the given filespath")
    # Concatenate the dataframes if requested (default)
    if seperate_df:
        return dfs
    else:
        try:
            return pd.concat(list(dfs.values()), ignore_index=True, sort=False)
        except:
            raise ValueError("Input CSV files cannot be concatenated")
