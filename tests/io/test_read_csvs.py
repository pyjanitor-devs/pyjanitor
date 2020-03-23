import glob
import os

import pandas as pd
import pytest

from janitor import io

CSV_FILE_PATH = "my_test_csv_for_read_csvs_{}.csv"


def create_csv_file(number_of_files, col_names=None):
    for i in range(number_of_files):
        filename = CSV_FILE_PATH.format(i)
        df = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        df.to_csv(filename, index=False)


def remove_csv_files():
    # Get a list of all the file paths matching pattern in specified directory
    fileList = glob.glob(CSV_FILE_PATH.format("*"))

    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        os.remove(filePath)


@pytest.mark.functions
def test_read_csvs_one_csv_path():
    # Setup
    # When a CSV file with 3 cols and 4 rows is on disk
    number_of_files = 1
    create_csv_file(number_of_files)

    # If the csv file is read into DataFrame
    df = io.read_csvs(CSV_FILE_PATH.format("*"))

    # Then the dataframe has 3 cols and 4 rows
    try:
        assert len(df.columns) == 3
        assert len(df) == 4
    finally:
        # Cleanup
        remove_csv_files()


@pytest.mark.functions
def test_read_csvs_zero_csv_path():
    # Setup
    # When no CSV files are on disk

    # When reading files the functions raises ValueError.
    try:
        io.read_csvs("nofilesondisk.csv")
        raise Exception
    except ValueError:
        pass
    finally:
        remove_csv_files()


@pytest.mark.functions
def test_read_csvs_three_csv_path():
    # Setup
    # When a CSV file with 3 cols and 4 rows is on disk
    number_of_files = 3
    create_csv_file(number_of_files)

    # If the csv file is read into DataFrame
    df = io.read_csvs(CSV_FILE_PATH.format("*"))

    # Then the dataframe has 3 cols and 12 rows
    try:
        assert len(df.columns) == 3
        assert len(df) == 4 * number_of_files
    finally:
        # Cleanup
        remove_csv_files()


@pytest.mark.functions
def test_read_csvs_three_separated_csv_path():
    # Setup
    # When a CSV file with 3 cols and 4 rows is on disk
    number_of_files = 3
    create_csv_file(number_of_files)

    # If the csv file is read into DataFrame
    dfs_dict = io.read_csvs(CSV_FILE_PATH.format("*"), separate_df=True)

    # Then the dataframe list has 3 dataframes
    try:
        assert len(dfs_dict) == number_of_files
        for df in dfs_dict.values():  # noqa: PD011
            assert len(df) == 4
            assert len(df.columns) == 3
    finally:
        # Cleanup
        remove_csv_files()


@pytest.mark.functions
def test_read_csvs_two_unmatching_csv_files():
    # Setup
    # When two csv files do not have same column names
    df = pd.DataFrame(
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"]
    )
    df.to_csv(CSV_FILE_PATH.format(0), index=False)
    df = pd.DataFrame(
        [[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["d", "e", "f"]
    )
    df.to_csv(CSV_FILE_PATH.format(1), index=False)

    # If the csv files are read into DataFrame
    try:
        io.read_csvs(CSV_FILE_PATH.format("*"))
        # if read does read the unmatching files give an error
        raise ValueError
    except ValueError:
        # If the read raises an exception it is ok
        pass
    finally:
        remove_csv_files()


@pytest.mark.functions
def test_read_csvs_lists():
    # Setup
    # When a CSV file with 3 cols and 4 rows is on disk
    number_of_files = 3
    create_csv_file(number_of_files)
    csvs_list = [CSV_FILE_PATH.format(i) for i in range(number_of_files)]

    # If the list of csv files is read into DataFrame
    dfs_list = io.read_csvs(files_path=csvs_list, separate_df=True)

    # Then the dataframe list has 3 dataframes
    try:
        assert len(dfs_list) == number_of_files
        for df in dfs_list.values():  # noqa: PD011
            assert len(df) == 4
            assert len(df.columns) == 3
    finally:
        # Cleanup
        remove_csv_files()
