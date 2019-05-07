from janitor import io
import pandas as pd
import os
import pytest
import glob

CSV_FILE_PATH = "my_test_csv_for_read_csvs_{}.csv"


def create_csv_file(number_of_files):
    for i in range(number_of_files):
        filename = CSV_FILE_PATH.format(i)
        df = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
        df.to_csv(filename, index=False)


def remove_csv_files():
    # Get a list of all the file paths that ends with .txt from in specified directory
    fileList = glob.glob(CSV_FILE_PATH.format("*"))

    # Iterate over the list of filepaths & remove each file.
    for filePath in fileList:
        try:
            os.remove(filePath)
        except:
            print("Error while deleting file : ", filePath)


@pytest.mark.functions
def test_read_csvs_one_csv_path():
    # Setup
    # When a CSV file with 3 cols and 3 rows is on disk
    create_csv_file(1)

    # If the csv file is read into DataFrame
    dfs = io.read_csvs(CSV_FILE_PATH.format("*"))

    # Then the dataframe has 3 cols and 3 rows
    try:
        assert len(dfs.columns) == 3
        assert len(dfs) == 3
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


@pytest.mark.functions
def test_read_csvs_three_csv_path():
    # Setup
    # When a CSV file with 3 cols and 3 rows is on disk
    create_csv_file(3)

    # If the csv file is read into DataFrame
    dfs = io.read_csvs(CSV_FILE_PATH.format("*"))

    # Then the dataframe has 3 cols and 9 rows
    try:
        assert len(dfs.columns) == 3
        assert len(dfs) == 9
    finally:
        # Cleanup
        remove_csv_files()


@pytest.mark.functions
def test_read_csvs_three_separated_csv_path():
    # Setup
    # When a CSV file with 3 cols and 3 rows is on disk
    create_csv_file(3)

    # If the csv file is read into DataFrame
    dfs = io.read_csvs(CSV_FILE_PATH.format("*"), seperate_df=True)

    # Then the dataframe list has 3 dataframes
    try:
        assert len(dfs) == 3
        for df in dfs.values():
            assert len(df) == 3
            assert len(df.columns) == 3
    finally:
        # Cleanup
        remove_csv_files()
