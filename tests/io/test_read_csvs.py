import pandas as pd
import pytest

from janitor import io


def create_csv_files(tmp_path, number_of_files):
    """Create test CSV files in the given temporary directory."""
    filepaths = []
    for i in range(number_of_files):
        filepath = tmp_path / f"test_csv_{i}.csv"
        df = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        df.to_csv(filepath, index=False)
        filepaths.append(filepath)
    return filepaths


@pytest.mark.functions
def test_read_csvs_one_csv_path(tmp_path):
    """Test reading a single CSV file."""
    create_csv_files(tmp_path, 1)

    df = io.read_csvs(str(tmp_path / "test_csv_*.csv"))

    assert len(df.columns) == 3
    assert len(df) == 4


@pytest.mark.functions
def test_read_csvs_zero_csv_path(tmp_path):
    """Test that reading non-existent files raises ValueError."""
    with pytest.raises(ValueError, match="No CSV files to read"):
        io.read_csvs(str(tmp_path / "nofilesondisk.csv"))


@pytest.mark.functions
def test_read_csvs_three_csv_path(tmp_path):
    """Test reading multiple CSV files concatenated."""
    number_of_files = 3
    create_csv_files(tmp_path, number_of_files)

    df = io.read_csvs(str(tmp_path / "test_csv_*.csv"))

    assert len(df.columns) == 3
    assert len(df) == 4 * number_of_files


@pytest.mark.functions
def test_read_csvs_three_separated_csv_path(tmp_path):
    """Test reading multiple CSV files as separate DataFrames."""
    number_of_files = 3
    create_csv_files(tmp_path, number_of_files)

    dfs_dict = io.read_csvs(str(tmp_path / "test_csv_*.csv"), separate_df=True)

    assert len(dfs_dict) == number_of_files
    for df in dfs_dict.values():
        assert len(df) == 4
        assert len(df.columns) == 3


@pytest.mark.functions
def test_read_csvs_two_unmatching_csv_files(tmp_path):
    """Test that mismatched column names raise ValueError."""
    df1 = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["a", "b", "c"])
    df1.to_csv(tmp_path / "test_csv_0.csv", index=False)

    df2 = pd.DataFrame([[1, 2, 3], [1, 2, 3], [1, 2, 3]], columns=["d", "e", "f"])
    df2.to_csv(tmp_path / "test_csv_1.csv", index=False)

    with pytest.raises(ValueError, match="Columns in input CSV files do not match"):
        io.read_csvs(str(tmp_path / "test_csv_*.csv"))


@pytest.mark.functions
def test_read_csvs_lists(tmp_path):
    """Test reading CSV files from a list of paths."""
    number_of_files = 3
    filepaths = create_csv_files(tmp_path, number_of_files)
    csvs_list = [str(fp) for fp in filepaths]

    dfs_dict = io.read_csvs(files_path=csvs_list, separate_df=True)

    assert len(dfs_dict) == number_of_files
    for df in dfs_dict.values():
        assert len(df) == 4
        assert len(df.columns) == 3
