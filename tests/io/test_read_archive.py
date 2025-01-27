import io
import tarfile
import zipfile
from unittest.mock import patch

import pandas as pd
import pytest

from janitor.io import (
    _infer_file_type,
    read_archive,
)


# Fixtures for creating test archives
@pytest.fixture
def dummy_zip_file(tmp_path):
    """Create a dummy ZIP file containing two CSV files."""
    zip_path = tmp_path / "dummy.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr("file1.csv", "col1,col2\n1,2\n3,4")
        zf.writestr("file2.csv", "col3,col4\n5,6\n7,8")
    return zip_path


@pytest.fixture
def dummy_tar_file(tmp_path):
    """Create a dummy TAR file containing two CSV files."""
    tar_path = tmp_path / "dummy.tar.gz"
    with tarfile.open(tar_path, mode="w:gz") as tf:
        info1 = tarfile.TarInfo(name="file1.csv")
        data1 = io.BytesIO(b"col1,col2\n1,2\n3,4")
        info1.size = data1.getbuffer().nbytes
        tf.addfile(info1, data1)

        info2 = tarfile.TarInfo(name="file2.csv")
        data2 = io.BytesIO(b"col3,col4\n5,6\n7,8")
        info2.size = data2.getbuffer().nbytes
        tf.addfile(info2, data2)
    return tar_path


# Tests for reading archives via `read_archive`
def test_read_zip_archive(dummy_zip_file):
    """Test reading a specific file from a ZIP archive."""
    result = read_archive(
        str(dummy_zip_file), extract_to_df=True, selected_files=["file1.csv"]
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["col1", "col2"]
    assert result.shape == (2, 2)


def test_list_files_in_zip(dummy_zip_file):
    """Test listing files in a ZIP archive."""
    result = read_archive(str(dummy_zip_file), extract_to_df=False)
    assert isinstance(result, list)
    assert "file1.csv" in result
    assert "file2.csv" in result


def test_no_compatible_files_in_zip(tmp_path):
    """Test handling a ZIP archive with no compatible files."""
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr("file1.txt", "Just some text")
    with pytest.raises(
        ValueError, match="No compatible files found in the archive"
    ):
        read_archive(str(zip_path))


def test_read_tar_archive(dummy_tar_file):
    """Test reading a specific file from a TAR archive."""
    result = read_archive(
        str(dummy_tar_file), extract_to_df=True, selected_files=["file1.csv"]
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["col1", "col2"]
    assert result.shape == (2, 2)


def test_list_files_in_tar(dummy_tar_file):
    """Test listing files in a TAR archive."""
    result = read_archive(str(dummy_tar_file), extract_to_df=False)
    assert isinstance(result, list)
    assert "file1.csv" in result
    assert "file2.csv" in result


def test_no_compatible_files_in_tar(tmp_path):
    """Test handling a TAR archive with no compatible files."""
    tar_path = tmp_path / "invalid.tar.gz"
    with tarfile.open(tar_path, mode="w:gz") as tf:
        info = tarfile.TarInfo(name="file1.txt")
        data = io.BytesIO(b"Just some text")
        info.size = data.getbuffer().nbytes
        tf.addfile(info, data)
    with pytest.raises(
        ValueError, match="No compatible files found in the archive"
    ):
        read_archive(str(tar_path))


# Tests for unsupported file types
def test_read_archive_unsupported_file():
    """Test handling unsupported file types."""
    with pytest.raises(
        ValueError,
        match="Cannot infer file type from the file extension. "
        "Please specify the 'file_type' parameter.",
    ):
        read_archive("test.unsupported")


def test_read_archive_no_extension():
    """Test handling files with no extension."""
    with pytest.raises(
        ValueError,
        match="Cannot infer file type from the file extension. "
        "Please specify the 'file_type' parameter.",
    ):
        read_archive("testfile")


# Tests for interactive file selection
def test_interactive_file_selection_valid(dummy_zip_file):
    """Test valid input for interactive file selection."""
    user_input = "1,2"
    with patch("builtins.input", return_value=user_input):
        result = read_archive(str(dummy_zip_file), extract_to_df=False)
        assert "file1.csv" in result
        assert "file2.csv" in result


def test_interactive_file_selection_invalid(dummy_zip_file):
    """Test invalid input for interactive file selection."""
    user_input = "4,abc"
    with patch("builtins.input", return_value=user_input):
        with pytest.raises(ValueError, match="No valid files selected"):
            read_archive(str(dummy_zip_file), extract_to_df=False)


# Tests for file type inference
def test_infer_file_type_valid():
    """Test valid file type inference."""
    assert _infer_file_type("test.zip") == "zip"
    assert _infer_file_type("test.tar") == "tar"
    assert _infer_file_type("test.tar.gz") == "tar.gz"


def test_infer_file_type_invalid():
    """Test invalid file type inference."""
    with pytest.raises(
        ValueError,
        match="Cannot infer file type from the file extension. "
        "Please specify the 'file_type' parameter.",
    ):
        _infer_file_type("testfile")
