import io
import os
import tarfile
import zipfile

import pandas as pd
import pytest

from janitor.functions.read_archive import read_archive


# Helper function to create ZIP archives
def create_test_zip(archive_path, files):
    with zipfile.ZipFile(archive_path, "w") as archive:
        for file_name, content in files.items():
            archive.writestr(file_name, content)


# Helper function to create TAR archives
def create_test_tar(archive_path, files):
    with tarfile.open(archive_path, "w:gz") as archive:
        for file_name, content in files.items():
            data = content.encode("utf-8")
            tarinfo = tarfile.TarInfo(name=file_name)
            tarinfo.size = len(data)
            archive.addfile(tarinfo, io.BytesIO(data))


# Fixture for creating a test ZIP archive
@pytest.fixture
def test_zip(tmp_path):
    archive_path = tmp_path / "test.zip"
    files = {"file1.csv": "col1,col2\n1,2\n3,4"}
    create_test_zip(archive_path, files)
    return str(archive_path)


# Fixture for creating a test TAR.GZ archive
@pytest.fixture
def test_tar(tmp_path):
    archive_path = tmp_path / "test.tar.gz"
    files = {"file1.csv": "col1,col2\n1,2\n3,4"}
    create_test_tar(archive_path, files)
    return str(archive_path)


# Test reading a ZIP archive and extracting content to a DataFrame
def test_read_zip_archive(test_zip):
    result = read_archive(test_zip, extract_to_df=True)
    expected = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
    pd.testing.assert_frame_equal(result, expected)


# Test reading a TAR.GZ archive and extracting content to a DataFrame
def test_read_tar_archive(test_tar):
    result = read_archive(test_tar, extract_to_df=True)
    expected = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
    pd.testing.assert_frame_equal(result, expected)


# Test with an unsupported file type
def test_read_archive_invalid_type():
    with pytest.raises(
        ValueError,
        match=(
            r"Cannot infer file type from the file extension\."
            r"Please specify the 'file_type' parameter\."
        ),
    ):
        read_archive("invalid_file.txt")


# Test with a ZIP archive containing no compatible files
def test_read_archive_no_csv(tmp_path):
    archive_path = tmp_path / "empty.zip"
    create_test_zip(archive_path, {"file1.txt": "No CSV here!"})
    assert os.path.exists(archive_path)  # Ensure the archive exists
    with pytest.raises(
        ValueError, match=r"No compatible files found in the archive\."
    ):
        read_archive(str(archive_path), extract_to_df=True)


# Test listing files in a ZIP archive without extracting
def test_read_archive_file_list(test_zip):
    result = read_archive(test_zip, extract_to_df=False)
    assert isinstance(result, list)
    assert "file1.csv" in result
    assert len(result) == 1  # Ensure only compatible files are listed
