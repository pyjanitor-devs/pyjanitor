import io
import tarfile
import zipfile

import pandas as pd
import pytest

from janitor.io import read_archive


@pytest.fixture
def zip_test_file(tmp_path):
    """Fixture pour créer un fichier ZIP de test."""
    zip_path = tmp_path / "test.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr("file1.csv", "col1,col2\n1,2\n3,4")
        zf.writestr("file2.csv", "col3,col4\n5,6\n7,8")
    return zip_path


@pytest.fixture
def tar_test_file(tmp_path):
    """Fixture pour créer un fichier TAR de test."""
    tar_path = tmp_path / "test.tar.gz"
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


def test_read_zip_archive(zip_test_file):
    result = read_archive(
        str(zip_test_file), extract_to_df=True, selected_files=["file1.csv"]
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["col1", "col2"]
    assert result.shape == (2, 2)


def test_list_files_in_zip(zip_test_file):
    result = read_archive(str(zip_test_file), extract_to_df=False)
    assert isinstance(result, list)
    assert "file1.csv" in result
    assert "file2.csv" in result


def test_no_compatible_files(tmp_path):
    zip_path = tmp_path / "empty.zip"
    with zipfile.ZipFile(zip_path, mode="w") as zf:
        zf.writestr("file1.txt", "Just some text")
    with pytest.raises(
        ValueError, match="No compatible files found in the archive"
    ):
        read_archive(str(zip_path))


def test_read_tar_archive(tar_test_file):
    result = read_archive(
        str(tar_test_file), extract_to_df=True, selected_files=["file1.csv"]
    )
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == ["col1", "col2"]
    assert result.shape == (2, 2)


def test_list_files_in_tar(tar_test_file):
    result = read_archive(str(tar_test_file), extract_to_df=False)
    assert isinstance(result, list)
    assert "file1.csv" in result
    assert "file2.csv" in result
