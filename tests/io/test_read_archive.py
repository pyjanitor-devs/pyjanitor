"""Tests for reading data files stored inside zip and tar archives."""

import io
import tarfile
import warnings
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.io import read_archive

CSV_A = b"col1,col2\n1,2\n3,4\n"
CSV_B = b"col1,col2\n5,6\n7,8\n"
FRAME_A = pd.DataFrame({"col1": [1, 3], "col2": [2, 4]})
FRAME_B = pd.DataFrame({"col1": [5, 7], "col2": [6, 8]})
SEMICOLON_CSV = b"col1;col2\n1;2\n3;4\n"
TSV = b"col1\tcol2\n1\t2\n3\t4\n"
JSON_RECORDS = b'[{"id": 1, "city": "Lisbon"}, {"id": 2, "city": "Porto"}]'
JSON_LINES = b'{"id": 1, "city": "Lisbon"}\n{"id": 2, "city": "Porto"}\n'
JSON_FRAME = pd.DataFrame({"id": [1, 2], "city": ["Lisbon", "Porto"]})


def write_zip(path: Path, members: dict[str, bytes]) -> Path:
    """Write a zip archive and return its path."""

    with zipfile.ZipFile(path, mode="w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return path


def write_tar(
    path: Path,
    members: dict[str, bytes],
    mode: str = "w",
) -> Path:
    """Write a tar archive and return its path."""

    with tarfile.open(path, mode) as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def excel_payload(frame: pd.DataFrame) -> bytes:
    """Serialize a DataFrame to xlsx bytes."""

    buffer = io.BytesIO()
    frame.to_excel(buffer, index=False)
    return buffer.getvalue()


@pytest.mark.functions
def test_read_single_csv_from_zip(tmp_path: Path):
    """A zip with one CSV returns that DataFrame."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    assert_frame_equal(read_archive(archive), FRAME_A)


@pytest.mark.functions
@pytest.mark.parametrize(
    ("suffix", "mode"),
    [
        (".tar", "w"),
        (".tar.gz", "w:gz"),
        (".tgz", "w:gz"),
        (".tar.bz2", "w:bz2"),
        (".tbz2", "w:bz2"),
        (".tar.xz", "w:xz"),
        (".txz", "w:xz"),
    ],
)
def test_read_single_csv_from_tar(tmp_path: Path, suffix: str, mode: str):
    """Tar archives, including compressed aliases, yield the member."""

    archive = write_tar(
        tmp_path / f"data{suffix}",
        {"file1.csv": CSV_A},
        mode=mode,
    )

    assert_frame_equal(read_archive(archive), FRAME_A)


@pytest.mark.functions
def test_multiple_files_keep_archive_order(tmp_path: Path):
    """Several supported members come back as a dict in archive order."""

    archive = write_zip(
        tmp_path / "data.zip",
        {"b.csv": CSV_B, "a.csv": CSV_A, "notes.txt": b"ignore me"},
    )

    result = read_archive(archive)

    assert list(result) == ["b.csv", "a.csv"]
    assert_frame_equal(result["b.csv"], FRAME_B)
    assert_frame_equal(result["a.csv"], FRAME_A)


@pytest.mark.functions
def test_filename_string_selects_one_frame(tmp_path: Path):
    """A string filename is positional and returns one DataFrame."""

    archive = write_zip(
        tmp_path / "data.zip",
        {"file1.csv": CSV_A, "file2.csv": CSV_B},
    )

    assert_frame_equal(read_archive(archive, "file2.csv"), FRAME_B)


@pytest.mark.functions
def test_filename_list_returns_dict_in_request_order(tmp_path: Path):
    """A list of names stays a dict, including a list of one name."""

    archive = write_zip(
        tmp_path / "data.zip",
        {"file1.csv": CSV_A, "file2.csv": CSV_B},
    )

    several = read_archive(archive, filenames=["file2.csv", "file1.csv"])
    one = read_archive(archive, filenames=["file1.csv"])
    as_tuple = read_archive(archive, filenames=("file2.csv", "file1.csv"))

    assert list(several) == ["file2.csv", "file1.csv"]
    assert_frame_equal(several["file1.csv"], FRAME_A)
    assert list(one) == ["file1.csv"]
    assert_frame_equal(one["file1.csv"], FRAME_A)
    assert list(as_tuple) == ["file2.csv", "file1.csv"]


@pytest.mark.functions
def test_nested_member_path(tmp_path: Path):
    """Members in subdirectories are read by their archive-relative name."""

    archive = write_zip(
        tmp_path / "data.zip",
        {"nested/file1.csv": CSV_A, "nested/notes.txt": b"skip"},
    )

    assert_frame_equal(read_archive(archive, "nested/file1.csv"), FRAME_A)
    assert_frame_equal(read_archive(archive), FRAME_A)


@pytest.mark.functions
def test_macos_metadata_is_ignored(tmp_path: Path):
    """AppleDouble entries under __MACOSX are not treated as data files."""

    archive = write_zip(
        tmp_path / "data.zip",
        {
            "sales.csv": CSV_A,
            "__MACOSX/._sales.csv": b"\x00\x00not a csv",
        },
    )

    assert_frame_equal(read_archive(archive), FRAME_A)


@pytest.mark.functions
def test_uppercase_extension(tmp_path: Path):
    """Extensions are matched without case sensitivity."""

    archive = write_zip(tmp_path / "data.zip", {"FILE1.CSV": CSV_A})

    assert_frame_equal(read_archive(archive), FRAME_A)


@pytest.mark.functions
def test_string_path(tmp_path: Path):
    """String paths are accepted as well as path objects."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    assert_frame_equal(read_archive(str(archive)), FRAME_A)


@pytest.mark.functions
def test_kwargs_pass_through_and_are_not_mutated(tmp_path: Path):
    """Reader keyword arguments apply to every selected file."""

    archive = write_zip(
        tmp_path / "data.zip",
        {"a.csv": SEMICOLON_CSV, "b.csv": SEMICOLON_CSV},
    )
    options = {"sep": ";", "usecols": ["col1"]}

    result = read_archive(archive, **options)

    assert options == {"sep": ";", "usecols": ["col1"]}
    assert list(result) == ["a.csv", "b.csv"]
    assert list(result["a.csv"].columns) == ["col1"]
    assert result["a.csv"]["col1"].tolist() == [1, 3]
    assert result["b.csv"]["col1"].tolist() == [1, 3]


@pytest.mark.functions
def test_read_tsv(tmp_path: Path):
    """Tab-separated members are read without an explicit separator."""

    archive = write_zip(tmp_path / "data.zip", {"file1.tsv": TSV})

    assert_frame_equal(read_archive(archive), FRAME_A)


@pytest.mark.functions
def test_read_json_and_json_lines(tmp_path: Path):
    """JSON arrays and JSON lines are both supported."""

    records = write_zip(tmp_path / "records.zip", {"cities.json": JSON_RECORDS})
    lines = write_zip(tmp_path / "lines.zip", {"cities.jsonl": JSON_LINES})
    ndjson = write_zip(tmp_path / "ndjson.zip", {"cities.ndjson": JSON_LINES})

    assert_frame_equal(read_archive(records), JSON_FRAME)
    assert_frame_equal(read_archive(lines), JSON_FRAME)
    assert_frame_equal(read_archive(ndjson), JSON_FRAME)


@pytest.mark.functions
def test_mixed_csv_and_json(tmp_path: Path):
    """Each member is parsed with the reader for its own extension."""

    archive = write_zip(
        tmp_path / "data.zip",
        {"rows.csv": CSV_A, "cities.json": JSON_RECORDS},
    )

    result = read_archive(archive)

    assert list(result) == ["rows.csv", "cities.json"]
    assert_frame_equal(result["rows.csv"], FRAME_A)
    assert_frame_equal(result["cities.json"], JSON_FRAME)


@pytest.mark.functions
def test_read_excel(tmp_path: Path):
    """Excel workbooks stored in an archive are read back."""

    pytest.importorskip("openpyxl")
    frame = pd.DataFrame({"id": [1, 2], "city": ["Lisbon", "Porto"]})
    archive = write_zip(
        tmp_path / "data.zip",
        {"cities.xlsx": excel_payload(frame)},
    )

    assert_frame_equal(read_archive(archive), frame)


@pytest.mark.functions
def test_read_parquet_pandas(tmp_path: Path):
    """Parquet members are read with pandas when a parquet engine exists."""

    pytest.importorskip("pyarrow")
    frame = pd.DataFrame({"id": [1, 2], "city": ["Lisbon", "Porto"]})
    buffer = io.BytesIO()
    frame.to_parquet(buffer, index=False)
    archive = write_zip(
        tmp_path / "data.zip",
        {"cities.parquet": buffer.getvalue()},
    )

    assert_frame_equal(read_archive(archive), frame)


@pytest.mark.functions
def test_empty_zip(tmp_path: Path):
    """An archive with no members raises a clear error."""

    archive = write_zip(tmp_path / "empty.zip", {})

    with pytest.raises(ValueError, match="The archive is empty."):
        read_archive(archive)


@pytest.mark.functions
def test_empty_tar(tmp_path: Path):
    """An empty tar archive raises the same empty-archive error."""

    archive = write_tar(tmp_path / "empty.tar.gz", {}, mode="w:gz")

    with pytest.raises(ValueError, match="The archive is empty."):
        read_archive(archive)


@pytest.mark.functions
def test_directory_only_archive_is_empty(tmp_path: Path):
    """Directory entries do not count as files."""

    zip_path = tmp_path / "dirs.zip"
    with zipfile.ZipFile(zip_path, mode="w") as archive:
        archive.writestr("nested/", b"")
    tar_path = tmp_path / "dirs.tar"
    with tarfile.open(tar_path, mode="w") as archive:
        info = tarfile.TarInfo(name="nested")
        info.type = tarfile.DIRTYPE
        archive.addfile(info)

    with pytest.raises(ValueError, match="The archive is empty."):
        read_archive(zip_path)
    with pytest.raises(ValueError, match="The archive is empty."):
        read_archive(tar_path)


@pytest.mark.functions
def test_archive_with_no_supported_files(tmp_path: Path):
    """Members that are not data files are reported separately from empty."""

    archive = write_zip(tmp_path / "notes.zip", {"notes.txt": b"hello"})

    with pytest.raises(ValueError, match="No supported data files"):
        read_archive(archive)


@pytest.mark.functions
def test_unsupported_archive_type(tmp_path: Path):
    """Unrecognized archive extensions fail before the file is opened."""

    path = tmp_path / "notes.txt"
    path.write_text("hello", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported archive type"):
        read_archive(path)
    with pytest.raises(ValueError, match="Unsupported archive type"):
        read_archive("archive.rar")


@pytest.mark.functions
def test_missing_member(tmp_path: Path):
    """Requesting a member that is not in the archive is an error."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    with pytest.raises(ValueError, match="File not found in the archive"):
        read_archive(archive, filenames="missing.csv")


@pytest.mark.functions
def test_unsupported_member_requested_explicitly(tmp_path: Path):
    """An explicit request for a non-data member explains the extension."""

    archive = write_zip(
        tmp_path / "data.zip",
        {"file1.csv": CSV_A, "notes.txt": b"hello"},
    )

    with pytest.raises(ValueError, match="Unsupported data file"):
        read_archive(archive, filenames="notes.txt")


@pytest.mark.functions
def test_missing_archive_path(tmp_path: Path):
    """A missing zip path raises FileNotFoundError from the archive reader."""

    with pytest.raises(FileNotFoundError):
        read_archive(tmp_path / "missing.zip")


@pytest.mark.functions
def test_missing_polars_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A missing polars install prints guidance and raises ImportError."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})
    real_import = __import__

    def hide_polars(name: str, *args, **kwargs):
        if name == "polars" or name.startswith("polars."):
            raise ImportError("polars is not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr("builtins.__import__", hide_polars)

    with pytest.raises(ImportError, match="polars is required"):
        read_archive(archive, engine="polars")


@pytest.mark.functions
def test_invalid_engine(tmp_path: Path):
    """The engine argument only accepts pandas and polars."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    with pytest.raises(ValueError, match="either pandas or polars"):
        read_archive(archive, engine="spark")


@pytest.mark.functions
def test_archive_format_override(tmp_path: Path):
    """archive_format reads archives whose names have no suffix."""

    zip_path = write_zip(tmp_path / "data.bin", {"file1.csv": CSV_A})
    tar_path = write_tar(tmp_path / "data.bin.tar-contents", {"file1.csv": CSV_A})
    renamed_tar = tmp_path / "plain"
    renamed_tar.write_bytes(tar_path.read_bytes())

    assert_frame_equal(read_archive(zip_path, archive_format="ZIP"), FRAME_A)
    assert_frame_equal(read_archive(renamed_tar, archive_format="tar"), FRAME_A)


@pytest.mark.functions
def test_invalid_archive_format(tmp_path: Path):
    """Compression is not a separate archive_format value."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    with pytest.raises(ValueError, match="either 'zip' or 'tar'"):
        read_archive(archive, archive_format="tar.gz")


@pytest.mark.functions
def test_duplicate_member_names(tmp_path: Path):
    """Duplicate member names are rejected instead of silently collapsing."""

    archive = tmp_path / "dup.zip"
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Duplicate name: 'file1.csv'",
            category=UserWarning,
        )
        with zipfile.ZipFile(archive, mode="w") as zip_archive:
            zip_archive.writestr("file1.csv", CSV_A)
            zip_archive.writestr("file1.csv", CSV_B)

    with pytest.raises(ValueError, match="duplicate member names"):
        read_archive(archive)


@pytest.mark.functions
def test_duplicate_filename_request(tmp_path: Path):
    """The same member cannot be requested twice in one call."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    with pytest.raises(ValueError, match="duplicate names"):
        read_archive(archive, filenames=["file1.csv", "file1.csv"])


@pytest.mark.functions
def test_empty_filename_list(tmp_path: Path):
    """An empty selection is an error."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    with pytest.raises(ValueError, match="No files selected."):
        read_archive(archive, filenames=[])


@pytest.mark.functions
def test_interactive_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Interactive mode reads the members chosen by number."""

    archive = write_zip(
        tmp_path / "data.zip",
        {"file1.csv": CSV_A, "file2.csv": CSV_B},
    )
    monkeypatch.setattr("builtins.input", lambda prompt: "2, 1")

    result = read_archive(archive, interactive=True)

    assert list(result) == ["file2.csv", "file1.csv"]
    assert_frame_equal(result["file2.csv"], FRAME_B)


@pytest.mark.functions
def test_interactive_invalid_selection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    """Out-of-range interactive input does not read any member."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})
    monkeypatch.setattr("builtins.input", lambda prompt: "9")

    with pytest.raises(ValueError, match="No valid files selected."):
        read_archive(archive, interactive=True)


@pytest.mark.functions
def test_interactive_and_filenames_conflict(tmp_path: Path):
    """Filename selection and the interactive prompt are separate modes."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    with pytest.raises(ValueError, match="not both"):
        read_archive(archive, filenames="file1.csv", interactive=True)


@pytest.mark.functions
def test_read_does_not_print(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    """Successful reads stay quiet unless interactive mode is requested."""

    archive = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})

    read_archive(archive)
    captured = capsys.readouterr()

    assert captured.out == ""


@pytest.mark.functions
def test_exported_from_package():
    """read_archive is part of the top-level janitor package."""

    import janitor

    assert janitor.read_archive is read_archive


@pytest.mark.polars
@pytest.mark.functions
def test_polars_engine_csv_from_zip_and_tar_gz(tmp_path: Path):
    """The polars engine reads CSV members from zip and gzip tar archives."""

    pl = pytest.importorskip("polars")
    zip_path = write_zip(tmp_path / "data.zip", {"file1.csv": CSV_A})
    tar_path = write_tar(
        tmp_path / "data.tar.gz",
        {"file1.csv": CSV_A},
        mode="w:gz",
    )

    for archive in (zip_path, tar_path):
        result = read_archive(archive, engine="polars")
        assert isinstance(result, pl.DataFrame)
        assert result.columns == ["col1", "col2"]
        assert result.get_column("col1").to_list() == [1, 3]
        assert result.get_column("col2").to_list() == [2, 4]


@pytest.mark.polars
@pytest.mark.functions
def test_polars_kwargs_and_multiple_files(tmp_path: Path):
    """Polars receives reader arguments and returns one frame per member."""

    pl = pytest.importorskip("polars")
    archive = write_zip(
        tmp_path / "data.zip",
        {"b.csv": SEMICOLON_CSV, "a.csv": SEMICOLON_CSV},
    )

    result = read_archive(archive, engine="polars", separator=";")

    assert list(result) == ["b.csv", "a.csv"]
    assert isinstance(result["a.csv"], pl.DataFrame)
    assert result["a.csv"].columns == ["col1", "col2"]
    assert result["a.csv"].get_column("col1").to_list() == [1, 3]


@pytest.mark.polars
@pytest.mark.functions
def test_polars_json_tsv_and_parquet(tmp_path: Path):
    """Polars reads JSON, tab-separated text, and Parquet members."""

    pl = pytest.importorskip("polars")
    frame = pl.DataFrame({"id": [1, 2], "city": ["Lisbon", "Porto"]})
    parquet_buffer = io.BytesIO()
    frame.write_parquet(parquet_buffer)
    archive = write_zip(
        tmp_path / "data.zip",
        {
            "cities.json": JSON_RECORDS,
            "cities.jsonl": JSON_LINES,
            "cities.ndjson": JSON_LINES,
            "cities.tsv": b"id\tcity\n1\tLisbon\n2\tPorto\n",
            "cities.parquet": parquet_buffer.getvalue(),
            "cities.pq": parquet_buffer.getvalue(),
        },
    )

    result = read_archive(archive, engine="polars")

    assert list(result) == [
        "cities.json",
        "cities.jsonl",
        "cities.ndjson",
        "cities.tsv",
        "cities.parquet",
        "cities.pq",
    ]
    for name in result:
        assert result[name].equals(frame)


@pytest.mark.polars
@pytest.mark.functions
def test_polars_excel(tmp_path: Path):
    """Polars reads Excel members when its Excel reader is installed."""

    pl = pytest.importorskip("polars")
    pytest.importorskip("fastexcel")
    frame = pd.DataFrame({"id": [1, 2], "city": ["Lisbon", "Porto"]})
    archive = write_zip(
        tmp_path / "data.zip",
        {"cities.xlsx": excel_payload(frame)},
    )

    result = read_archive(archive, engine="polars")

    assert isinstance(result, pl.DataFrame)
    assert result.columns == ["id", "city"]
    assert result.get_column("id").to_list() == [1, 2]
    assert result.get_column("city").to_list() == ["Lisbon", "Porto"]
