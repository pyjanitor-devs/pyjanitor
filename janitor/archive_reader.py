"""Read data files stored inside zip and tar archives."""

from __future__ import annotations

import os
import tarfile
import warnings
import zipfile
from dataclasses import dataclass
from io import BytesIO
from pathlib import PurePosixPath
from typing import IO, Any

import pandas as pd

from janitor.utils import check, find_stack_level, import_message

SUPPORTED_MEMBER_SUFFIXES = frozenset(
    {
        ".csv",
        ".tsv",
        ".json",
        ".jsonl",
        ".ndjson",
        ".xlsx",
        ".xlsm",
        ".parquet",
        ".pq",
    }
)
DEFAULT_MAX_MEMBER_BYTES = 1024**3
READ_BLOCK_BYTES = 1024 * 1024


@dataclass(frozen=True)
class ArchiveContents:
    """Members discovered while an archive is open.

    Attributes:
        names: Unique canonical member names in first-seen order.
        counts: How many times each canonical name occurs.
        payloads: Decompressed bytes for members that were loaded.
        skipped: Unsupported members skipped while reading every file.
    """

    names: list[str]
    counts: dict[str, int]
    payloads: dict[str, bytes]
    skipped: list[str]


def canonical_member_name(name: str) -> str:
    """Drop one leading ``./`` from an archive member name.

    ``tar czf archive.tgz .`` stores entries as ``./a.csv``. Callers can
    pass either that stored name or ``a.csv``.

    Args:
        name: Member name as stored, or as supplied by the caller.

    Returns:
        The name without one leading ``./``.
    """

    if name.startswith("./"):
        return name[2:]
    return name


def archive_member_suffix(name: str) -> str:
    """Return the lowercase extension of an archive member.

    Args:
        name: Canonical member name.

    Returns:
        The extension, including the dot, or an empty string.
    """

    return PurePosixPath(name).suffix.lower()


def archive_member_is_metadata(name: str) -> bool:
    """Return whether a member is macOS archive metadata.

    Args:
        name: Member name as stored in the archive.

    Returns:
        True when the name is under ``__MACOSX``.
    """

    return "__MACOSX" in PurePosixPath(name).parts


def infer_archive_kind(
    path: str | os.PathLike[str],
    archive_format: str | None,
) -> str:
    """Return ``zip`` or ``tar`` from an explicit value or the file name.

    Args:
        path: Path of the archive.
        archive_format: ``zip`` or ``tar`` when the name has no suffix.

    Returns:
        ``zip`` or ``tar``.

    Raises:
        ValueError: If the archive type cannot be determined.
    """

    if archive_format is not None:
        return archive_format
    name = os.fspath(path).lower()
    if name.endswith(".zip"):
        return "zip"
    tar_suffixes = (
        ".tar",
        ".tar.gz",
        ".tgz",
        ".tar.bz2",
        ".tbz2",
        ".tar.xz",
        ".txz",
    )
    if name.endswith(tar_suffixes):
        return "tar"
    raise ValueError(
        "Unsupported archive type. Expected a zip or tar archive "
        "(including gz, bz2, and xz). Pass archive_format='zip' or "
        "archive_format='tar' when the name has no recognized extension."
    )


def read_bounded_stream(
    stream: IO[bytes],
    declared_size: int,
    limit: int,
    name: str,
) -> bytes:
    """Read one member, stopping above the decompressed size limit.

    Args:
        stream: Open member body.
        declared_size: Uncompressed size from the archive header.
        limit: Maximum decompressed bytes to accept.
        name: Canonical member name, used in the error.

    Returns:
        The member body.

    Raises:
        ValueError: If the member is larger than `limit`.
    """

    if declared_size > limit:
        raise ValueError(
            f"Archive member {name!r} declares {declared_size} decompressed "
            f"bytes, above the limit of {limit} bytes."
        )
    chunks: list[bytes] = []
    total = 0
    while True:
        block = stream.read(READ_BLOCK_BYTES)
        if not block:
            break
        total += len(block)
        if total > limit:
            raise ValueError(
                f"Archive member {name!r} exceeds the decompressed size "
                f"limit of {limit} bytes."
            )
        chunks.append(block)
    return b"".join(chunks)


def warn_about_unsupported_members(names: list[str]) -> None:
    """Warn that automatic selection skipped unsupported members.

    Args:
        names: Canonical names that were not read.
    """

    if not names:
        return
    shown = names[:20]
    preview = ", ".join(repr(name) for name in shown)
    if len(names) > len(shown):
        preview = f"{preview}, and {len(names) - len(shown)} more"
    supported = ", ".join(sorted(SUPPORTED_MEMBER_SUFFIXES))
    warnings.warn(
        "Skipped unsupported archive members: "
        f"{preview}. Supported extensions are {supported}.",
        UserWarning,
        stacklevel=find_stack_level(),
    )


def choose_archive_members(candidates: list[str]) -> list[str]:
    """Prompt for member numbers and return the chosen names.

    Args:
        candidates: Supported member names, in archive order.

    Returns:
        The chosen names, in the order entered.

    Raises:
        ValueError: If the entry is empty or not a listed number.
    """

    print("Compatible files found in the archive:")
    for index, name in enumerate(candidates, start=1):
        print(f"{index}. {name}")
    raw = input(
        "Enter the numbers of the files to read, separated by commas (e.g., 1,2,3): "
    )
    chosen_names: list[str] = []
    for token in raw.split(","):
        text = token.strip()
        if not text.isdigit() or not 0 < int(text) <= len(candidates):
            raise ValueError("No valid files selected.")
        chosen_names.append(candidates[int(text) - 1])
    if not chosen_names or len(chosen_names) != len(set(chosen_names)):
        raise ValueError("No valid files selected.")
    return chosen_names


def member_should_be_loaded(
    name: str,
    wanted: set[str] | None,
) -> bool:
    """Return whether this member's bytes should be kept.

    Args:
        name: Canonical member name.
        wanted: Names the caller asked for, or None to take every
            supported member.

    Returns:
        True when the member is supported and selected.
    """

    if archive_member_suffix(name) not in SUPPORTED_MEMBER_SUFFIXES:
        return False
    return wanted is None or name in wanted


def record_archive_member(
    contents: ArchiveContents,
    raw_name: str,
    declared_size: int,
    stream: IO[bytes] | None,
    wanted: set[str] | None,
    limit: int,
) -> None:
    """Record one regular file and load it when it will be parsed.

    Args:
        contents: Accumulator for this archive. Mutated in place.
        raw_name: Name stored in the archive.
        declared_size: Uncompressed size from the archive header.
        stream: Open body when the member should be loaded.
        wanted: Names the caller asked for, or None for every
            supported member.
        limit: Maximum decompressed bytes for one member.

    Raises:
        ValueError: If a loaded member exceeds `limit`.
    """

    name = canonical_member_name(raw_name)
    contents.counts[name] = contents.counts.get(name, 0) + 1
    if contents.counts[name] == 1:
        contents.names.append(name)
        if (
            wanted is None
            and archive_member_suffix(name) not in SUPPORTED_MEMBER_SUFFIXES
        ):
            contents.skipped.append(name)
    if stream is not None:
        contents.payloads[name] = read_bounded_stream(
            stream, declared_size, limit, name
        )


def load_zip_archive(
    path: str | os.PathLike[str],
    wanted: set[str] | None,
    limit: int,
) -> ArchiveContents:
    """Read a zip archive once and load the members that will be parsed.

    Args:
        path: Path of the zip archive.
        wanted: Names the caller asked for, or None for every
            supported member.
        limit: Maximum decompressed bytes for one member.

    Returns:
        Names, counts, payloads, and skipped members.
    """

    contents = ArchiveContents(names=[], counts={}, payloads={}, skipped=[])
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or archive_member_is_metadata(info.filename):
                continue
            name = canonical_member_name(info.filename)
            stream = None
            if member_should_be_loaded(name, wanted):
                stream = archive.open(info)
            try:
                record_archive_member(
                    contents,
                    info.filename,
                    info.file_size,
                    stream,
                    wanted,
                    limit,
                )
            finally:
                if stream is not None:
                    stream.close()
    return contents


def load_tar_archive(
    path: str | os.PathLike[str],
    wanted: set[str] | None,
    limit: int,
) -> ArchiveContents:
    """Read a tar archive once and load the members that will be parsed.

    The archive is opened once. Member bodies that are not parsed are
    skipped by the tar reader as it advances, including gzip streams.

    Args:
        path: Path of the tar archive.
        wanted: Names the caller asked for, or None for every
            supported member.
        limit: Maximum decompressed bytes for one member.

    Returns:
        Names, counts, payloads, and skipped members.

    Raises:
        ValueError: If a member that should be loaded cannot be read.
    """

    contents = ArchiveContents(names=[], counts={}, payloads={}, skipped=[])
    with tarfile.open(path, mode="r:*") as archive:
        for member in archive:
            if not member.isfile() or archive_member_is_metadata(member.name):
                continue
            name = canonical_member_name(member.name)
            stream = None
            try:
                if member_should_be_loaded(name, wanted):
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise ValueError(f"File not found in the archive: {name!r}.")
                record_archive_member(
                    contents,
                    member.name,
                    member.size,
                    stream,
                    wanted,
                    limit,
                )
            finally:
                if stream is not None:
                    stream.close()
    return contents


def requested_member_names(
    filenames: str | list[str] | tuple[str, ...],
) -> list[str]:
    """Return canonical names for an explicit selection.

    Args:
        filenames: One member name, or a list or tuple of names.

    Returns:
        Canonical names in the requested order.

    Raises:
        ValueError: If the selection is empty or repeats a name.
    """

    raw_names = [filenames] if isinstance(filenames, str) else list(filenames)
    names = [canonical_member_name(name) for name in raw_names]
    if not names:
        raise ValueError("No files selected.")
    if len(names) != len(set(names)):
        raise ValueError("filenames contains duplicate names.")
    return names


def select_archive_members(
    contents: ArchiveContents,
    filenames: str | list[str] | tuple[str, ...] | None,
    interactive: bool,
) -> tuple[list[str], bool]:
    """Choose members and whether a single frame should be unwrapped.

    A list or tuple stays a dictionary. A string, one interactive choice,
    or a single supported member is unwrapped. A duplicated name raises
    only when that name is among the chosen members.

    Args:
        contents: Members found in the archive.
        filenames: Explicit selection, or None.
        interactive: Whether to prompt when `filenames` is omitted.

    Returns:
        Chosen canonical names, and True when the result is one frame.

    Raises:
        ValueError: If the archive is empty, a name is missing or
            unsupported, a chosen name is duplicated, or nothing
            supported is present.
    """

    if not contents.names:
        raise ValueError("The archive is empty.")
    if filenames is not None:
        requested = requested_member_names(filenames)
        missing = [name for name in requested if name not in contents.counts]
        if missing:
            rendered = ", ".join(repr(name) for name in missing)
            raise ValueError(f"File not found in the archive: {rendered}.")
        unsupported = [
            name
            for name in requested
            if archive_member_suffix(name) not in SUPPORTED_MEMBER_SUFFIXES
        ]
        if unsupported:
            rendered = ", ".join(repr(name) for name in unsupported)
            raise ValueError(f"Unsupported data file {rendered}.")
        chosen = requested
        unwrap = isinstance(filenames, str)
    else:
        chosen = [
            name
            for name in contents.names
            if archive_member_suffix(name) in SUPPORTED_MEMBER_SUFFIXES
        ]
        if not chosen:
            raise ValueError("No supported data files found in the archive.")
        if interactive:
            chosen = choose_archive_members(chosen)
        unwrap = len(chosen) == 1
    duplicated = [name for name in chosen if contents.counts[name] > 1]
    if duplicated:
        rendered = ", ".join(repr(name) for name in duplicated)
        raise ValueError(f"Duplicate archive members selected: {rendered}.")
    return chosen, unwrap


def read_member_frame(
    payload: bytes,
    suffix: str,
    engine: str,
    reader_kwargs: dict[str, Any],
    polars_module: Any,
) -> Any:
    """Parse one member with the selected dataframe engine.

    Args:
        payload: Decompressed member bytes.
        suffix: Lowercase extension, including the dot.
        engine: ``pandas`` or ``polars``.
        reader_kwargs: Keyword arguments for the reader. Not mutated.
        polars_module: Imported polars module when `engine` is polars.

    Returns:
        A pandas or polars DataFrame.
    """

    buffer = BytesIO(payload)
    options = dict(reader_kwargs)
    if engine == "pandas":
        if suffix == ".tsv":
            options.setdefault("sep", "\t")
            return pd.read_csv(buffer, **options)
        if suffix in {".jsonl", ".ndjson"}:
            options.setdefault("lines", True)
            return pd.read_json(buffer, **options)
        if suffix in {".xlsx", ".xlsm"}:
            return pd.read_excel(buffer, **options)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(buffer, **options)
        if suffix == ".json":
            return pd.read_json(buffer, **options)
        return pd.read_csv(buffer, **options)
    if suffix == ".tsv":
        options.setdefault("separator", "\t")
        return polars_module.read_csv(buffer, **options)
    if suffix in {".jsonl", ".ndjson"}:
        return polars_module.read_ndjson(buffer, **options)
    if suffix in {".xlsx", ".xlsm"}:
        return polars_module.read_excel(buffer, **options)
    if suffix in {".parquet", ".pq"}:
        return polars_module.read_parquet(buffer, **options)
    if suffix == ".json":
        return polars_module.read_json(buffer, **options)
    return polars_module.read_csv(buffer, **options)


def read_archive(
    path: str | os.PathLike[str],
    filenames: str | list[str] | tuple[str, ...] | None = None,
    engine: str = "pandas",
    archive_format: str | None = None,
    interactive: bool = False,
    max_member_bytes: int = DEFAULT_MAX_MEMBER_BYTES,
    **kwargs: Any,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Read data files stored inside a zip or tar archive.

    Supported member extensions are `.csv`, `.tsv`, `.json`, `.jsonl`,
    `.ndjson`, `.xlsx`, `.xlsm`, `.parquet`, and `.pq`. When `filenames`
    is omitted and at least one supported member is read, other regular
    files are skipped and a `UserWarning` lists them. Requesting one of
    those names raises `ValueError`. An archive with no supported members
    raises `ValueError` as well. Directory entries and macOS `__MACOSX`
    metadata are ignored. Nested archives are not opened; pass the
    archive-relative name, such as `nested/sales.csv`, to read a file in
    a subdirectory. A single leading `./` is removed from member names,
    so an archive made with `tar czf archive.tgz .` matches `a.csv` as
    well as `./a.csv`. Dictionary keys omit that prefix.

    A duplicated name raises `ValueError` only when that name is selected.
    Duplicates of members that are not selected are ignored. The archive
    is opened once. Each selected member is refused when its decompressed
    size is above `max_member_bytes`.

    A single selected file is returned as one DataFrame. Several files are
    returned as a dictionary keyed by member name, in archive order when
    every supported file is read and in the requested order when `filenames`
    is a list or tuple. A list or tuple always returns a dictionary, even
    when it contains one name. Tab-separated files use a tab separator
    unless one is already present in `kwargs` (`sep` for pandas, `separator`
    for polars).

    `engine` chooses pandas or polars and is not forwarded to the reader.
    Pandas `read_csv`, pandas `read_excel`, and polars `read_excel` also
    take an `engine` argument; those values cannot be set through this
    function. Polars Excel reads use polars' default spreadsheet engine
    and require the fastexcel package.
    Keyword arguments are forwarded to every reader, so select members that
    share a reader when passing format-specific options.

    Examples:
        >>> import tempfile
        >>> import zipfile
        >>> from pathlib import Path
        >>> import pandas as pd
        >>> from janitor import read_archive
        >>> folder = Path(tempfile.mkdtemp())
        >>> sales = folder / "sales.csv"
        >>> pd.DataFrame({"id": [1, 2], "city": ["Lisbon", "Porto"]}).to_csv(
        ...     sales, index=False
        ... )
        >>> archive_path = folder / "sales.zip"
        >>> with zipfile.ZipFile(archive_path, mode="w") as archive:
        ...     archive.write(sales, arcname="sales.csv")
        ...     archive.writestr("notes.txt", "ignore me")

        One supported file is returned as a DataFrame. `notes.txt` is
        skipped and a warning is emitted:

        >>> read_archive(archive_path)
           id    city
        0   1  Lisbon
        1   2   Porto

        Keyword arguments are passed through to the underlying reader:

        >>> read_archive(archive_path, usecols=["city"])
             city
        0  Lisbon
        1   Porto

        Several data files are returned as a dictionary of DataFrames:

        >>> pd.DataFrame({"id": [1], "city": ["Lisbon"]}).to_csv(
        ...     folder / "east.csv", index=False
        ... )
        >>> pd.DataFrame({"id": [3], "city": ["London"]}).to_csv(
        ...     folder / "west.csv", index=False
        ... )
        >>> with zipfile.ZipFile(folder / "regions.zip", mode="w") as archive:
        ...     archive.write(folder / "east.csv", arcname="east.csv")
        ...     archive.write(folder / "west.csv", arcname="west.csv")
        >>> frames = read_archive(folder / "regions.zip")
        >>> sorted(frames)
        ['east.csv', 'west.csv']
        >>> frames["west.csv"]
           id    city
        0   3  London

        `engine="polars"` returns polars objects:

        >>> frame = read_archive(archive_path, engine="polars")
        >>> frame.columns
        ['id', 'city']
        >>> frame.get_column("id").to_list()
        [1, 2]

    Args:
        path: Path of a zip or tar archive. Gzip, bzip2, and xz compressed
            tar archives are detected from the name (`.tar.gz`, `.tgz`,
            `.tar.bz2`, `.tbz2`, `.tar.xz`, `.txz`) or by reading the file.
        filenames: Member to read, or a list or tuple of members. A string
            returns one DataFrame. A list or tuple returns a dictionary.
            `None` reads every supported data file and warns about the
            rest. A leading `./` is ignored.
        engine: DataFrame library used to parse each member. Either
            `"pandas"` or `"polars"`. Defaults to `"pandas"`.
        archive_format: `"zip"` or `"tar"` when the file name does not use
            a recognized extension. Tar compression is detected from the
            file itself.
        interactive: When `True` and `filenames` is omitted, prompt for the
            members to read. The prompt lists supported files and expects
            comma-separated numbers such as `1,2`.
        max_member_bytes: Largest decompressed member that will be read.
            Defaults to 1 GiB.
        **kwargs: Keyword arguments forwarded to the underlying reader
            (`pandas.read_csv`, `polars.read_csv`, and the matching reader
            for each other extension).

    Raises:
        ValueError: If the archive type, `engine`, or `archive_format` is
            unsupported, the archive is empty, no supported data file is
            present, a requested member is missing or has an unsupported
            extension, a selected member name is duplicated, a selected
            member exceeds `max_member_bytes`, or interactive selection is
            empty or combined with `filenames`.
        ImportError: If `engine` is `"polars"` and polars is not installed.

    Returns:
        A pandas or polars DataFrame when one member is read, or a
        dictionary of DataFrames keyed by member name.
    """

    check("path", path, [str, os.PathLike])
    check("engine", engine, [str])
    check("interactive", interactive, [bool])
    check("max_member_bytes", max_member_bytes, [int])
    if isinstance(max_member_bytes, bool) or max_member_bytes < 1:
        raise ValueError("max_member_bytes must be at least 1.")
    if engine not in {"pandas", "polars"}:
        raise ValueError("engine should be either pandas or polars.")
    if filenames is not None:
        check("filenames", filenames, [str, list, tuple])
        if isinstance(filenames, (list, tuple)):
            for index, entry in enumerate(filenames):
                check(f"filenames[{index}]", entry, [str])
    if archive_format is not None:
        check("archive_format", archive_format, [str])
        archive_format = archive_format.lower()
        if archive_format not in {"zip", "tar"}:
            raise ValueError("archive_format should be either 'zip' or 'tar'.")
    if interactive and filenames is not None:
        raise ValueError("Pass either filenames or interactive=True, not both.")

    polars_module = None
    if engine == "polars":
        try:
            import polars as polars_module
        except ImportError as error:
            import_message(
                submodule="polars",
                package="polars",
                conda_channel="conda-forge",
                pip_install=True,
            )
            raise ImportError("polars is required when engine='polars'.") from error

    kind = infer_archive_kind(path, archive_format)
    wanted = None
    if filenames is not None:
        wanted = set(requested_member_names(filenames))
    if kind == "zip":
        contents = load_zip_archive(path, wanted, max_member_bytes)
    else:
        contents = load_tar_archive(path, wanted, max_member_bytes)
    chosen, unwrap = select_archive_members(contents, filenames, interactive)
    if filenames is None and contents.skipped:
        warn_about_unsupported_members(contents.skipped)
    frames = {
        name: read_member_frame(
            contents.payloads[name],
            archive_member_suffix(name),
            engine,
            kwargs,
            polars_module,
        )
        for name in chosen
    }
    if unwrap:
        return next(iter(frames.values()))
    return frames
