from __future__ import annotations

import tarfile
import zipfile

import pandas as pd
import pandas_flavor as pf

from janitor.utils import check


@pf.register_dataframe_method
def read_archive(
    file_path: str,
    extract_to_df: bool = True,
    file_type: str | None = None,
) -> pd.DataFrame | list[str]:
    """
    Reads an archive file (.zip, .tar, .tar.gz) and optionally lists its content
    or extracts specific files into a DataFrame.

    Examples:
        >>> # Example usage
        >>> df = pd.read_archive("data.zip", extract_to_df=True)

    Args:
        file_path: The path to the archive file.
        extract_to_df: Whether to read the contents into a DataFrame
            (for CSV or similar formats). Default is True.
        file_type: Optional file type hint ('zip', 'tar', 'tar.gz').
            If None, it will be inferred from the file extension.

    Returns:
        - A pandas DataFrame if extract_to_df is True
          and the user selects a file to load.
        - A list of compatible file names in the archive otherwise.
    """

    check("file_path", file_path, [str])
    check("extract_to_df", extract_to_df, [bool])

    file_type = file_type or _infer_file_type(file_path)

    if file_type == "zip":
        return _process_zip_archive(file_path, extract_to_df)
    elif file_type in {"tar", "tar.gz"}:
        return _process_tar_archive(file_path, extract_to_df)
    else:
        raise ValueError(
            "Unsupported archive format.Supported formats are .zip, .tar, or .tar.gz."
        )


def _infer_file_type(file_path: str) -> str:
    """
    Infer the type of the archive based on the file extension.

    Args:
        file_path: Path to the file.

    Returns:
        A string representing the archive type ('zip', 'tar', 'tar.gz').

    Raises:
        ValueError if the file extension is unsupported.
    """
    if file_path.endswith(".zip"):
        return "zip"
    elif file_path.endswith((".tar", ".tar.gz")):
        return "tar.gz" if file_path.endswith(".tar.gz") else "tar"
    else:
        raise ValueError(
            "Cannot infer file type from the file extension. "
            "Please specify the 'file_type' parameter."
        )


def _process_zip_archive(
    file_path: str, extract_to_df: bool
) -> pd.DataFrame | list[str]:
    """
    Process a ZIP archive.

    Args:
        file_path: Path to the ZIP file.
        extract_to_df: Whether to extract the content into a DataFrame.

    Returns:
        A DataFrame or a list of files in the archive.
    """
    with zipfile.ZipFile(file_path) as archive:
        compatible_files = _list_compatible_files(archive.namelist())

        if extract_to_df:
            return _select_and_extract_from_zip(archive, compatible_files)
        return compatible_files


def _process_tar_archive(
    file_path: str, extract_to_df: bool
) -> pd.DataFrame | list[str]:
    """
    Process a TAR archive.

    Args:
        file_path: Path to the TAR file.
        extract_to_df: Whether to extract the content into a DataFrame.

    Returns:
        A DataFrame or a list of files in the archive.
    """
    mode = "r:gz" if file_path.endswith(".gz") else "r"
    with tarfile.open(file_path, mode) as archive:
        compatible_files = _list_compatible_files(archive.getnames())

        if extract_to_df:
            return _select_and_extract_from_tar(archive, compatible_files)
        return compatible_files


def _list_compatible_files(file_names: list[str]) -> list[str]:
    """
    Helper function to list compatible files (e.g., .csv, .xlsx) from an archive.

    Args:
        file_names: List of file names in the archive.

    Returns:
        List of compatible file names.
    """
    compatible_files = [
        file_name
        for file_name in file_names
        if file_name.endswith((".csv", ".xlsx"))
    ]
    print("Fichiers compatibles détectés :", compatible_files)
    if not compatible_files:
        raise ValueError("No compatible files found in the archive.")
    return compatible_files


def _select_and_extract_from_zip(
    archive: zipfile.ZipFile, compatible_files: list[str]
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Helper function to allow the user to select
    and read specific files from a ZIP archive.

    Args:
        archive: The ZIP archive object.
        compatible_files: List of compatible file names.

    Returns:
        A single DataFrame or a list of DataFrames.
    """
    selected_files = _select_files_interactively(compatible_files)
    dfs = []
    for selected_file in selected_files:
        with archive.open(selected_file) as file:
            if selected_file.endswith(".csv"):
                dfs.append(pd.read_csv(file))
            elif selected_file.endswith(".xlsx"):
                dfs.append(pd.read_excel(file))
    return dfs if len(dfs) > 1 else dfs[0]


def _select_and_extract_from_tar(
    archive: tarfile.TarFile, compatible_files: list[str]
) -> pd.DataFrame | list[pd.DataFrame]:
    """
    Helper function to allow the user to select
    and read specific files from a TAR archive.

    Args:
        archive: The TAR archive object.
        compatible_files: List of compatible file names.

    Returns:
        A single DataFrame or a list of DataFrames.
    """
    selected_files = _select_files_interactively(compatible_files)
    dfs = []
    for selected_file in selected_files:
        member = archive.getmember(selected_file)
        with archive.extractfile(member) as file:
            if selected_file.endswith(".csv"):
                dfs.append(pd.read_csv(file))
            elif selected_file.endswith(".xlsx"):
                dfs.append(pd.read_excel(file))
    return dfs if len(dfs) > 1 else dfs[0]


def _select_files_interactively(compatible_files: list[str]) -> list[str]:
    """
    Allow the user to select files from a list interactively.

    Args:
        compatible_files: List of compatible file names.

    Returns:
        List of selected file names.
    """
    print("Compatible files found in the archive:")
    for idx, file_name in enumerate(compatible_files, 1):
        print(f"{idx}. {file_name}")

    selected_indices = (
        input(
            "Enter the numbers of the files to read, "
            "separated by commas (e.g., 1,2,3): "
        )
        .strip()
        .split(",")
    )
    selected_files = [
        compatible_files[int(idx) - 1]
        for idx in selected_indices
        if idx.strip().isdigit() and 0 < int(idx) <= len(compatible_files)
    ]
    if not selected_files:
        raise ValueError("No valid files selected.")
    return selected_files
