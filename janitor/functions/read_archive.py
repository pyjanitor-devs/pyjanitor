import zipfile
import tarfile
import pandas as pd


def read_archive(file_path: str, extract_to_df: bool = True, file_type: str = None) -> pd.DataFrame | list[str]:
    """
    Reads an archive  file (.zip, .tar, .tar.gz) and optionally lists its content or extracts specific files into a DataFrame.

    Args:
        file_path: The path to the archive file.
        extract_to_df: Whether to attempt reading the contents into a DataFrame (for CSV or similar formats). Default is True.
        file_type: Optional file type hint. Can be 'zip', 'tar' or 'tar.gz'. If None, it will be inferred from the file extension.

    Returns:
        - A pandas DataFrame if extract_to_df is True and the user selects a file to load.
        - A list of compatible file names in the archive otherwise.

    Raises:
        ValueError: If the file format is unsupported or if no readable files are found in the archive.
    """

    # Detect file type if not provided
    if not file_type:
        if file_path.endswith('.zip'):
            file_type = 'zip'
        elif file_path.endswith(('.tar', '.tar.gz', '.tgz')):
            file_type = 'tar'
        else:
            raise ValueError("Unsupported archive format. Please provide a valid .zip, .tar or .tar.gz file.")

    # Process ZIP files
    if file_type == 'zip':
        with zipfile.ZipFile(file_path, 'r') as archive:
            file_names = archive.namelist()
            compatible_files = _list_compatible_files(file_names)
            if extract_to_df:
                return _select_and_extract_from_zip(archive, compatible_files)
            return compatible_files

    # Process TAR files (including .tar.gz)
    elif file_type == 'tar':
        mode = 'r:gz' if file_path.endswith('.gz') else 'r'
        with tarfile.open(file_path, mode) as archive:
            file_names = archive.getnames()
            compatible_files = _list_compatible_files(file_names)
            if extract_to_df:
                return _select_and_extract_from_tar(archive, compatible_files)
            return compatible_files


def _list_compatible_files(file_names: list[str]) -> list[str]:
    """Helper function to list compatible files (e.g., .csv, .xlsx) from an archive."""
    compatible_files = [file_name for file_name in file_names if file_name.endswith(('.csv', '.xlsx'))]
    print("Fichiers compatibles détectés :", compatible_files)
    if not compatible_files:
        raise ValueError("No compatible files found in the archive.")
    return compatible_files


def _select_and_extract_from_zip(archive: zipfile.ZipFile, compatible_files: list[str]) -> pd.DataFrame | list[pd.DataFrame]:
    """Helper function to allow the user to select and read specific files from a ZIP archive."""
    if not compatible_files:
        raise ValueError("No compatible files found in the archive.")

    print("Compatible files found in the archive:")
    for i, file_name in enumerate(compatible_files):
        print(f"{i + 1}. {file_name}")

    selected_files = input("Enter the numbers of the files you want to read, separated by commas (e.g., 1,3): ").strip()
    if not selected_files:
        raise ValueError("No files selected.")

    selected_indices = []
    for index in selected_files.split(','):
        index = index.strip()
        if index.isdigit():
            index = int(index) - 1
            if 0 <= index < len(compatible_files):
                selected_indices.append(index)
            else:
                print(f"Index out of range : {index + 1}")
        else:
            print(f"Invalid Index : '{index}'")

    if not selected_indices:
        raise ValueError("No valid indices selected.")

    dfs = []
    for index in selected_indices:
        file_name = compatible_files[index]
        try:
            with archive.open(file_name) as file:
                if file_name.endswith('.csv'):
                    dfs.append(pd.read_csv(file))
                elif file_name.endswith('.xlsx'):
                    dfs.append(pd.read_excel(file))
        except Exception as e:
            print(f"Error reading the file {file_name}: {e}")

    if not dfs:
        raise ValueError("No files could be read successfully.")

    return dfs if len(dfs) > 1 else dfs[0]


def _select_and_extract_from_tar(archive: tarfile.TarFile, compatible_files: list[str]) -> pd.DataFrame | list[pd.DataFrame]:
    """Helper function to allow the user to select and read specific files from a TAR archive."""
    if not compatible_files:
        raise ValueError("No compatible files found in the archive.")

    print("Compatible files found in the archive:")
    for i, file_name in enumerate(compatible_files):
        print(f"{i + 1}. {file_name}")

    selected_files = input("Enter the numbers of the files you want to read, separated by commas (e.g., 1,3): ").strip()
    if not selected_files:
        raise ValueError("No files selected.")

    selected_indices = [int(index.strip()) - 1 for index in selected_files.split(',') if index.strip().isdigit()]
    dfs = []
    for index in selected_indices:
        member = archive.getmember(compatible_files[index])
        extracted_file = archive.extractfile(member)
        if extracted_file:
            try:
                if member.name.endswith('.csv'):
                    dfs.append(pd.read_csv(extracted_file))
                elif member.name.endswith('.xlsx'):
                    dfs.append(pd.read_excel(extracted_file))
            except Exception as e:
                print(f"Error reading the file {member.name}: {e}")

    if not dfs:
        raise ValueError("No files could be read successfully.")

    return dfs if len(dfs) > 1 else dfs[0]


