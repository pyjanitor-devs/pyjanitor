"""
Author: Eric J. Ma, borrowing code from Samuel Oranyeli.
This Python script checks the docs pages for functions
that have not been added to the "General Functions" API docs.

Only the "general_functions.rst" needs the checks,
as the others are auto-populated.
This situation simply evolved out of how the docs are structured,
I am not sure exactly how it turned out this way. :)
"""

import re
from pathlib import Path
from typing import List, Tuple


def extract_function_names(
    test_folder: Path, exclude_names: List[str]
) -> List[str]:  # skipcq
    """Extract function names from the list of functions."""
    function_names = []  # skipcq
    for name in test_folder.iterdir():
        if not name.is_dir() and path_does_not_contain(name, exclude_names):
            function_names.append(name.stem.split("_", 1)[-1].strip())
    return function_names


def extract_documented_functions(docs: Path) -> List[str]:  # skipcq
    """Extract documented functions from docs page."""
    pattern = re.compile(r"\s{4}[a-zA-Z_]+")

    # get the names in the general_functions page
    with docs.open() as doc:
        doc_functions = [  # skipcq
            pattern.search(line).group().strip()
            for line in doc
            if pattern.search(line)
        ]
    return doc_functions


def path_does_not_contain(path: Path, names: List[str]) -> bool:
    """Check if path does not contain a list of names."""
    for name in names:
        if name in str(path):
            return False
    return True


def extract_folder_names(test_dir: Path) -> Tuple[Path, str]:
    """Extract folder names.

    This function could be used later.
    """
    # folder_names = []
    for name in test_dir.iterdir():
        if name.is_dir() and path_does_not_contain(
            name, ["__pycache__", "test_data", "data_description"]
        ):
            module_name = str(name).split("/")[-1]
            yield Path(name), module_name


test_folder = Path("./tests/functions")
exclude_names = ["__pycache__", "test_data", "data_description"]
function_names = extract_function_names(test_folder, exclude_names)
docs = Path("./docs/reference/general_functions.rst")
doc_functions = extract_documented_functions(docs)
missing_funcs = set(function_names).difference(doc_functions)

if len(missing_funcs) > 0:
    raise Exception(
        f"The following functions have not yet been added: {missing_funcs}"
    )
