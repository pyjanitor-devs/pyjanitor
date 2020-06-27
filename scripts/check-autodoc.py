"""
This Python script checks the docs pages for functions
that have not been added to the "General Functions" API docs.
"""

import re
from pathlib import Path

test_folder = Path("./tests/functions")

# Extract the names from the test_functions folder
# function_names = [
#     name.stem.split("_", 1)[-1].strip() for name in test_folder.iterdir()
# ]
function_names = []
for name in test_folder.iterdir():
    if not name.is_dir():
        print(name)
        print(name.stem.split("_", 1)[-1].strip())
        function_names.append(name.stem.split("_", 1)[-1].strip())

docs = Path("./docs/reference/general_functions.rst")

pattern = re.compile("\s{4}[a-zA-Z_]+")

# get the names in the general_functions page
with docs.open() as doc:
    doc_functions = [
        pattern.search(line).group().strip()
        for line in doc
        if pattern.search(line)
    ]

# get the functions that have not yet been added
missing_funcs = set(function_names).difference(doc_functions)

if len(missing_funcs) > 0:
    raise Exception(
        f"The following functions have not yet been added: {missing_funcs}"
    )
