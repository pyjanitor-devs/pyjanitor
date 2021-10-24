"""
A script to count the number of functions inside each source file.

Can be used for many purposes.

Intended to be run from pyjanitor's top-level directory.


"""
import ast
import os
from pathlib import Path


def count_number_of_functions(filepath):
    """Count number of functions inside a .py file."""
    # Taken from: https://stackoverflow.com/a/37514895/1274908
    with open(filepath, "r+") as f:
        tree = ast.parse(f.read())
        return sum(isinstance(exp, ast.FunctionDef) for exp in tree.body)


def janitor_submodules():
    """Yield a list of janitor submodules and their full paths."""
    files = [f for f in os.listdir("janitor") if f.endswith(".py")]

    for file in files:
        yield Path("janitor") / file


def main():
    """Main executable function."""
    for filepath in janitor_submodules():
        num_funcs = count_number_of_functions(filepath)
        print(filepath, num_funcs)


if __name__ == "__main__":
    main()
