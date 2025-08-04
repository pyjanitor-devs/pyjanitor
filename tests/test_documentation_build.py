"""Tests for documentation build."""

import os

import pytest

# If `mkdocs` wasn't installed in environment, just skip.
# Can't use `pytest.importorskip("mkdocs")`, 'mkdocs' is also
# a folder name to pyjanitor project.
pytest.importorskip("mkdocstrings")


@pytest.mark.documentation
def test_docs_general_functions_present():
    """Test that all docs pages build correctly.

    TODO: There has to be a better way to automatically check that
    all of the functions are present in the docs.
    This is an awesome thing that we could use help with in the future.
    """
    # Build docs using mkdocs
    os.system("mkdocs build --clean")

    # We want to check that the following keywords are all present.
    # I put in a subsample of general functions.
    # This can be made much more robust.
    rendered_correctly = False
    with open("./site/api/functions/index.html", "r+") as f:
        for line in f.readlines():
            if "add_columns" in line or "update_where" in line:
                rendered_correctly = True
    assert rendered_correctly
