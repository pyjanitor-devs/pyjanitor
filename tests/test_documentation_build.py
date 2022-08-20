"""Tests for documentation build."""

import pytest

try:
    import mkdocs

    # Even if 'mkdocs' is installed
    # 'mkdocs' is could be imported as <module 'mkdocs' (namespace)>
    # Need to check if 'mkdocs' has '__version__' attribute

    mkdocs_installed = hasattr(mkdocs, "__version__")
except (ImportError, AttributeError):
    mkdocs_installed = False


@pytest.mark.skipif(
    not mkdocs_installed,
    reason=(
        "Requires the MkDocs library. "
        "And only test documentation in documentation building CI."
    ),
)
def test_docs_general_functions_present():
    """Test that all docs pages build correctly.

    TODO: There has to be a better way to automatically check that
    all of the functions are present in the docs.
    This is an awesome thing that we could use help with in the future.
    """

    # We want to check that the following keywords are all present.
    # I put in a subsample of general functions.
    # This can be made much more robust.
    rendered_correctly = False
    with open("./site/api/functions/index.html", "r+") as f:
        for line in f.readlines():
            if "add_columns" in line or "update_where" in line:
                rendered_correctly = True
    assert rendered_correctly
