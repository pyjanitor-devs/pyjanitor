"""Helper functions for running tests."""
import os
import sys
from typing import Any, Optional

import pytest
from _pytest.outcomes import Skipped
from packaging.version import Version


def running_on_ci() -> bool:
    """Return True if running on CI machine."""
    return os.environ.get("JANITOR_CI_MACHINE") is not None


def importorskip(
    modname: str,
    minversion: Optional[str] = None,
    reason: Optional[str] = None,
) -> Any:
    """Import and return the requested module ``modname``.
        Doesn't allow skips on CI machine.
        Borrowed and modified from ``pytest.importorskip``.
    :param str modname: the name of the module to import
    :param str minversion: if given, the imported module's ``__version__``
        attribute must be at least this minimal version, otherwise the test is
        still skipped.
    :param str reason: if given, this reason is shown as the message when the
        module cannot be imported.
    :returns: The imported module. This should be assigned to its canonical
        name.
    Example::
        docutils = pytest.importorskip("docutils")
    """
    # JANITOR_CI_MACHINE is True if tests run on CI, where JANITOR_CI_MACHINE env variable exists
    JANITOR_CI_MACHINE = running_on_ci()
    if JANITOR_CI_MACHINE:
        import warnings

        compile(modname, "", "eval")  # to catch syntaxerrors

        with warnings.catch_warnings():
            # make sure to ignore ImportWarnings that might happen because
            # of existing directories with the same name we're trying to
            # import but without a __init__.py file
            warnings.simplefilter("ignore")
            __import__(modname)
        mod = sys.modules[modname]
        if minversion is None:
            return mod
        verattr = getattr(mod, "__version__", None)
        if minversion is not None:
            if verattr is None or Version(verattr) < Version(minversion):
                raise Skipped(
                    "module %r has __version__ %r, required is: %r"
                    % (modname, verattr, minversion),
                    allow_module_level=True,
                )
        return mod
    else:
        return pytest.importorskip(
            modname=modname, minversion=minversion, reason=reason
        )
