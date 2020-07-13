"""Helper functions for running tests."""
import os


def running_on_ci() -> bool:
    """Return True if running on CI machine."""
    return os.environ.get("JANITOR_CI_MACHINE") is not None
