"""Tests for test helper functions."""

from helpers import running_on_ci


def test_running_on_ci_local(monkeypatch):
    """Test running_on_ci run on local machine returns False."""
    monkeypatch.delenv("JANITOR_CI_MACHINE", raising=False)
    assert running_on_ci() is False


def test_running_on_ci_ci(monkeypatch):
    """Test running_on_ci run on CI machine returns True."""
    monkeypatch.setenv("JANITOR_CI_MACHINE", "1")
    assert running_on_ci() is True
