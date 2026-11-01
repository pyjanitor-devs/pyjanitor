"""Tests for pull-request and release-time changelog validation."""

import sys
from difflib import unified_diff
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "validate_changelog.py"
_SPEC = spec_from_file_location("validate_changelog", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
validate_changelog_diff = _MODULE.validate_changelog_diff


def _diff(base: str, head: str) -> str:
    """Build a zero-context diff matching the CI validator's input."""

    return "".join(
        unified_diff(
            base.splitlines(keepends=True),
            head.splitlines(keepends=True),
            fromfile="a/CHANGELOG.md",
            tofile="b/CHANGELOG.md",
            n=0,
        )
    )


def test_accepts_entry_under_unreleased():
    """Accept a new entry in the unreleased section."""

    base = "# Changelog\n\n## [Unreleased]\n\n## [v1.0.0] - 2026-01-01\n"
    head = base.replace("## [Unreleased]\n", "## [Unreleased]\n\n- New feature.\n")

    assert validate_changelog_diff(base, head, _diff(base, head)) == []


def test_rejects_new_entry_under_versioned_section():
    """Reject a new entry added under a released version."""

    base = "# Changelog\n\n## [Unreleased]\n\n## [v1.0.0] - 2026-01-01\n"
    head = base.replace(
        "## [v1.0.0] - 2026-01-01\n", "## [v1.0.0] - 2026-01-01\n\n- Misfiled entry.\n"
    )

    errors = validate_changelog_diff(base, head, _diff(base, head))

    assert errors == [
        "CHANGELOG.md:7: entry added under [v1.0.0]; "
        "new entries must be under [Unreleased]"
    ]


def test_reproduces_release_race():
    """Catch an entry stranded under a version after release rotation."""

    base = (
        "# Changelog\n\n## [Unreleased]\n\n## [v1.1.0] - 2026-10-25\n\n"
        "- entry1\n\n[Unreleased]: https://example.test/v1.1.0...HEAD\n"
    )
    head = base.replace("- entry1\n", "- entry1\n- entry2\n")

    errors = validate_changelog_diff(base, head, _diff(base, head))

    assert errors == [
        "CHANGELOG.md:8: entry added under [v1.1.0]; "
        "new entries must be under [Unreleased]"
    ]


def test_allows_explicit_backfill_override():
    """Allow an explicit manual override for historical backfills."""

    base = "# Changelog\n\n## [Unreleased]\n\n## [v1.0.0] - 2026-01-01\n"
    head = base.replace(
        "## [v1.0.0] - 2026-01-01\n",
        "## [v1.0.0] - 2026-01-01\n\n- Backfill.\n",
    )

    assert (
        validate_changelog_diff(
            base,
            head,
            _diff(base, head),
            allow_backfill=True,
        )
        == []
    )


def test_allows_typo_fix_in_historical_section():
    """Allow a paired typo replacement in an old release section."""

    base = (
        "# Changelog\n\n## [Unreleased]\n\n## [v1.0.0] - 2026-01-01\n\n- recive data\n"
    )
    head = base.replace("- recive data", "- receive data")

    assert validate_changelog_diff(base, head, _diff(base, head)) == []


def test_does_not_exempt_new_entry_matching_unreleased_text():
    """Do not exempt a new entry based only on matching text."""

    base = (
        "# Changelog\n\n## [Unreleased]\n\n- Shared wording.\n\n"
        "## [v1.0.0] - 2026-01-01\n"
    )
    head = base.replace(
        "## [v1.0.0] - 2026-01-01\n",
        "## [v1.0.0] - 2026-01-01\n\n- Shared wording.\n",
    )

    assert validate_changelog_diff(base, head, _diff(base, head))


def test_accepts_link_definition_updates():
    """Ignore comparison-link definition changes."""

    base = "# Changelog\n\n## [Unreleased]\n\n[Unreleased]: https://example.test/v1...HEAD\n"
    head = base.replace("v1...HEAD", "v2...HEAD")

    assert validate_changelog_diff(base, head, _diff(base, head)) == []


def test_handles_added_line_starting_with_double_plus():
    """Parse added content beginning with two plus signs."""

    base = "# Changelog\n\n## [Unreleased]\n\n## [v1.0.0] - 2026-01-01\n"
    head = base.replace(
        "## [v1.0.0] - 2026-01-01\n", "## [v1.0.0] - 2026-01-01\n\n++ Misfiled.\n"
    )

    assert validate_changelog_diff(base, head, _diff(base, head))


def test_rejects_colored_or_unparsed_diff():
    """Fail closed for colored or otherwise unparseable diffs."""

    base = "# Changelog\n\n## [Unreleased]\n"
    colored = "\x1b[1mdiff --git a/CHANGELOG.md b/CHANGELOG.md\x1b[m\n"

    with pytest.raises(ValueError, match="colored"):
        validate_changelog_diff(base, base, colored)

    with pytest.raises(ValueError, match="could not parse"):
        validate_changelog_diff(base, base, "+- Misfiled\n")


def test_main_uses_base_and_head_revisions(monkeypatch):
    """Read both revisions and request an uncolored git diff."""

    calls = []

    class Result:
        """Minimal subprocess result for the command mock."""

        def __init__(self, stdout):
            self.stdout = stdout

    def fake_run(command, **kwargs):
        """Return changelog text or an empty diff for each mocked command."""

        calls.append(command)
        if command[1] == "show":
            return Result("# Changelog\n\n## [Unreleased]\n")
        return Result("")

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(
        sys, "argv", ["validate_changelog", "--base", "base", "--head", "head"]
    )

    _MODULE.main()

    assert calls == [
        ["git", "show", "head:CHANGELOG.md"],
        ["git", "show", "base:CHANGELOG.md"],
        [
            "git",
            "diff",
            "--no-color",
            "--unified=0",
            "--no-ext-diff",
            "base",
            "head",
            "--",
            "CHANGELOG.md",
        ],
    ]
