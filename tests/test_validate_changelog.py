"""Tests for pull-request changelog validation."""

from difflib import unified_diff
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

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
    base = "# Changelog\n\n## [Unreleased]\n\n## [v1.0.0] - 2026-01-01\n"
    head = base.replace("## [Unreleased]\n", "## [Unreleased]\n\n- New feature.\n")

    assert validate_changelog_diff(base, head, _diff(base, head)) == []


def test_rejects_new_entry_under_versioned_section():
    base = "# Changelog\n\n## [Unreleased]\n\n## [v1.0.0] - 2026-01-01\n"
    head = base.replace(
        "## [v1.0.0] - 2026-01-01\n", "## [v1.0.0] - 2026-01-01\n\n- Misfiled entry.\n"
    )

    errors = validate_changelog_diff(base, head, _diff(base, head))

    assert errors == [
        "CHANGELOG.md:7: entry added under [v1.0.0]; "
        "new entries must be under [Unreleased]"
    ]


def test_allows_release_rotation_of_existing_unreleased_entry():
    base = (
        "# Changelog\n\n## [Unreleased]\n\n- Pending entry.\n\n"
        "## [v1.0.0] - 2026-01-01\n"
    )
    head = (
        "# Changelog\n\n## [Unreleased]\n\n## [v1.1.0] - 2026-10-25\n\n"
        "- Pending entry.\n\n## [v1.0.0] - 2026-01-01\n"
    )

    assert validate_changelog_diff(base, head, _diff(base, head)) == []


def test_ignores_link_definition_updates():
    base = "# Changelog\n\n## [Unreleased]\n\n[Unreleased]: https://example.test/v1...HEAD\n"
    head = base.replace("v1...HEAD", "v2...HEAD")

    assert validate_changelog_diff(base, head, _diff(base, head)) == []


def test_ignores_section_heading_additions():
    base = "# Changelog\n\n## [Unreleased]\n"
    head = base + "\n## [v1.0.0] - 2026-01-01\n"

    assert validate_changelog_diff(base, head, _diff(base, head)) == []
