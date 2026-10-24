"""Tests for the automated changelog rotation script."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts" / "update_changelog.py"
_SPEC = spec_from_file_location("update_changelog", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
rotate_changelog = _MODULE.rotate_changelog


def test_rotate_changelog_moves_entries_and_resets_unreleased(tmp_path: Path):
    """Move pending entries into a dated version section."""

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n"
        "- [BUG] Fix a pending issue.\n"
        "- [ENH] Add a pending feature.\n\n"
        "## [v0.1.0] - 2026-01-01\n"
        "- An older entry.\n",
        encoding="utf-8",
    )

    changelog.write_text(
        changelog.read_text(encoding="utf-8")
        + "\n[Unreleased]: https://example.com/compare/v0.1.0...HEAD\n\n"
        "[v0.0.1]: https://example.com/compare/v0.0.0...v0.0.1\n",
        encoding="utf-8",
    )

    assert rotate_changelog(changelog, "v0.2.0", "2026-10-25")

    assert changelog.read_text(encoding="utf-8") == (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [v0.2.0] - 2026-10-25\n\n"
        "- [BUG] Fix a pending issue.\n"
        "- [ENH] Add a pending feature.\n\n"
        "## [v0.1.0] - 2026-01-01\n"
        "- An older entry.\n\n"
        "[Unreleased]: https://example.com/compare/v0.2.0...HEAD\n\n"
        "[v0.2.0]: https://example.com/compare/v0.1.0...v0.2.0\n\n"
        "[v0.0.1]: https://example.com/compare/v0.0.0...v0.0.1\n"
    )


def test_rotate_changelog_rejects_duplicate_version(tmp_path: Path):
    """Reject a release when its version already exists."""

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n## [Unreleased]\n\n## [v0.2.0] - 2026-01-01\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="already exists"):
        rotate_changelog(changelog, "0.2.0", "2026-10-25")


def test_rotate_changelog_records_empty_unreleased(tmp_path: Path):
    """Create a release heading when no entries are pending."""

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [v0.1.0] - 2026-01-01\n\n"
        "[Unreleased]: https://example.com/compare/v0.1.0...HEAD\n",
        encoding="utf-8",
    )

    assert rotate_changelog(changelog, "0.2.0", "2026-10-25")
    assert changelog.read_text(encoding="utf-8") == (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [v0.2.0] - 2026-10-25\n\n"
        "-   No user-facing changes; see the generated release notes.\n\n"
        "## [v0.1.0] - 2026-01-01\n\n"
        "[Unreleased]: https://example.com/compare/v0.2.0...HEAD\n\n"
        "[v0.2.0]: https://example.com/compare/v0.1.0...v0.2.0\n"
    )


def test_rotate_changelog_rejects_empty_previous_version(tmp_path: Path):
    """Reject an empty previous version instead of writing a broken link."""

    changelog = tmp_path / "CHANGELOG.md"
    changelog.write_text(
        "# Changelog\n\n"
        "## [Unreleased]\n"
        "- [BUG] Fix a pending issue.\n\n"
        "[Unreleased]: https://example.com/compare/v0.1.0...HEAD\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="must not be empty"):
        rotate_changelog(changelog, "0.2.0", "2026-10-25", "")
