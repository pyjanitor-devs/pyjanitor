"""Tests for the automated changelog rotation script."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))

from scripts.update_changelog import rotate_changelog


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

    rotate_changelog(changelog, "v0.2.0", "2026-10-25")

    assert changelog.read_text(encoding="utf-8") == (
        "# Changelog\n\n"
        "## [Unreleased]\n\n"
        "## [v0.2.0] - 2026-10-25\n\n"
        "- [BUG] Fix a pending issue.\n"
        "- [ENH] Add a pending feature.\n\n"
        "## [v0.1.0] - 2026-01-01\n"
        "- An older entry.\n"
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
