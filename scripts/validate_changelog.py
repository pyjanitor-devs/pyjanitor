"""Validate that changelog entries are added under ``[Unreleased]``."""

from __future__ import annotations

import argparse
import re
import subprocess
from typing import NamedTuple

_HUNK = re.compile(r"^@@ -(?P<old>\d+)(?:,\d+)? \+(?P<new>\d+)(?:,\d+)? @@")
_SECTION = re.compile(r"^## \[(?P<name>[^]]+)\]", re.MULTILINE)
_REFERENCE = re.compile(r"^\[[^]]+\]:\s*\S+")
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


class ChangedLine(NamedTuple):
    """A changed line and its line number in one side of a diff."""

    number: int
    text: str


class DiffHunk(NamedTuple):
    """Added and removed lines from one unified-diff hunk."""

    added: tuple[ChangedLine, ...]
    removed: tuple[ChangedLine, ...]


def _parse_diff(diff: str) -> list[DiffHunk]:
    """Parse changelog hunks and fail closed for malformed colored diffs."""

    if _ANSI.search(diff):
        raise ValueError("refusing to parse a colored git diff")

    hunks: list[DiffHunk] = []
    added: list[ChangedLine] = []
    removed: list[ChangedLine] = []
    old_line = new_line = None

    def finish_hunk() -> None:
        """Store the current hunk before starting the next one."""

        if old_line is not None:
            hunks.append(DiffHunk(tuple(added), tuple(removed)))

    for line in diff.splitlines():
        hunk = _HUNK.match(line)
        if hunk:
            finish_hunk()
            added = []
            removed = []
            old_line = int(hunk["old"])
            new_line = int(hunk["new"])
            continue
        if line.startswith("+++") and old_line is None:
            continue
        if old_line is None:
            continue
        if line.startswith("+"):
            added.append(ChangedLine(new_line, line[1:]))
            new_line += 1
        elif line.startswith("-"):
            removed.append(ChangedLine(old_line, line[1:]))
            old_line += 1
        elif line.startswith("\\"):
            continue
        else:
            old_line += 1
            new_line += 1
    finish_hunk()

    if not hunks and any(line.startswith("+") for line in diff.splitlines()):
        raise ValueError("could not parse added lines from the git diff")
    return hunks


def _section_at(lines: list[str], line_number: int) -> str | None:
    """Return the changelog section containing a one-based line number."""

    section = None
    for number, line in enumerate(lines, start=1):
        match = _SECTION.match(line)
        if match:
            section = match["name"]
        if number == line_number:
            return section
    return section


def _structural_line(text: str) -> bool:
    """Return whether a changed line is changelog structure, not an entry."""

    stripped = text.strip()
    return (
        not stripped
        or stripped.startswith("#")
        or _SECTION.match(stripped) is not None
        or _REFERENCE.match(stripped) is not None
    )


def validate_changelog_diff(
    base: str,
    head: str,
    diff: str,
    allow_backfill: bool = False,
) -> list[str]:
    """Return errors for new entries outside ``[Unreleased]``.

    A paired replacement in the same historical section is treated as an edit
    to existing history, which permits typo fixes without permitting a new
    entry to pass merely because it has matching text elsewhere.
    """

    base_lines = base.splitlines()
    head_lines = head.splitlines()
    errors = []
    hunks = _parse_diff(diff)
    for hunk in hunks:
        for index, added in enumerate(hunk.added):
            if _structural_line(added.text):
                continue
            section = _section_at(head_lines, added.number)
            paired_removed = hunk.removed[index] if index < len(hunk.removed) else None
            same_section_edit = (
                paired_removed is not None
                and _section_at(base_lines, paired_removed.number) == section
            )
            if section != "Unreleased" and not same_section_edit and not allow_backfill:
                section_name = section or "no section"
                errors.append(
                    f"CHANGELOG.md:{added.number}: entry added under [{section_name}]; "
                    "new entries must be under [Unreleased]"
                )
    return errors


def _git_diff(base: str, head: str) -> str:
    """Return the uncolored changelog diff between two git revisions."""

    result = subprocess.run(
        [
            "git",
            "diff",
            "--no-color",
            "--unified=0",
            "--no-ext-diff",
            base,
            head,
            "--",
            "CHANGELOG.md",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> None:
    """Validate a changelog diff supplied by git revisions."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="base git revision")
    parser.add_argument("--head", default="HEAD", help="head git revision")
    parser.add_argument(
        "--allow-backfill",
        action="store_true",
        help="allow intentional changelog backfills",
    )
    args = parser.parse_args()

    head_result = subprocess.run(
        ["git", "show", f"{args.head}:CHANGELOG.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    base_result = subprocess.run(
        ["git", "show", f"{args.base}:CHANGELOG.md"],
        check=True,
        capture_output=True,
        text=True,
    )
    errors = validate_changelog_diff(
        base_result.stdout,
        head_result.stdout,
        _git_diff(args.base, args.head),
        allow_backfill=args.allow_backfill,
    )
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
