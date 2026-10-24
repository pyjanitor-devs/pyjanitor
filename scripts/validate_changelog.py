"""Validate that pull-request changelog entries are under ``[Unreleased]``."""

from __future__ import annotations

import argparse
import re
import subprocess
from typing import NamedTuple

_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,\d+)? @@")
_SECTION = re.compile(r"^## \[(?P<name>[^]]+)\]")
_REFERENCE = re.compile(r"^\[[^]]+\]:\s*\S+")


class AddedLine(NamedTuple):
    """A line added to the changelog and its line number in the new file."""

    number: int
    text: str


def _added_lines(diff: str) -> list[AddedLine]:
    """Extract added lines and their new-file line numbers from a diff."""

    added: list[AddedLine] = []
    new_line = None
    for line in diff.splitlines():
        hunk = _HUNK.match(line)
        if hunk:
            new_line = int(hunk["start"])
            continue
        if new_line is None or line.startswith("\\"):
            continue
        if line.startswith("+++"):
            continue
        if line.startswith("+"):
            added.append(AddedLine(new_line, line[1:]))
            new_line += 1
        elif line.startswith("-"):
            continue
        else:
            new_line += 1
    return added


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


def _unreleased_content(text: str) -> set[str]:
    """Return non-structural lines currently under ``[Unreleased]``."""

    lines = text.splitlines()
    start = next(
        (number for number, line in enumerate(lines) if line == "## [Unreleased]"),
        None,
    )
    if start is None:
        return set()

    content: set[str] = set()
    for line in lines[start + 1 :]:
        if _SECTION.match(line):
            break
        if line.strip() and not _REFERENCE.match(line):
            content.add(line.strip())
    return content


def validate_changelog_diff(base: str, head: str, diff: str) -> list[str]:
    """Return errors for new changelog entries outside ``[Unreleased]``.

    Lines already present in the base revision's ``[Unreleased]`` section are
    allowed to move into a versioned section. This accounts for release
    rotation commits without allowing a new entry to bypass the guard.
    """

    head_lines = head.splitlines()
    previous_unreleased = _unreleased_content(base)
    errors = []
    for added in _added_lines(diff):
        stripped = added.text.strip()
        if (
            not stripped
            or stripped.startswith("#")
            or _SECTION.match(stripped)
            or _REFERENCE.match(stripped)
        ):
            continue
        section = _section_at(head_lines, added.number)
        if section != "Unreleased" and stripped not in previous_unreleased:
            section_name = section or "no section"
            errors.append(
                f"CHANGELOG.md:{added.number}: entry added under [{section_name}]; "
                "new entries must be under [Unreleased]"
            )
    return errors


def _git_diff(base: str, head: str) -> str:
    """Return the changelog diff between two git revisions."""

    result = subprocess.run(
        [
            "git",
            "diff",
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
    args = parser.parse_args()

    result = subprocess.run(
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
        result.stdout,
        _git_diff(args.base, args.head),
    )
    if errors:
        raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
