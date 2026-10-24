"""Move the pending changelog entries into a versioned release section."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

_UNRELEASED = re.compile(r"^## \[Unreleased\]\s*$", re.MULTILINE)
_SECTION = re.compile(r"^## \[.+?\].*$", re.MULTILINE)


def rotate_changelog(
    path: Path,
    version: str,
    release_date: str,
) -> None:
    """Move the current ``[Unreleased]`` body under a version heading."""

    text = path.read_text(encoding="utf-8")
    matches = list(_UNRELEASED.finditer(text))
    if len(matches) != 1:
        raise ValueError(
            f"expected exactly one '## [Unreleased]' heading in {path}, "
            f"found {len(matches)}"
        )

    normalized_version = version.removeprefix("v")
    version_heading = f"## [v{normalized_version}] - {release_date}"
    if re.search(
        rf"^## \[v{re.escape(normalized_version)}\](?:\s|$)",
        text,
        re.MULTILINE,
    ):
        raise ValueError(f"version {normalized_version} already exists in {path}")

    unreleased = matches[0]
    next_section = _SECTION.search(text, unreleased.end())
    body_end = next_section.start() if next_section else len(text)
    body = text[unreleased.end() : body_end].strip("\n")

    replacement = f"## [Unreleased]\n\n{version_heading}\n"
    if body:
        replacement += f"\n{body}\n\n"
    else:
        replacement += "\n"

    updated = text[: unreleased.start()] + replacement + text[body_end:]
    path.write_text(updated.rstrip("\n") + "\n", encoding="utf-8")


def main() -> None:
    """Parse command-line arguments and rotate the requested changelog."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    args = parser.parse_args()
    rotate_changelog(args.path, args.version, args.date)


if __name__ == "__main__":
    main()
