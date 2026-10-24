"""Move the pending changelog entries into a versioned release section."""

from __future__ import annotations

import argparse
import re
from datetime import date
from pathlib import Path

_UNRELEASED = re.compile(r"^## \[Unreleased\]\s*$", re.MULTILINE)
_SECTION = re.compile(r"^## \[.+?\].*$", re.MULTILINE)
_UNRELEASED_LINK = re.compile(r"^\[Unreleased\]: (?P<url>\S+)[ \t]*$", re.MULTILINE)


def rotate_changelog(
    path: Path,
    version: str,
    release_date: str,
    previous_version: str | None = None,
) -> bool:
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
    if not body.strip():
        body = "- No user-facing changes; see the generated release notes."

    replacement = f"## [Unreleased]\n\n{version_heading}\n"
    replacement += f"\n{body}\n\n"

    updated = text[: unreleased.start()] + replacement + text[body_end:]
    unreleased_link = _UNRELEASED_LINK.search(updated)
    if unreleased_link is None:
        raise ValueError(f"expected an '[Unreleased]' link in {path}")

    if previous_version is None:
        previous_version = _latest_version_from_url(unreleased_link["url"])
    previous_version = previous_version.strip().removeprefix("v")
    if not previous_version:
        raise ValueError("previous version must not be empty")
    base_url = unreleased_link["url"].split("/compare/", 1)[0]
    compare_url = f"{base_url}/compare"
    links = (
        f"[Unreleased]: {compare_url}/v{normalized_version}...HEAD\n"
        f"[v{normalized_version}]: "
        f"{compare_url}/v{previous_version}...v{normalized_version}"
    )
    updated = (
        updated[: unreleased_link.start()] + links + updated[unreleased_link.end() :]
    )
    path.write_text(updated.rstrip("\n") + "\n", encoding="utf-8")
    return True


def _latest_version_from_url(url: str) -> str:
    """Extract the previous version from an Unreleased comparison URL."""

    match = re.search(r"/compare/(v\d+\.\d+\.\d+)\.\.\.HEAD$", url)
    if match is None:
        raise ValueError(f"could not determine the previous version from {url}")
    return match.group(1)


def main() -> None:
    """Parse command-line arguments and rotate the requested changelog."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", type=Path, default=Path("CHANGELOG.md"))
    parser.add_argument("--version", required=True)
    parser.add_argument("--date", default=date.today().isoformat())
    parser.add_argument("--previous-version")
    args = parser.parse_args()
    rotate_changelog(
        args.path,
        args.version,
        args.date,
        args.previous_version,
    )


if __name__ == "__main__":
    main()
