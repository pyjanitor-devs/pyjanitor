#!/usr/bin/env python3
"""Discover contributors from the GitHub API and update .all-contributorsrc.

Recognition sources (contribution type in parentheses):
  - commit authors via /repos/{o}/{r}/contributors         -> "code"  (default)
  - issue authors via /repos/{o}/{r}/issues                -> "bug"   (opt-in)

Commit-author discovery is on by default because it is unambiguous and
high-signal. Issue-author discovery is opt-in (set INCLUDE_ISSUES=1) because
issue threads on busy repos attract disposable/spam accounts; when enabled it
adds the "bug" contribution type for issue authors.

Merged pull-request authors are already captured by the contributors endpoint
(their commits are attributed to them on merge), so a separate PR query is
not needed. PR reviews and discussion contributions are not auto-discovered
yet; add those manually via the all-contributors workflow if desired.

Behaviour:
  - Appends newly discovered contributors to .all-contributorsrc.
  - Additively merges newly discovered contribution types for existing
    contributors (e.g. an existing "code" contributor who also filed issues
    gains "bug"). Never removes entries or types, so manual tags such as
    "talk", "question", "ideas" are always preserved.
  - Excludes bots and automated/AI accounts (see BOT_DENYLIST plus any login
    ending in "[bot]").

Usage (run from the repository root):
  GITHUB_TOKEN=$(gh auth token) python scripts/update_contributors.py

After this script updates .all-contributorsrc, regenerate the rendered table
in README.md with:
  npx all-contributors-cli@latest generate
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

API = "https://api.github.com"
REPO = "pyjanitor-devs/pyjanitor"
RC_PATH = Path(".all-contributorsrc")
MAX_PAGES = 15
REQUEST_TIMEOUT = 30
MAX_RETRIES = 3

# Accounts that commit/act via automation and should not be credited as people.
BOT_DENYLIST = {
    "dependabot",
    "dependabot-preview",
    "pyup-bot",
    "pre-commit-ci",
    "deepsource-autofix",
    "cursoragent",
    "github-actions",
    "github-action",
    "netlify",
    "renovate",
    "renovate-bot",
    "renovate-approve",
    "allcontributors",
    "imgbot",
    "semantic-release-bot",
    "codeclimate",
    "mergify",
}

SOURCE_TO_TYPE = {
    "commits": "code",
    "issues": "bug",
}


class GitHub:
    """Minimal authenticated client for the GitHub REST API used by this script.

    The ``token`` is sent as a bearer token on every request. Transient HTTP
    errors (rate limits 403/429, 5xx) are retried with exponential backoff;
    other non-2xx responses raise ``urllib.error.HTTPError``.
    """

    def __init__(self, token: str) -> None:
        self.token = token

    def _get(self, url: str) -> tuple[list | dict, str | None]:
        """Issue an authenticated GET with timeout, retrying transient errors.

        Args:
            url: The full API URL to request.

        Returns:
            A tuple of the decoded JSON body and the raw ``Link`` header
            value (``None`` if absent).

        Raises:
            urllib.error.HTTPError: For non-transient HTTP failures
                (e.g. 401, 404) after retries are exhausted.
        """
        req = urllib.request.Request(
            url,
            headers={
                "Authorization": f"Bearer {self.token}",
                "Accept": "application/vnd.github+json",
                "User-Agent": "pyjanitor-contributor-sync",
            },
        )
        last_exc: urllib.error.HTTPError | None = None
        for attempt in range(MAX_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT) as resp:
                    body = json.loads(resp.read().decode("utf-8"))
                return body, resp.headers.get("Link")
            except urllib.error.HTTPError as exc:
                last_exc = exc
                transient = exc.code in (403, 429) or 500 <= exc.code < 600
                if not transient or attempt == MAX_RETRIES - 1:
                    raise
                wait = 2**attempt
                label = "rate-limited" if exc.code in (403, 429) else f"HTTP {exc.code}"
                print(
                    f"  {label}; retrying in {wait}s "
                    f"(attempt {attempt + 1}/{MAX_RETRIES})",
                    file=sys.stderr,
                )
                time.sleep(wait)
        assert last_exc is not None
        raise last_exc

    def _paginate(self, url: str, label: str) -> list:
        """Follow the ``rel="next"`` Link header, up to ``MAX_PAGES`` pages.

        Args:
            url: The first page URL.
            label: Human-readable label printed to stderr for progress.

        Returns:
            The concatenated list of items across all fetched pages.
        """
        items: list = []
        next_url: str | None = url
        page = 0
        while next_url and page < MAX_PAGES:
            body, link = self._get(next_url)
            if not isinstance(body, list) or not body:
                break
            items.extend(body)
            page += 1
            next_url = self._next_link(link)
        if next_url:
            print(
                f"  {label}: WARNING — truncated at {MAX_PAGES} pages, "
                "more pages remain",
                file=sys.stderr,
            )
        else:
            print(
                f"  {label}: fetched {len(items)} items ({page} page(s))",
                file=sys.stderr,
            )
        return items

    @staticmethod
    def _next_link(link: str | None) -> str | None:
        """Extract the ``rel="next"`` URL from a GitHub ``Link`` header.

        Args:
            link: The raw ``Link`` header value, or ``None``.

        Returns:
            The next-page URL, or ``None`` if there is no next page.
        """
        if not link:
            return None
        for part in link.split(","):
            seg = part.strip()
            if 'rel="next"' in seg:
                start = seg.find("<")
                end = seg.find(">")
                if start != -1 and end != -1 and end > start:
                    return seg[start + 1 : end]
        return None

    def contributors(self) -> list[str]:
        """Return logins of all commit authors (``/contributors`` endpoint)."""
        logins = [
            c["login"]
            for c in self._paginate(
                f"{API}/repos/{REPO}/contributors?per_page=100", "contributors"
            )
            if isinstance(c, dict) and c.get("login")
        ]
        return logins

    def issue_authors(self) -> list[str]:
        """Return logins of issue authors (excluding pull requests)."""
        issues = self._paginate(
            f"{API}/repos/{REPO}/issues?state=all&per_page=100", "issues"
        )
        logins = [
            i["user"]["login"]
            for i in issues
            if isinstance(i, dict)
            and "pull_request" not in i
            and isinstance(i.get("user"), dict)
            and i["user"].get("login")
        ]
        return logins

    def user(self, login: str) -> dict:
        """Fetch profile metadata for a login (``/users/{login}``).

        Args:
            login: The GitHub login to look up.

        Returns:
            The user object (``name``, ``avatar_url``, ``html_url``, ...),
            or an empty dict if the lookup fails (e.g. the account was
            deleted or renamed).
        """
        try:
            body, _ = self._get(f"{API}/users/{urllib.parse.quote(login)}")
        except urllib.error.HTTPError as exc:
            print(
                f"  warning: user lookup for {login} failed: {exc.code}",
                file=sys.stderr,
            )
            return {}
        return body if isinstance(body, dict) else {}


def is_denied(login: str) -> bool:
    """Return True if a login is a bot/automation account that must be skipped."""
    low = login.lower()
    if low.endswith("[bot]"):
        return True
    return low in BOT_DENYLIST


def discover(gh: GitHub) -> dict[str, set[str]]:
    """Return ``{login: {contribution_types}}`` for all recognized sources.

    Args:
        gh: An authenticated ``GitHub`` client.

    Returns:
        A mapping from contributor login to the set of contribution types
        discovered for them this run.
    """
    found: dict[str, set[str]] = {}
    for login in gh.contributors():
        if not is_denied(login):
            found.setdefault(login, set()).add(SOURCE_TO_TYPE["commits"])
    if os.environ.get("INCLUDE_ISSUES") == "1":
        for login in gh.issue_authors():
            if not is_denied(login):
                found.setdefault(login, set()).add(SOURCE_TO_TYPE["issues"])
    else:
        print(
            "Issue-author discovery skipped (set INCLUDE_ISSUES=1 to enable).",
            file=sys.stderr,
        )
    return found


def load_config() -> dict:
    """Load and return the parsed ``.all-contributorsrc`` document."""
    with RC_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def save_config(config: dict) -> None:
    """Atomically write the config to ``.all-contributorsrc`` (2-space indent).

    Writes to a sibling temp file then ``os.replace``s it into place, so a
    crash mid-write cannot corrupt the existing config.

    Args:
        config: The all-contributors config to serialize.
    """
    tmp = RC_PATH.with_suffix(".rc.tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(config, fh, indent=2, ensure_ascii=False)
        fh.write("\n")
    os.replace(tmp, RC_PATH)


def main() -> int:
    """Discover contributors, update ``.all-contributorsrc``, and report.

    Merges discovered contribution types into existing entries additively
    (never removing manual tags) and appends brand-new contributors with
    metadata fetched from the GitHub API.

    Returns:
        ``0`` on success.
    """
    token = os.environ.get("GITHUB_TOKEN")
    if not token:
        sys.exit("GITHUB_TOKEN env var is required.")
    gh = GitHub(token)
    discovered = discover(gh)
    config = load_config()
    existing = {
        c["login"]: c
        for c in config.get("contributors", [])
        if isinstance(c, dict) and c.get("login")
    }
    existing_lower = {k.lower(): k for k in existing}

    added: list[str] = []
    updated: list[str] = []

    for login, types in discovered.items():
        key = existing_lower.get(login.lower())
        if key is None:
            profile = gh.user(login)
            existing[login] = {
                "login": login,
                "name": profile.get("name") or login,
                "avatar_url": (
                    profile.get("avatar_url")
                    or f"https://github.com/identicons/{login}.png"
                ),
                "profile": profile.get("html_url") or f"https://github.com/{login}",
                "contributions": sorted(types),
            }
            existing_lower[login.lower()] = login
            added.append(login)
        else:
            entry = existing[key]
            before = set(entry.get("contributions", []))
            merged = before | types
            if merged != before:
                entry["contributions"] = sorted(merged)
                updated.append(key)

    config["contributors"] = list(existing.values())
    save_config(config)

    print(f"Discovered {len(discovered)} contributors from GitHub.")
    print(f"Added {len(added)} new: {', '.join(sorted(added)) or '(none)'}")
    print(f"Updated types for {len(updated)}: {', '.join(sorted(updated)) or '(none)'}")
    print("Next: run `npx all-contributors-cli@latest generate` to refresh README.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
