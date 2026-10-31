"""Regression tests for the introductory example published in the README."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON_FENCE = re.compile(r"```python\n(.*?)```", re.DOTALL)
PYJANITOR_HEADING = "With `pyjanitor`"
CURRENT_VERBS = ("clean_names", "remove_empty", "encode_categorical")
DEPRECATED_CALLS = (
    "remove_columns",
    "rename_column",
    "rename_columns",
    "add_column",
    "add_columns",
)


def published_introductions() -> list[Path]:
    """Return each distinct copy of the published introduction.

    ``README.md`` is normally a symlink to ``mkdocs/index.md``. When it is a
    separate file, that copy is tested too.

    Returns:
        Introduction documents, one path per resolved file.
    """

    candidates = (
        REPO_ROOT / "mkdocs" / "index.md",
        REPO_ROOT / "README.md",
    )
    unique: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique.append(path)
    return unique


def introduction_text(markdown: str) -> str:
    """Return the page text that precedes the installation section.

    Args:
        markdown: Full Markdown source.

    Returns:
        The introduction, excluding the installation heading and everything
        after it.
    """

    marker = "\n## Installation\n"
    if marker not in markdown:
        raise AssertionError("introduction is missing the Installation heading")
    return markdown.split(marker, maxsplit=1)[0]


def python_blocks(markdown: str) -> list[re.Match[str]]:
    """Return fenced Python blocks in ``markdown``.

    Args:
        markdown: Markdown source.

    Returns:
        Matches whose group 1 is the Python source, in document order.
    """

    return list(PYTHON_FENCE.finditer(markdown))


def expected_sales() -> pd.DataFrame:
    """Return the frame every introductory example should produce.

    Returns:
        Sales table with cleaned names, empty columns removed, and the month
        stored as a categorical.
    """

    sales = pd.DataFrame(
        {
            "sales_month": ["Jan", "Feb", "Mar", "April"],
            "company_1": [150.0, 200.0, 300.0, 400.0],
            "company_2": [180.0, 250.0, float("nan"), 500.0],
            "company_3": [400.0, 500.0, 600.0, 675.0],
        }
    )
    return sales.astype({"sales_month": "category"})


@pytest.mark.documentation
@pytest.mark.parametrize(
    "document",
    published_introductions(),
    ids=lambda path: path.name,
)
def test_introductory_example_emits_no_deprecation_warning(document: Path) -> None:
    """The published chain must run when deprecation warnings are errors."""

    introduction = introduction_text(document.read_text(encoding="utf-8"))
    blocks = python_blocks(introduction)
    heading_at = introduction.find(PYJANITOR_HEADING)
    assert heading_at >= 0, f"{document} is missing the pyjanitor example heading"

    earlier = [match.group(1) for match in blocks if match.start() < heading_at]
    following = [match.group(1) for match in blocks if match.start() > heading_at]
    assert earlier, f"{document} is missing the sample data for the example"
    assert following, f"{document} is missing the Python block after the heading"
    pyjanitor_example = following[0]

    labeled_blocks = [
        (f"setup block {index}", source)
        for index, source in enumerate(earlier, start=1)
    ]
    labeled_blocks.append(("pyjanitor example", pyjanitor_example))

    namespace: dict[str, object] = {}
    results: list[pd.DataFrame] = []
    with warnings.catch_warnings():
        warnings.filterwarnings("error", category=FutureWarning)
        warnings.filterwarnings("error", category=DeprecationWarning)
        for label, block in labeled_blocks:
            try:
                exec(block, namespace)
            except (FutureWarning, DeprecationWarning) as caught:
                raise AssertionError(
                    f"{document} {label} raised {type(caught).__name__}: {caught}"
                ) from caught
            frame = namespace.get("df")
            if isinstance(frame, pd.DataFrame):
                results.append(frame.copy())

    for name in DEPRECATED_CALLS:
        assert name not in pyjanitor_example, (
            f"{document} pyjanitor example still calls {name}"
        )
    for verb in CURRENT_VERBS:
        assert f".{verb}(" in pyjanitor_example, (
            f"{document} pyjanitor example does not chain .{verb}()"
        )

    assert results, "the introductory examples did not build a DataFrame"
    expected = expected_sales()
    for result in results:
        pd.testing.assert_frame_equal(result, expected)
