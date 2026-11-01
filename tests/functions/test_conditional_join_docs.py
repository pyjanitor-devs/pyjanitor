"""The staircase example in the ``conditional_join`` docstring stays true.

The example is written as fenced code plus Markdown tables so the rendered
docs show real tables. Doctests cannot check that, so this test runs the
fenced code and compares each table with the DataFrame it describes.
"""

import importlib
import inspect
import re
import textwrap

import numpy as np
import pandas as pd

START = "To ask only whether a left row has a partner"
END = "The staircase answers existence only"
TABLE = re.compile(r"^\|.*\|\n^\|[ -|]+\|\n(?:^\|.*\|\n?)+", re.MULTILINE)
EXPECTED_TABLES = ["a", "b", "stairs", "marked", "semi", "anti", "joined"]


def _section():
    doc = inspect.getdoc(
        importlib.import_module("janitor.functions.conditional_join").conditional_join
    )
    assert START in doc and END in doc
    start = doc.rindex("\n", 0, doc.index(START)) + 1  # keep the line's indent
    return textwrap.dedent(doc[start : doc.index(END)])


def _as_markdown(frame):
    def cell(value):
        if isinstance(value, (float, np.floating)) and np.isnan(value):
            return "NaN"
        return str(value)

    columns = [""] + [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for label, row in zip(frame.index, frame.itertuples(index=False)):
        cells = [str(label)] + [cell(value) for value in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def test_docstring_example_tables_match_the_code():
    section = _section()
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", section, flags=re.DOTALL):
        exec(block, namespace)  # noqa: S102 - executes our own docstring
    tables = [match.group(0).strip() for match in TABLE.finditer(section)]
    assert len(tables) == len(EXPECTED_TABLES)
    for name, table in zip(EXPECTED_TABLES, tables):
        assert table == _as_markdown(namespace[name]), name


def test_docstring_example_semi_and_anti_partition_the_left_rows():
    namespace = {}
    for block in re.findall(r"```python\n(.*?)```", _section(), flags=re.DOTALL):
        exec(block, namespace)  # noqa: S102
    a, semi, anti = namespace["a"], namespace["semi"], namespace["anti"]
    assert sorted([*semi.index, *anti.index]) == list(a.index)
    pd.testing.assert_frame_equal(
        anti.reset_index(drop=True), namespace["joined"].reset_index(drop=True)
    )
