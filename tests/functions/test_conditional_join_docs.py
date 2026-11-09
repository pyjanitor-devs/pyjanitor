"""The staircase tip in the ``conditional_join`` docstring stays true.

Doctests do not run inside an admonition, because the docs build renders
``>>>`` there as nested block quotes. The tip therefore uses fenced ``python``
blocks, each followed by a fenced ``text`` block holding the output pandas
prints. These tests run every python block in order, compare the repr of its
last expression with the text block that follows it, and check the operator
tables by running the recipe they describe against a brute-force join.
"""

import ast
import importlib
import inspect
import itertools
import operator
import re

import numpy as np
import pandas as pd

START = '!!! tip "Existence tests: the staircase"'
FENCE = re.compile(r"```(python|text)\n(.*?)```", re.DOTALL)
OPERATORS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def _trim(text):
    """Ignore trailing spaces, which pandas pads onto some headers and the
    repository's pre-commit hooks strip from docstrings."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _tip():
    """Return the tip's content with the admonition's own indent removed."""
    function = importlib.import_module("janitor.functions.conditional_join")
    lines = inspect.getdoc(function.conditional_join).split("\n")
    first = next(i for i, line in enumerate(lines) if line.strip() == START)
    indent = len(lines[first]) - len(lines[first].lstrip())
    content = []
    for line in lines[first + 1 :]:
        # the tip ends at the first non-blank line that is not indented under it
        if line.strip() and len(line) - len(line.lstrip()) <= indent:
            break
        content.append(line[indent + 4 :] if line.strip() else "")
    return "\n".join(content)


def _blocks():
    """Return ``(code, expected_output_or_None)`` for each python block."""
    fences = FENCE.findall(_tip())
    pairs = []
    for index, (language, text) in enumerate(fences):
        if language != "python":
            continue
        following = fences[index + 1] if index + 1 < len(fences) else None
        has_output = following and following[0] == "text"
        pairs.append((text, following[1].rstrip("\n") if has_output else None))
    return pairs


def _run_blocks():
    namespace = {}
    for code, _ in _blocks():
        exec(code, namespace)  # noqa: S102 - executes our own docstring
    return namespace


def test_docstring_tip_outputs_match_the_code():
    namespace = {}
    blocks = _blocks()
    assert len(blocks) >= 8
    checked = 0
    for code, expected in blocks:
        tree = ast.parse(code)
        *head, last = tree.body
        for node in head:
            exec(compile(ast.Module([node], []), "<tip>", "exec"), namespace)
        if expected is None:
            exec(compile(ast.Module([last], []), "<tip>", "exec"), namespace)
            continue
        assert isinstance(last, ast.Expr), code
        value = eval(compile(ast.Expression(last.value), "<tip>", "eval"), namespace)
        assert _trim(repr(value)) == _trim(expected), code
        checked += 1
    assert checked >= 7


def test_docstring_tip_semi_and_anti_partition_the_left_rows():
    namespace = _run_blocks()
    marked, a, b = namespace["marked"], namespace["a"], namespace["b"]
    semi = marked.loc[marked["has_partner"], ["x", "y"]]
    anti = marked.loc[~marked["has_partner"], ["x", "y"]]
    assert sorted([*semi.index, *anti.index]) == list(a.index)
    joined = a.conditional_join(b, ("x", "x", "<="), ("y", "y", ">"), keep="any")
    # compare as sets of rows: no left-row order is promised
    left = joined["left"]
    assert sorted(map(tuple, left.to_numpy())) == sorted(map(tuple, semi.to_numpy()))
    assert (left["x"] <= joined["right"]["x"]).all()
    assert (left["y"] > joined["right"]["y"]).all()


def _tables():
    """Parse the tip's Markdown tables into ``{first header cell: rows}``."""
    tables, current = {}, None
    for line in _tip().split("\n"):
        if not line.startswith("|"):
            current = None
            continue
        cells = [cell.strip().replace("``", "") for cell in line.strip("|").split("|")]
        if set("".join(cells)) <= set("- "):
            continue
        if current is None:
            current = tables.setdefault(cells[0], [])
            continue
        current.append(cells)
    return tables


def _recipe(a, b, x_op, y_op, rules):
    """Mark the rows of ``a`` with a partner in ``b``, following the tip's tables."""
    x_rule = rules["x"][f"a.x {x_op} b.x"]
    y_rule = rules["y"][f"a.y {y_op} b.y"]
    order = np.argsort(b["x"].to_numpy(), kind="stable")
    bx, by = b["x"].to_numpy()[order], b["y"].to_numpy()[order]
    to_the_end = "to the end" in x_rule["counts"]
    running = np.minimum if y_rule["running"].startswith("minimum") else np.maximum
    if to_the_end:  # take the running value from the largest x down
        steps = running.accumulate(by[::-1])[::-1]
    else:  # take it from the smallest x up, without reversing
        steps = running.accumulate(by)
    position = np.searchsorted(bx, a["x"].to_numpy(), side=x_rule["side"])
    read = position + (-1 if "position - 1" in x_rule["step"] else 0)
    usable = (read >= 0) & (read < len(bx))
    step = steps[np.where(usable, read, 0)]
    compare = OPERATORS[y_rule["compare"]]
    return usable & compare(a["y"].to_numpy(), step)


def test_docstring_tip_operator_tables_are_correct_for_every_pair():
    tables = _tables()
    rules = {"x": {}, "y": {}}
    for row in tables["Condition on x"]:
        rules["x"][row[0]] = {
            "side": row[1].strip('"'),
            "counts": row[2],
            "step": row[3],
        }
    for row in tables["Condition on y"]:
        rules["y"][row[0]] = {"running": row[1], "compare": row[2]}
    assert len(rules["x"]) == 4 and len(rules["y"]) == 4
    rng = np.random.default_rng(0)
    for x_op, y_op in itertools.product(OPERATORS, OPERATORS):
        for _ in range(40):
            # small value range so that ties are common
            a = pd.DataFrame({"x": rng.integers(0, 5, 12), "y": rng.integers(0, 5, 12)})
            b = pd.DataFrame({"x": rng.integers(0, 5, 9), "y": rng.integers(0, 5, 9)})
            expected = (
                OPERATORS[x_op](a["x"].to_numpy()[:, None], b["x"].to_numpy())
                & OPERATORS[y_op](a["y"].to_numpy()[:, None], b["y"].to_numpy())
            ).any(axis=1)
            actual = _recipe(a, b, x_op, y_op, rules)
            np.testing.assert_array_equal(actual, expected, err_msg=f"{x_op} {y_op}")


def test_docstring_has_exactly_one_staircase_tip():
    function = importlib.import_module("janitor.functions.conditional_join")
    doc = inspect.getdoc(function.conditional_join)
    assert doc.count(START) == 1
    assert doc.count("To ask only whether a left row has a partner") == 1
