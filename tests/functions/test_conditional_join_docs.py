"""The staircase tip in the ``conditional_join`` docstring stays true.

Doctests do not run inside an admonition, because the docs build renders
``>>>`` there as nested block quotes. The tip therefore uses fenced ``python``
blocks, each followed by a fenced ``text`` block holding the output pandas
prints. This test runs every python block in order and compares the repr of its
last expression with the text block that follows it.
"""

import ast
import importlib
import inspect
import re
import textwrap

START = '!!! tip "Existence tests: the staircase"'
FENCE = re.compile(r"```(python|text)\n(.*?)```", re.DOTALL)


def _trim(text):
    """Ignore trailing spaces, which pandas pads onto some headers and the
    repository's pre-commit hooks strip from docstrings."""
    return "\n".join(line.rstrip() for line in text.split("\n"))


def _tip():
    function = importlib.import_module("janitor.functions.conditional_join")
    doc = inspect.getdoc(function.conditional_join)
    assert START in doc
    start = doc.rindex("\n", 0, doc.index(START)) + 1  # keep the line's indent
    rest = doc[start:]
    # the tip ends at the next admonition, which is the version history
    end = rest.index('!!! abstract "Version Changed"')
    lines = textwrap.dedent(rest[:end]).split("\n")
    # drop the tip's own four-space content indent; keep pandas' spacing intact
    return "\n".join(
        line[4:] if line.startswith("    ") else line for line in lines[1:]
    )


def _blocks():
    """Return ``(code, expected_output_or_None)`` for each python block."""
    fences = FENCE.findall(_tip())
    pairs = []
    for index, (language, text) in enumerate(fences):
        if language != "python":
            continue
        following = fences[index + 1] if index + 1 < len(fences) else None
        output = (
            following[1].rstrip("\n") if following and following[0] == "text" else None
        )
        pairs.append((text, output))
    return pairs


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
    namespace = {}
    for code, _ in _blocks():
        exec(code, namespace)  # noqa: S102 - executes our own docstring
    marked, a = namespace["marked"], namespace["a"]
    semi = marked.loc[marked["has_partner"], ["x", "y"]]
    anti = marked.loc[~marked["has_partner"], ["x", "y"]]
    assert sorted([*semi.index, *anti.index]) == list(a.index)
    joined = namespace["a"].conditional_join(
        namespace["b"], ("x", "x", "<="), ("y", "y", ">"), keep="any"
    )
    assert joined["left"].reset_index(drop=True).equals(semi.reset_index(drop=True))
    assert (joined["left"]["x"] <= joined["right"]["x"]).all()
    assert (joined["left"]["y"] > joined["right"]["y"]).all()
