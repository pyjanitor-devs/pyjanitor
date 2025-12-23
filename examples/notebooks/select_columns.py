import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Selecting Columns
    """)
    return


@app.cell
def _():
    import datetime
    import re

    import numpy as np
    import pandas as pd
    from pandas.api.types import is_datetime64_dtype

    from janitor import patterns
    return datetime, is_datetime64_dtype, np, patterns, pd, re


@app.cell
def _(datetime, np, pd):
    df = pd.DataFrame(
        {
            "id": [0, 1],
            "Name": ["ABC", "XYZ"],
            "code": [1, 2],
            "code1": [4, np.nan],
            "code2": ["8", 5],
            "type": ["S", "R"],
            "type1": ["E", np.nan],
            "type2": ["T", "U"],
            "code3": pd.Series(["a", "b"], dtype="category"),
            "type3": pd.to_datetime(
                [np.datetime64("2018-01-01"), datetime.datetime(2018, 1, 1)]
            ),
        }
    )

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Select by string:
    """)
    return


@app.cell
def _(df):
    df.select_columns("id")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Select via shell-like glob strings (`*`) is possible:
    """)
    return


@app.cell
def _(df):
    df.select_columns("type*")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Select by slice:
    """)
    return


@app.cell
def _(df):
    df.select_columns(slice("code1", "type1"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Select by `Callable` (the callable is applied to every column  and should return a single `True` or `False` per column):
    """)
    return


@app.cell
def _(df, is_datetime64_dtype):
    df.select_columns(is_datetime64_dtype)
    return


@app.cell
def _(df):
    df.select_columns(lambda x: x.name.startswith("code") or x.name.endswith("1"))
    return


@app.cell
def _(df):
    df.select_columns(lambda x: x.isna().any())
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Select by regular expression:
    """)
    return


@app.cell
def _(df, re):
    df.select_columns(re.compile("\\d+"))
    return


@app.cell
def _(df, patterns):
    # same as above, with janitor.patterns
    # simply a wrapper around re.compile

    df.select_columns(patterns("\\d+"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Select a combination of the above (you can combine any of the previous options):
    """)
    return


@app.cell
def _(df):
    df.select_columns("id", "code*", slice("code", "code2"))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - You can also pass a sequence of booleans:
    """)
    return


@app.cell
def _(df):
    df.select_columns([True, False, True, True, True, False, False, False, True, False])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Setting `invert` to `True` returns the complement of the columns provided:
    """)
    return


@app.cell
def _(df):
    df.select_columns("id", "code*", slice("code", "code2"), invert=True)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
