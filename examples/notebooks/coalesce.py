import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coalesce
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    return np, pd


@app.cell
def _(np, pd):
    df = pd.DataFrame({"A": [1, 2, np.nan], "B": [np.nan, 10, np.nan], "C": [5, 10, 7]})

    df
    return (df,)


@app.cell
def _(df):
    df.coalesce("A", "B", "C", target_column_name="D")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If no target column is provided, then the first column is updated, with the null values removed:
    """)
    return


@app.cell
def _(df):
    df.coalesce("A", "B", "C")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If nulls remain, it can be filled with the `default_value`:
    """)
    return


@app.cell
def _(np, pd):
    df_1 = pd.DataFrame({"s1": [np.nan, np.nan, 6, 9, 9], "s2": [np.nan, 8, 7, 9, 9]})
    df_1
    return (df_1,)


@app.cell
def _(df_1):
    df_1.coalesce("s1", "s2", target_column_name="s3", default_value=0)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
