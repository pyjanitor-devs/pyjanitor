import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Text Processing
    """)
    return


@app.cell
def _():
    import re

    import pandas as pd
    return pd, re


@app.cell
def _(pd):
    df = pd.DataFrame({"text": ["Ragnar", "sammywemmy", "ginger"], "code": [1, 2, 3]})

    df
    return (df,)


@app.cell
def _(df):
    df.process_text(column_name="text", string_function="lower")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For string methods with parameters, simply pass the keyword arguments::
    """)
    return


@app.cell
def _(df, re):
    df.process_text(
        column_name="text",
        string_function="extract",
        pat=r"(ag)",
        expand=False,
        flags=re.IGNORECASE,
    )
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
