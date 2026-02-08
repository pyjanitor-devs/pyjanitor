import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Fill on a Single Column
    """)
    return


@app.cell
def _():
    import pandas as pd
    from numpy import nan

    return nan, pd


@app.cell
def _(nan, pd):
    df = pd.DataFrame(
        {
            "text": ["ragnar", nan, "sammywemmy", nan, "ginger"],
            "code": [nan, 2.0, 3.0, nan, 5.0],
        }
    )

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Fill on a single column:
    """)
    return


@app.cell
def _(df):
    df.fill_direction(code="up")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Fill on multiple columns:
    """)
    return


@app.cell
def _(df):
    df.fill_direction(text="down", code="down")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Fill multiple columns in different directions:
    """)
    return


@app.cell
def _(df):
    df.fill_direction(text="up", code="down")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
