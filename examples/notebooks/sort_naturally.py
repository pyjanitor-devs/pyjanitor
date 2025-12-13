import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Using `sort_naturally`
    """)
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's say we have a pandas DataFrame that contains wells that we need to sort alphanumerically.
    """)
    return


@app.cell
def _(pd):
    data = {
        "Well": ["A21", "A3", "A21", "B2", "B51", "B12"],
        "Value": [1, 2, 13, 3, 4, 7],
    }
    df = pd.DataFrame(data)
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A human would sort it in the order:

        A3, A21, A21, B2, B12, B51

    However, default sorting in `pandas` doesn't allow that:
    """)
    return


@app.cell
def _(df):
    df.sort_values("Well")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Lexiographic sorting doesn't get us to where we want. A12 shouldn't come before A3, and B11 shouldn't come before B2. How might we fix this?
    """)
    return


@app.cell
def _(df):
    df.sort_naturally("Well")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now we're in sorting bliss! :)
    """)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
