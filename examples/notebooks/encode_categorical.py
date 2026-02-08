import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Convert Columns to categoricals
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
            "col1": [2.0, 1.0, 3.0, 1.0, nan],
            "col2": ["a", "b", "c", "d", "a"],
            "col3": [
                "2020-01-01",
                "2020-01-02",
                "2020-01-03",
                "2020-01-04",
                "2020-01-05",
            ],
        }
    )

    df
    return (df,)


@app.cell
def _(df):
    df.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Specific columns can be converted to category type:
    """)
    return


@app.cell
def _(df):
    cat = df.encode_categorical(column_names=["col1", "col2", "col3"])
    return (cat,)


@app.cell
def _(cat):
    cat.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that for the code above, the categories were inferred from the columns, and is unordered:
    """)
    return


@app.cell
def _(cat):
    cat["col3"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Explicit categories can be provided, and ordered via the `kwargs`` parameter:
    """)
    return


@app.cell
def _(df):
    cat_1 = df.encode_categorical(
        col1=([3, 2, 1, 4], "appearance"), col2=(["a", "d", "c", "b"], "sort")
    )
    return (cat_1,)


@app.cell
def _(cat_1):
    cat_1["col1"]
    return


@app.cell
def _(cat_1):
    cat_1["col2"]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When the `order` parameter is `appearance`, the `categories` argument is used as-is; if the `order` is `sort`, the `categories` argument is sorted in ascending order; if `order` is `None``, then the `categories` argument is applied unordered.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A User Warning will be generated if some or all of the unique values in the column are not present in the provided `categories` argument.
    """)
    return


@app.cell
def _(df):
    cat_2 = df.encode_categorical(col1=([4, 5, 6], "appearance"))
    return (cat_2,)


@app.cell
def _(cat_2):
    cat_2["col1"]
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
