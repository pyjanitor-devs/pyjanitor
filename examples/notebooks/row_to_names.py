import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # row_to_names : Elevates a row to be the column names of a DataFrame.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Background
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This notebook serves to show a brief and simple example of how to swap column names using one of the rows in the dataframe.
    """)
    return


@app.cell
def _():
    from io import StringIO

    import pandas as pd
    return StringIO, pd


@app.cell
def _():
    data = """shoe, 220, 100
              shoe, 450, 40
              item, retail_price, cost
              shoe, 200, 38
              bag, 305, 25
           """
    return (data,)


@app.cell
def _(StringIO, data, pd):
    temp = pd.read_csv(StringIO(data), header=None)
    temp
    return (temp,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Looking at the dataframe above, we would love to use row 2 as our column names. One way to achieve this involves a couple of steps

    1. Use loc/iloc to assign row 2 to columns.
    2. Strip off any whitespace.
    2. Drop row 2 from the dataframe using the drop method.
    3. Set axis name to none.
    """)
    return


@app.cell
def _(temp):
    temp.columns = temp.iloc[2, :]
    temp.columns = temp.columns.str.strip()
    temp_1 = temp.drop(2, axis=0)
    temp_1 = temp_1.rename_axis(None, axis='columns')
    temp_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    However, the first two steps prevent us from method chaining. This is easily resolved using the row_to_names function
    """)
    return


@app.cell
def _(StringIO, data, pd):
    df = pd.read_csv(StringIO(data), header=None).row_to_names(
        row_number=2, remove_row=True
    )

    df
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
