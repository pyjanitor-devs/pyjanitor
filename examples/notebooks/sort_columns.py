import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## SOrt Column Values in Order
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
        [
            {"SalesMonth": "Jan", "Company2": 180.0, "Company3": 400.0},
            {"SalesMonth": "Feb", "Company2": 250.0, "Company3": 500.0},
            {"SalesMonth": "Feb", "Company2": 250.0, "Company3": 500.0},
            {"SalesMonth": "Mar", "Company2": nan, "Company3": 600.0},
            {"SalesMonth": "April", "Company2": 500.0, "Company3": 675.0},
        ]
    )

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Given the current DataFrame, we want to order the sales month in desc order. To achieve this we would assign the later months with smaller values with the latest month, such as April with the precedence of 0.
    """)
    return


@app.cell
def _(df):
    df.sort_column_value_order("SalesMonth", {"April": 1, "Mar": 2, "Feb": 3, "Jan": 4})
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
