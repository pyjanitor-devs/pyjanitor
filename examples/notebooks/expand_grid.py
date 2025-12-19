import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Expand_grid : Create a dataframe from all combinations of inputs.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Background

    This notebook serves to show examples of how expand_grid works. Expand_grid aims to offer similar functionality to R's [expand_grid](https://tidyr.tidyverse.org/reference/expand_grid.html) function.<br><br>
    Expand_grid creates a dataframe from a combination of all inputs. <br><br>One requirement is that a dictionary be provided. If a dataframe is provided, a key must be provided as well.

    Some of the examples used here are from tidyr's expand_grid page and from Pandas' cookbook.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd

    from janitor import expand_grid
    return expand_grid, np, pd


@app.cell
def _(expand_grid):
    _data = {'x': [1, 2, 3], 'y': [1, 2]}
    result = expand_grid(others=_data)
    result
    return


@app.cell
def _(expand_grid):
    # combination of letters
    _data = {'l1': list('abcde'), 'l2': list('ABCDE')}
    letters = expand_grid(others=_data)
    letters.head(10)
    return


@app.cell
def _(expand_grid):
    _data = {'height': [60, 70], 'weight': [100, 140, 180], 'sex': ['Male', 'Female']}
    measurements = expand_grid(others=_data)
    measurements
    return (measurements,)


@app.cell
def _(expand_grid, np):
    _data = {'x1': np.array([[1, 3], [2, 4]]), 'x2': np.array([[5, 7], [6, 8]])}
    result_1 = expand_grid(others=_data)
    result_1
    return


@app.cell
def _(pd):
    df = pd.DataFrame({'x': [1, 2], 'y': [2, 1]})
    _data = {'z': [1, 2, 3]}
    result_2 = df.expand_grid(df_key='df', others=_data)
    result_2
    return


@app.cell
def _(expand_grid, pd):
    df1 = pd.DataFrame({('x', 'y'): range(1, 3), ('y', 'x'): [2, 1]})
    df2 = pd.DataFrame({'x': [1, 2, 3], 'y': [3, 2, 1]})
    df3 = pd.DataFrame({'x': [2, 3], 'y': ['a', 'b']})
    _data = {'df1': df1, 'df2': df2, 'df3': df3}
    result_3 = expand_grid(others=_data)
    result_3
    return (result_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Columns can be flattened with pyjanitor's `collapse_levels`:
    """)
    return


@app.cell
def _(result_3):
    result_3.collapse_levels()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or a level dropped with Pandas' `droplevel` method:
    """)
    return


@app.cell
def _(measurements):
    measurements.droplevel(level=-1, axis="columns")
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
