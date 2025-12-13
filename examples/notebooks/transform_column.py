import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Transforming columns
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Introduction

    There are two ways to use the `transform_column` function: by passing in a function that operates elementwise, or by passing in a function that operates columnwise.

    We will show you both in this notebook.
    """)
    return


@app.cell
def _():
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Numeric Data
    """)
    return


@app.cell
def _(np):
    data = np.random.normal(size=(1_000_000, 4))
    return (data,)


@app.cell
def _(data, pd):
    df = pd.DataFrame(data).clean_names()
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Using the elementwise application:
    """)
    return


@app.cell
def _(df, np):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    # We are using a lambda function that operates on each element,
    # to highlight the point about elementwise operations.
    df.transform_column("0", lambda x: np.abs(x), "abs_0")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And now using columnwise application:
    """)
    return


@app.cell
def _(df, np):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    df.transform_column("0", lambda s: np.abs(s), elementwise=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Because `np.abs` is vectorizable over the entire series,
    it runs about 50X faster.
    If you know your function is vectorizable,
    then take advantage of the fact,
    and use it inside `transform_column`.
    After all, all that `transform_column` has done
    is provide a method-chainable way of applying the function.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## String Data

    Let's see it in action with string-type data.
    """)
    return


@app.cell
def _(pd):
    from random import choice


    def make_strings(length: int):
        return "".join(choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(length))


    strings = (make_strings(30) for _ in range(1_000_000))

    stringdf = pd.DataFrame({"data": list(strings)})
    return (stringdf,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Firstly, by raw function application:
    """)
    return


@app.function
def first_five(s):
    return s.str[0:5]


@app.cell
def _(stringdf):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    stringdf.assign(data=first_five(stringdf["data"]))
    return


@app.cell
def _(stringdf):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    first_five(stringdf["data"])
    return


@app.cell
def _(stringdf):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    stringdf["data"].str[0:5]
    return


@app.cell
def _(stringdf):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    stringdf["data"].apply(lambda x: x[0:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    It appears assigning the result to a column comes with a bit of overhead.

    Now, by using `transform_column` with default settings:
    """)
    return


@app.cell
def _(stringdf):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    stringdf.transform_column("data", lambda x: x[0:5])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now by using `transform_column` while also leveraging string methods:
    """)
    return


@app.cell
def _(stringdf):
    # magic command not supported in marimo; please file an issue to add support
    # %%timeit
    stringdf.transform_column("data", first_five, elementwise=False)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
