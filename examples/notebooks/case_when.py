import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multiple Conditions with case_when
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    # https://stackoverflow.com/q/19913659/7175713
    df = pd.DataFrame({"col1": list("ABBC"), "col2": list("ZZXY")})

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Single Condition:
    """)
    return


@app.cell
def _(df):
    df.case_when(
        df.col1 == "Z",  # condition
        "green",  # value if True
        "red",  # value if False
        column_name="color",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    - Multiple Conditions:
    """)
    return


@app.cell
def _(df):
    df.case_when(
        df.col2.eq("Z") & df.col1.eq("A"),
        "yellow",  # first condition and value
        df.col2.eq("Z") & df.col1.eq("B"),
        "blue",  # second condition and value
        df.col1.eq("B"),
        "purple",  # third condition and value
        "black",  # default if no condition is True
        column_name="color",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Anonymous functions (lambda) are supported as well:
    """)
    return


@app.cell
def _(pd):
    # https://stackoverflow.com/q/43391591/7175713
    raw_data = {"age1": [23, 45, 21], "age2": [10, 20, 50]}
    df_1 = pd.DataFrame(raw_data, columns=["age1", "age2"])
    df_1
    return (df_1,)


@app.cell
def _(df_1):
    df_1.case_when(
        lambda df: df.age1 - df.age2 > 0,
        lambda df: df.age1 - df.age2,
        lambda df: df.age2 - df.age1,
        column_name="diff",
    )  # condition  # value if True  # default if False
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    data types are preserved; under the hood it uses [pd.Series.mask](https://pandas.pydata.org/docs/reference/api/pandas.Series.mask.html):
    """)
    return


@app.cell
def _(df_1):
    df_2 = df_1.astype("Int64")
    df_2.dtypes
    return (df_2,)


@app.cell
def _(df_2):
    result = df_2.case_when(
        lambda df: df.age1 - df.age2 > 0,
        lambda df: df.age1 - df.age2,
        lambda df: df.age2 - df.age1,
        column_name="diff",
    )
    result
    return (result,)


@app.cell
def _(result):
    result.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The conditions can be a string, as long as they can be evaluated with `pd.eval` on the DataFrame, and return a boolean array:
    """)
    return


@app.cell
def _(pd):
    # https://stackoverflow.com/q/54653356/7175713
    data = {
        "name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
        "age": [42, 52, 36, 24, 73],
        "preTestScore": [4, 24, 31, 2, 3],
        "postTestScore": [25, 94, 57, 62, 70],
    }
    df_3 = pd.DataFrame(data, columns=["name", "age", "preTestScore", "postTestScore"])
    df_3
    return (df_3,)


@app.cell
def _(df_3):
    df_3.case_when(
        "age < 10",
        "baby",
        "10 <= age < 20",
        "kid",
        "20 <= age < 30",
        "young",
        "30 <= age < 50",
        "mature",
        "grandpa",
        column_name="elderly",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    When multiple conditions are satisfied, the first one is used:
    """)
    return


@app.cell
def _(pd):
    df_4 = range(3, 30, 3)
    df_4 = pd.DataFrame(df_4, columns=["odd"])
    df_4
    return (df_4,)


@app.cell
def _(df_4):
    df_4.case_when(
        df_4.odd % 9 == 0,
        "divisible by 9",
        "divisible by 3",
        column_name="div_by_3_or_9",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    lines 2, 5 and 8 are divisible by 3; however, because the *first* condition tests if it is divisible by 9, that outcome is used instead.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If `column_name` exists in the DataFrame, then that column's values will be replaced with the outcome of `case_when`:
    """)
    return


@app.cell
def _(df_4):
    df_4.case_when(
        df_4.odd % 9 == 0, "divisible by 9", "divisible by 3", column_name="odd"
    )
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
