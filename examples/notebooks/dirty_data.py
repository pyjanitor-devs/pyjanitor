import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Processing Dirty Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Background

    This is fake data generated to demonstrate the capabilities of `pyjanitor`.  It contains a bunch of common problems that we regularly encounter when working with data.  Let's go fix it!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Load Packages

    Importing `pyjanitor` is all that's needed to give Pandas Dataframes extra methods to work with your data.
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load Data
    """)
    return


@app.cell
def _(pd):
    df = pd.read_excel("dirty_data.xlsx", engine="openpyxl")
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleaning Column Names

    There are a bunch of problems with this data. Firstly, the column names are not lowercase, and they have spaces. This will make it cumbersome to use in a programmatic function. To solve this, we can use the `clean_names()` method.
    """)
    return


@app.cell
def _(df):
    df_clean = df.clean_names()
    df_clean.head(2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice now how the column names have been made better.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you squint at the unclean dataset, you'll notice one row and one column of data that are missing. We can also fix this! Building on top of the code block from above, let's now remove those empty columns using the `remove_empty()` method:
    """)
    return


@app.cell
def _(df):
    df_clean_1 = df.clean_names().remove_empty()
    df_clean_1.head(9).tail(4)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now this is starting to shape up well!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Renaming Individual Columns

    Next, let's rename some of the columns. `%_allocated` and `full_time?` contain non-alphanumeric characters, so they make it a bit harder to use. We can rename them using the :py:meth:`rename_column()` method:
    """)
    return


@app.cell
def _(df):
    df_clean_2 = (
        df.clean_names()
        .remove_empty()
        .rename_column("%_allocated", "percent_allocated")
        .rename_column("full_time_", "full_time")
    )
    df_clean_2.head(5)
    return (df_clean_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note how now we have really nice column names! You might be wondering why I'm not modifying the two certification columns -- that is the next thing we'll tackle.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Coalescing Columns

    If we look more closely at the two `certification` columns, we'll see that they look like this:
    """)
    return


@app.cell
def _(df_clean_2):
    df_clean_2[["certification", "certification_1"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rows 8 and 11 have NaN in the left certification column, but have a value in the right certification column. Let's assume for a moment that the left certification column is intended to record the first certification that a teacher had obtained. In this case, the values in the right certification column on rows 8 and 11 should be moved to the first column. Let's do that with Janitor, using the `coalesce()` method, which does the following:
    """)
    return


@app.cell
def _(df):
    df_clean_3 = (
        df.clean_names()
        .remove_empty()
        .rename_column("%_allocated", "percent_allocated")
        .rename_column("full_time_", "full_time")
        .coalesce("certification", "certification_1", new_column_name="certification")
    )
    df_clean_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Awesome stuff! Now we don't have two columns of scattered data, we have one column of densely populated data.`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Dealing with Excel Dates

    Finally, notice how the `hire_date` column isn't date formatted. It's got this weird Excel serialization.
    To clean up this data, we can use the :py:meth:`convert_excel_date` method.
    """)
    return


@app.cell
def _(df):
    df_clean_4 = (
        df.clean_names()
        .remove_empty()
        .rename_column("%_allocated", "percent_allocated")
        .rename_column("full_time_", "full_time")
        .coalesce(
            "certification", "certification_1", target_column_name="certification"
        )
        .convert_excel_date("hire_date")
    )
    df_clean_4
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We have a cleaned dataframe!
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
