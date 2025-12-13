import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # pyjanitor Usage Walkthrough
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `pyjanitor` is a Python-based API on top of `pandas` inspired by the [janitor](https://github.com/sfirke/janitor) R package. It aims to provide a clean, understandable interface based on method chaining for common and less-common tasks involving data cleaning and `DataFrame` manipulation.

    The core philosophy and augmentations on `pandas`' approach to data cleaning and `DataFrame` manipulation include:

    - A method-chaining paradigm for coding efficiency & clarity of code through greatly improved readability.
    - Implementation of common, useful `DataFrame` manipulation tasks that saves on repetitive code.
    - Focus on active tense / verb approaches to function naming to provide at-a-glance understanding of a data manipulation pipeline.

    ## Why `pyjanitor`?

    Originally a simple port of the R package, `pyjanitor` has evolved from a set of convenient data cleaning routines into an experiment with the method-chaining paradigm.

    Data preprocessing is best expressed as a [directed acyclic graph (DAG)](https://en.wikipedia.org/wiki/Directed_acyclic_graph) of actions taken on data. We take a base data file as the starting point and perform actions on it such as removing null/empty rows, replacing them with other values, adding/renaming/removing columns of data, filtering rows, and more.

    The `pandas` API has been invaluable for the Python data science ecosystem and implements method chaining for a subset of methods as part of the API. For example, resetting indexes (`.reset_index()`), dropping null values (`.dropna()`), and more are accomplished via the appropriate `pd.DataFrame` method calls.

    Inspired by the R statistical language ecosystem, where consistent and good API design in the `dplyr` package enables end-users, who are not necessarily developers, to concisely express data processing code, `pyjanitor` has evolved into a language for expressing the data processing DAG for `pandas` users.

    ## What is method chaining?

    To accomplish these goals, actions for which we would need to invoke imperative-style statements can be replaced with method chains that allow the user to read off the logical order of actions taken. Note the annotated example below. First, we introduce the textual description of a sample data cleaning pipeline:

    - Create `DataFrame`.
    - Delete one column.
    - Drop rows with empty values in two particular columns.
    - Rename another two columns.
    - Add a new column.
    - Reset index to account for the missing row we removed above

    In `pandas` code, this would look as such:

    ```python
    df = pd.DataFrame(...)  # create a pandas DataFrame somehow.
    del df['column1']  # delete a column from the dataframe.
    df = df.dropna(subset=['column2', 'column3'])  # drop rows that have empty values in column 2 and 3.
    df = df.rename({'column2': 'unicorns', 'column3': 'dragons'})  # rename column2 and column3
    df['new_column'] = ['iterable', 'of', 'items']  # add a new column.
    df.reset_index(inplace=True, drop=True)  # reset index to account for the missing row we removed above
    ```

    ## The `pyjanitor` approach

    With `pyjanitor`, we enable method chaining with method names that are _verbs_ which describe the action taken:

    ```python
    df = (
        pd.DataFrame(...)
        .remove_columns(['column1'])
        .dropna(subset=['column2', 'column3'])
        .rename_column('column2', 'unicorns')
        .rename_column('column3', 'dragons')
        .add_column('new_column', ['iterable', 'of', 'items'])
        .reset_index(drop=True)
    )
    ```

    We believe the `pyjanitor` chaining-based approach leads to much cleaner code where the intent of a series of `DataFrame` manipulations is much more immediately clear.

    `pyjanitor`’s etymology has a two-fold relationship to “cleanliness”. Firstly, it’s about extending `pandas` with convenient data cleaning routines. Secondly, it’s about providing a cleaner, method-chaining, verb-based API for common `pandas` routines.

    ## A survey of `pyjanitor` functions

    - Cleaning column names (multi-indexes are possible!)
    - Removing empty rows and columns
    - Identifying duplicate entries
    - Encoding columns as categorical
    - Splitting your data into features and targets (for machine learning)
    - Adding, removing, and renaming columns
    - Coalesce multiple columns into a single column
    - Convert excel date (serial format) into a Python datetime format
    - Expand a single column that has delimited, categorical values into dummy-encoded variables

    A full list of functionality that `pyjanitor` implements can be found in the [API docs](https://ericmjl.github.io/pyjanitor/).

    ## Some things that are different

    Some `pyjanitor` methods are `DataFrame`-mutating operations, i.e., in place. Given that in a method-chaining paradigm, `DataFrame`s that would be created at each step of the chain cannot be accessed anyway, duplication of data at each step would lead to unnecessary, potential considerable slowdowns and increased memory usage due to data-copying operations. The severity of such copying scales with `DataFrame` size. Take care to understand which functions change the original `DataFrame` you are chaining on if it is necessary to preserve that data. If it is, you can simply `.copy()` it as the first step in a `df.copy().operation1().operation2()...` chain.

    ## How it works

    `pyjanitor` relies on the [Pandas Flavor](https://github.com/Zsailer/pandas_flavor) package to register new functions as object methods that can be called directly on `DataFrame`s. For example:

    ```python
    import pandas as pd
    import pandas_flavor as pf


    @pf.register_dataframe_method
    def remove_column(df, column_name: str):
        del df[column_name]
        return df

    df = (
        pd.read_csv('my_data.csv')
        .remove_column('my_column_name')
        .operation2(...)
    )
    ```

    Importing the `janitor` package immediately registers these functions. The fact that each `DataFrame` method `pyjanitor` registers returns the `DataFrame` is what gives it the capability to method chain.

    Note that `pyjanitor` explicitly does not modify any existing `pandas` methods / functionality.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Demo of various `DataFrame` manipulation tasks using `pyjanitor`
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here, we'll walk through some useful `pyjanitor`-based approaches to cleaning and manipulating `DataFrame`s.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Code preamble:
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
    ### Let's take a look at our dataset:
    """)
    return


@app.cell
def _(pd):
    df = pd.read_excel("dirty_data.xlsx", engine="openpyxl")
    df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can see that this dataset is dirty in a number of ways, including the following:

      * Column names contain spaces, punctuation marks, and inconsistent casing
      * One row (`7`) with completely missing data
      * One column (`do not edit! --->`) with completely missing data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Clean up our data using a `pyjanitor` method-chaining pipeline

    Let's run through a demo `DataFrame` cleaning procedure:
    """)
    return


@app.cell
def _(pd):
    cleaned_df = (
        pd.read_excel("dirty_data.xlsx", engine="openpyxl")
        .clean_names()
        .remove_empty()
        .rename_column("%_allocated", "percent_allocated")
        .rename_column("full_time_", "full_time")
        .coalesce(["certification", "certification_1"], "certification")
        .encode_categorical(["subject", "employee_status", "full_time"])
        .convert_excel_date("hire_date")
        .reset_index(drop=True)
    )

    cleaned_df
    return (cleaned_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The cleaned `DataFrame` looks much better and quite a bit more usable for our downstream tasks.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Step-by-step walkthrough of `pyjanitor` `DataFrame` manipulations

    Just for clearer understanding of the above, let's see how `pyjanitor` progressively modified the data.

    Loading data in:
    """)
    return


@app.cell
def _(pd):
    df_1 = pd.read_excel('dirty_data.xlsx', engine='openpyxl')
    df_1
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Clean up names by removing whitespace, punctuation / symbols, capitalization:
    """)
    return


@app.cell
def _(df_1):
    df_2 = df_1.clean_names()
    df_2
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Remove entirely empty rows / columns:
    """)
    return


@app.cell
def _(df_2):
    df_3 = df_2.remove_empty()
    df_3
    return (df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Rename particular columns:
    """)
    return


@app.cell
def _(df_3):
    df_4 = df_3.rename_column('%_allocated', 'percent_allocated').rename_column('full_time_', 'full_time')
    df_4
    return (df_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Take first non-`NaN` row value in two columns:
    """)
    return


@app.cell
def _(df_4):
    df_5 = df_4.coalesce(['certification', 'certification_1'], 'certification')
    df_5
    return (df_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Convert string object rows to categorical to save on memory consumption and speed up access:
    """)
    return


@app.cell
def _(df_5):
    df_5.dtypes
    return


@app.cell
def _(df_5):
    df_5.encode_categorical(['subject', 'employee_status', 'full_time'])
    df_5.dtypes
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Convert Excel date-formatted column to a more interpretable format:
    """)
    return


@app.cell
def _(df_5):
    df_5.convert_excel_date('hire_date')
    df_5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Example analysis of the data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's perform analysis on the above, cleaned `DataFrame`. First we add some additional, randomly-generated data. Note that we `.copy()` the original to preserve it, given that the following would otherwise modify it:
    """)
    return


@app.cell
def _(cleaned_df, np):
    data_df = cleaned_df.copy().add_columns(
        lucky_number=np.random.randint(0, 10, len(cleaned_df)),
        age=np.random.randint(10, 100, len(cleaned_df)),
        employee_of_month_count=np.random.randint(0, 5, len(cleaned_df)),
    )

    data_df
    return (data_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Calculate mean, median of all numerical columns after grouping by employee status. Use `.collapse_levels()`, a `pyjanitor` convenience function, to convert the `DataFrame` returned by `.agg()` from having multi-level columns (because we supplied a list of aggregation operations) to single-level by concatenating the level names with an underscore:
    """)
    return


@app.cell
def _(data_df):
    stats_df = (
        data_df.groupby("employee_status")
        .agg(["mean", "median"])
        .collapse_levels()
        .reset_index()
    )

    stats_df
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
