import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Groupby_agg : Shortcut for assigning a groupby-transform to a new column.
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
    This notebook serves to show how to use the `groupby_agg` method from pyjanitor's general functions submodule.

    The `groupby_agg` method allows us to add the result of an aggregation from a grouping, as a new column, back to the dataframe.

    Currently in pandas, to append a column back to a dataframe, you do it in three steps:
    1. Groupby a column or columns
    2. Apply the `transform` method with an aggregate function on the grouping, and finally
    3. Assign the result of the transform to a new column in the dataframe.

    In pseudo-code, this might look something like:
    ```python
    df = df.assign(
        new_column_name=df.groupby(...)[...].transform(...)
    )
    ```

    The `groupby_agg` method allows you to achieve the same result in a single function call and with sensible arguments. The example below illustrates the use of this function.
    """)
    return


@app.cell
def _():
    # load modules
    import numpy as np
    import pandas as pd
    return np, pd


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examples
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Basic example
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We start off with a simple example.
    Given a `df` as defined below, we wish to use `groupby_agg` to find the average price for each item, and join the results back to the original dataframe.
    """)
    return


@app.cell
def _(pd):
    df = pd.DataFrame(
        {
            "item": ["shoe", "shoe", "bag", "shoe", "bag"],
            "MRP": [220, 450, 320, 200, 305],
            "number_sold": [100, 40, 56, 38, 25],
        }
    )
    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the output of `groupby_agg` contains the same number of rows as the input dataframe, i.e., the operation here is a groupby + transform.

    Here, `by` is the name(s) of the column(s) being grouped over. `agg` is the aggregate function (e.g. sum, mean, count...), which is beinng applied to the data in the column specified by `agg_column_name`.
    Finally, `new_column_name` is the name of the newly-added column containing the transformed values.
    """)
    return


@app.cell
def _(df):
    df_1 = df.groupby_agg(by='item', agg='mean', agg_column_name='MRP', new_column_name='Avg_MRP')
    df_1
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Specifying multiple columns to group over
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The basic example shown above specified a single column in `by` to group over.
    Grouping over multiple columns is also supported in general, since `groupby_agg` is just using the standard pandas `DataFrame.groupby` method under the hood.

    An example is shown below:
    """)
    return


@app.cell
def _(pd):
    df_2 = pd.DataFrame({'date': pd.date_range('2021-01-12', periods=5, freq='W'), 'item': ['sneaker', 'boots', 'sneaker', 'bag', 'bag'], 'MRP': [230, 450, 300, 200, 305]})
    df_2
    return (df_2,)


@app.cell
def _(df_2):
    df_3 = df_2.groupby_agg(by=['item', df_2['date'].dt.month], agg='mean', agg_column_name='MRP', new_column_name='Avg_MRP_by_item_month')
    df_3
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### The `dropna` parameter
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If the column(s) being grouped over (`by`) contains null values, you can include the null values as its own individual group, by passing `False` to `dropna`. Otherwise, the default behaviour is to `dropna=True`, in which case, the corresponding transformed values (in `new_column_name`) will be left as NaN.
    This feature was introduced in Pandas 1.1.

    You may read more about this parameter in the [Pandas user guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/groupby.html#id2).
    """)
    return


@app.cell
def _(np, pd):
    df_4 = pd.DataFrame({'name': ('black', 'black', 'black', 'red', 'red'), 'type': ('chair', 'chair', 'sofa', 'sofa', 'plate'), 'num': (4, 5, 12, 4, 3), 'nulls': (1, 1, np.nan, np.nan, 3)})
    df_4
    return (df_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's get the value counts of the values in the `nulls` column.
    Compare the two outputs from the following cell when `dropna` is set to True and False respectively:
    """)
    return


@app.cell
def _(df_4, display):
    print('With dropna=True (default)')
    _filtered_df = df_4.groupby_agg(by=['nulls'], agg='size', agg_column_name='type', new_column_name='counter', dropna=True)
    display(_filtered_df)
    print('With dropna=False')
    _filtered_df = df_4.groupby_agg(by=['nulls'], agg='size', agg_column_name='type', new_column_name='counter', dropna=False)
    display(_filtered_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Method chaining
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The `groupby_agg` method can be extended for different purposes. One of these is groupwise filtering, where only groups that meet a condition are retained.
    Let's explore this with an example, reusing one of the small dataframe from before:
    """)
    return


@app.cell
def _(np, pd):
    df_5 = pd.DataFrame({'name': ('black', 'black', 'black', 'red', 'red'), 'type': ('chair', 'chair', 'sofa', 'sofa', 'plate'), 'num': (4, 5, 12, 4, 3), 'nulls': (1, 1, np.nan, np.nan, 3)})
    _filtered_df = df_5.groupby_agg(by=['name', 'type'], agg='size', agg_column_name='type', new_column_name='counter').query('counter > 1')
    _filtered_df
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
