import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pivot Data from Long to Wide Form
    """)
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell
def _(pd):
    df = [
        {"name": "Alice", "variable": "wk1", "value": 5},
        {"name": "Alice", "variable": "wk2", "value": 9},
        {"name": "Alice", "variable": "wk3", "value": 20},
        {"name": "Alice", "variable": "wk4", "value": 22},
        {"name": "Bob", "variable": "wk1", "value": 7},
        {"name": "Bob", "variable": "wk2", "value": 11},
        {"name": "Bob", "variable": "wk3", "value": 17},
        {"name": "Bob", "variable": "wk4", "value": 33},
        {"name": "Carla", "variable": "wk1", "value": 6},
        {"name": "Carla", "variable": "wk2", "value": 13},
        {"name": "Carla", "variable": "wk3", "value": 39},
        {"name": "Carla", "variable": "wk4", "value": 40},
    ]


    df = pd.DataFrame(df)

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Reshaping to wide form:
    """)
    return


@app.cell
def _(df):
    df.pivot_wider(index="name", names_from="variable", values_from="value")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pivoting on multiple columns is possible :
    """)
    return


@app.cell
def _(pd):
    df_1 = [{'name': 1, 'n': 10.0, 'pct': 0.1}, {'name': 2, 'n': 20.0, 'pct': 0.2}, {'name': 3, 'n': 30.0, 'pct': 0.3}]
    df_1 = pd.DataFrame(df_1)
    df_1
    return (df_1,)


@app.cell
def _(df_1):
    df_1.assign(num=0).pivot_wider(index='num', names_from='name', values_from=['n', 'pct'], names_sep='_')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You may choose not to flatten the columns, by setting `flatten_levels` to ``False``:
    """)
    return


@app.cell
def _(pd):
    df_2 = [{'dep': 5.5, 'step': 1, 'a': 20, 'b': 30}, {'dep': 5.5, 'step': 2, 'a': 25, 'b': 37}, {'dep': 6.1, 'step': 1, 'a': 22, 'b': 19}, {'dep': 6.1, 'step': 2, 'a': 18, 'b': 29}]
    df_2 = pd.DataFrame(df_2)
    df_2
    return (df_2,)


@app.cell
def _(df_2):
    df_2.pivot_wider(index='dep', names_from='step', flatten_levels=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The order of the levels can be changed with the `levels_order` parameter, which internally uses pandas' [reorder_levels](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.reorder_levels.html):
    """)
    return


@app.cell
def _(df_2):
    df_2.pivot_wider(index='dep', names_from='step', flatten_levels=False, levels_order=['step', None])
    return


@app.cell
def _(df_2):
    df_2.pivot_wider(index='dep', names_from='step', flatten_levels=True)
    return


@app.cell
def _(df_2):
    df_2.pivot_wider(index='dep', names_from='step', flatten_levels=True, levels_order=['step', None])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    `names_sep` and `names_glue` come in handy in situations where `names_from` and/or `values_from` contain multiple variables; it is used primarily when the columns are flattened. The default value for `names_sep` is ``_``:
    """)
    return


@app.cell
def _(df_2):
    # default value of names_sep is '_'
    df_2.pivot_wider(index='dep', names_from='step')
    return


@app.cell
def _(df_2):
    df_2.pivot_wider(index='dep', names_from='step', names_sep='')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    With `names_glue` you can glue the individual levels (if MultiIndex) into one (similar to `names_sep`), or you can modify the final columns, as long as it can be passed to `pd.Index.map`:
    """)
    return


@app.cell
def _(df_2):
    # replicate `names_sep`
    df_2.pivot_wider(index='dep', names_from='step', names_sep=None, names_glue='_'.join)
    return


@app.cell
def _(df_2):
    # going beyond names_sep
    df_2.pivot_wider(index='dep', names_from='step', names_sep=None, names_glue=lambda col: f'{col[0]}_step{col[1]}')
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are scenarios where the column order of the final dataframe is important:
    """)
    return


@app.cell
def _(pd):
    df_3 = [{'Salesman': 'Knut', 'Height': 6, 'product': 'bat', 'price': 5}, {'Salesman': 'Knut', 'Height': 6, 'product': 'ball', 'price': 1}, {'Salesman': 'Knut', 'Height': 6, 'product': 'wand', 'price': 3}, {'Salesman': 'Steve', 'Height': 5, 'product': 'pen', 'price': 2}]
    df_3 = pd.DataFrame(df_3)
    df_3
    return (df_3,)


@app.cell
def _(df_3):
    idx = df_3.groupby(['Salesman', 'Height']).cumcount().add(1)
    df_3.assign(idx=idx).pivot_wider(index=['Salesman', 'Height'], names_from='idx')
    return (idx,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    To get the columns in a form where `product` alternates with `price`, we can combine `pivot_wider` (or plain `pd.pivot`) with `pd.sort_index` and `janitor.collapse_levels`:
    """)
    return


@app.cell
def _(df_3, idx):
    df_3.assign(idx=idx).pivot_wider(index=['Salesman', 'Height'], names_from='idx', flatten_levels=False).sort_index(level='idx', axis='columns', sort_remaining=False).collapse_levels().reset_index()
    return


@app.cell
def _(pd):
    df_4 = pd.DataFrame({'geoid': [1, 1, 13, 13], 'name': ['Alabama', 'Alabama', 'Georgia', 'Georgia'], 'variable': ['pop_renter', 'median_rent', 'pop_renter', 'median_rent'], 'estimate': [1434765, 747, 3592422, 927], 'error': [16736, 3, 33385, 3]})
    df_4
    return (df_4,)


@app.cell
def _(df_4):
    df_4.pivot_wider(index=['geoid', 'name'], names_from='variable', values_from=['estimate', 'error'], levels_order=['variable', None])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the reshaping above, we would like to maintain the order in `variable`, where `pop_renter` comes before `median_rent`; this can be achieved by converting the `variable` column to a categorical, before reshaping:
    """)
    return


@app.cell
def _(df_4):
    df_4.encode_categorical(variable=(None, 'appearance')).pivot_wider(index=['geoid', 'name'], names_from='variable', values_from=['estimate', 'error'], levels_order=['variable', None])
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
