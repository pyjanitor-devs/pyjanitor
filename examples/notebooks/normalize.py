import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Normalization and Standardization

    [Normalization](https://en.wikipedia.org/wiki/Normalization_(statistics)) makes data more meaningful by converting absolute values into comparisons with related values.  [Chris Vallier](https://github.com/jcvall) has produced this demonstration of normalization using PyJanitor.

    pyjanitor functions demonstrated here:

    - [min_max_scale](../reference/functions.html#janitor.functions.min_max_scale)

    - [transform_column](../reference/functions.html#janitor.functions.transform_column)
    """)
    return


@app.cell
def _():
    import pandas as pd
    import seaborn as sns

    sns.set(style="whitegrid")
    return pd, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Load data

    We'll use a dataset with fuel efficiency in miles per gallon ("mpg"), engine displacement in cubic centimeters ("disp"), and horsepower ("hp") for a variety of car models.  It's a crazy, but customary, mix of units.
    """)
    return


@app.cell
def _(pd):
    csv_file = "https://gist.githubusercontent.com/seankross/a412dfbd88b3db70b74b/raw/5f23f993cd87c283ce766e7ac6b329ee7cc2e1d1/mtcars.csv"
    cars_df = pd.read_csv(csv_file)
    return (cars_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Quantities without units are dangerous, so let's use pyjanitor's `rename_column`...
    """)
    return


@app.cell
def _(cars_df):
    cars_df_1 = cars_df.rename_column("disp", "disp_cc")
    return (cars_df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Examine raw data
    """)
    return


@app.cell
def _(cars_df_1):
    cars_df_1.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Visualize

    Each value makes more sense viewed in comparison to the other models.  We'll use simple [Seaborn](https://seaborn.pydata.org/) bar plots.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### mpg by model
    """)
    return


@app.cell
def _(cars_df_1, sns):
    cars_df_2 = cars_df_1.sort_values("mpg", ascending=False)
    sns.barplot(y="model", x="mpg", data=cars_df_2, color="b", orient="h")
    return (cars_df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### displacement by model
    """)
    return


@app.cell
def _(cars_df_2, sns):
    cars_df_3 = cars_df_2.sort_values("disp_cc", ascending=False)
    sns.barplot(y="model", x="disp_cc", data=cars_df_3, color="b", orient="h")
    return (cars_df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### horsepower by model
    """)
    return


@app.cell
def _(cars_df_3, sns):
    cars_df_4 = cars_df_3.sort_values("hp", ascending=False)
    sns.barplot(y="model", x="hp", data=cars_df_4, color="b", orient="h")
    return (cars_df_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## [min-max normalization](https://en.wikipedia.org/wiki/Feature_scaling#Rescaling_\(min-max_normalization\))

    First we'll use pyjanitor's [min_max_scale](../reference/functions.html#janitor.functions.min_max_scale) to rescale the `mpg`, `disp_cc`, and `hp` columns in-place so that each value varies from 0 to 1.
    """)
    return


@app.cell
def _(cars_df_4):
    cars_df_4.min_max_scale(col_name="mpg", new_max=1, new_min=0).min_max_scale(
        col_name="disp_cc", new_max=1, new_min=0
    ).min_max_scale(col_name="hp", new_max=1, new_min=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The shapes of the bar graphs remain the same, but the horizontal axes show the new scale.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### mpg (min-max normalized)
    """)
    return


@app.cell
def _(cars_df_4, sns):
    cars_df_5 = cars_df_4.sort_values("mpg", ascending=False)
    sns.barplot(y="model", x="mpg", data=cars_df_5, color="b", orient="h")
    return (cars_df_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### displacement (min-max normalized)
    """)
    return


@app.cell
def _(cars_df_5, sns):
    cars_df_6 = cars_df_5.sort_values("disp_cc", ascending=False)
    sns.barplot(y="model", x="disp_cc", data=cars_df_6, color="b", orient="h")
    return (cars_df_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### horsepower (min-max normalized)
    """)
    return


@app.cell
def _(cars_df_6, sns):
    cars_df_7 = cars_df_6.sort_values("hp", ascending=False)
    sns.barplot(y="model", x="hp", data=cars_df_7, color="b", orient="h")
    return (cars_df_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Standardization (z-score)

    Next we'll convert to [standard scores](https://en.wikipedia.org/wiki/Standard_score).  This expresses each value in terms of its standard deviations from the mean, expressing where each model stands in relation to the others.

    We'll use pyjanitor's [transform_columns](../reference/functions.html#janitor.functions.transform_columns) to apply the standard score calculation, `(x - x.mean()) / x.std()`, to each value in each of the columns we're evaluating.
    """)
    return


@app.cell
def _(cars_df_7):
    cars_df_7.transform_columns(
        ["mpg", "disp_cc", "hp"], lambda x: (x - x.mean()) / x.std(), elementwise=False
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standardized mpg
    """)
    return


@app.cell
def _(cars_df_7, sns):
    cars_df_8 = cars_df_7.sort_values("mpg", ascending=False)
    sns.barplot(y="model", x="mpg", data=cars_df_8, color="b", orient="h")
    return (cars_df_8,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standardized displacement
    """)
    return


@app.cell
def _(cars_df_8, sns):
    cars_df_9 = cars_df_8.sort_values("disp_cc", ascending=False)
    sns.barplot(y="model", x="disp_cc", data=cars_df_9, color="b", orient="h")
    return (cars_df_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Standardized horsepower
    """)
    return


@app.cell
def _(cars_df_9, sns):
    cars_df_10 = cars_df_9.sort_values("hp", ascending=False)
    sns.barplot(y="model", x="hp", data=cars_df_10, color="b", orient="h")
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
