import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Processing Board Game Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Background

    This dataset comes from the [Board Game Geek database](http://boardgamegeek.com/). The site's database has more than 90,000 games, with crowd-sourced ratings. This particular subset is limited to only games with at least 50 ratings which were published between 1950 and 2016. This still leaves us with 10,532 games! For more information please check out the [tidytuesday repo](https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-03-12) which is where this example was taken from.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Cleaning
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### One-Shot
    This cell demonstrates the cleaning process using the call chaining approach championed in pyjanitor
    """)
    return


@app.cell
def _(pd):
    cleaned_df = (
        # ingest raw data
        pd.read_csv(
            "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-03-12//board_games.csv"
        )
        # removes whitespace, punctuation/symbols, capitalization
        .clean_names()
        # removes entirely empty rows / columns
        .remove_empty()
        # drops unnecessary columns
        .drop(columns=["image", "thumbnail", "compilation", "game_id"])
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Multi-Step
    These cells repeat the process in a step-by-step manner in order to explain it in more detail
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Read in the csv
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-03-12/board_games.csv"
    )
    df.head(3)
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove the whitespace, punctuation/symbols, and capitalization  form columns
    """)
    return


@app.cell
def _(df):
    df_1 = df.clean_names()
    df_1.head(3)
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Remove all the empty rows and columns if present
    """)
    return


@app.cell
def _(df_1):
    df_2 = df_1.remove_empty()
    df_2.head(3)
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check to see whether "min_playtime" and "max_playtime" columns are equal
    """)
    return


@app.cell
def _(df_2):
    len(df_2[df_2["min_playtime"] != df_2["max_playtime"]])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Check to see what percentage of the values in the "compilation" column are not null
    """)
    return


@app.cell
def _(df_2):
    len(df_2[df_2["compilation"].notnull()]) / len(df_2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Drop unnecessary columns
    The 'compilation' column was demonstrated to have little value, the "image" and "thumbnail" columns
    link to images and are not a factor in this analysis. The "game_id" column can be replaced by using the index.
    """)
    return


@app.cell
def _(df_2):
    df_3 = df_2.drop(columns=["image", "thumbnail", "compilation", "game_id"])
    df_3.head(3)
    return (df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Sample Analysis
    """)
    return


@app.cell
def _():
    # allow plots to appear directly in the notebook
    # '%matplotlib inline' command supported automatically in marimo
    import seaborn as sns

    return (sns,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What Categories appear most often?
    """)
    return


@app.cell
def _(df_3):
    df_3["category"].value_counts().head(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### What is the relationship between games' player numbers, recommended minimum age, and the game's estimated length?
    """)
    return


@app.cell
def _(df_3, sns):
    sns.pairplot(
        df_3,
        x_vars=["min_age", "min_players", "min_playtime"],
        y_vars="users_rated",
        height=7,
        aspect=0.7,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Preliminary analysis
    Without digging into the data too much more it becomes apparent that there are some entries that were improperly entered e.g. having a minimum playtime of 60000 minutes. Otherwise we see some nice bell curves.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
