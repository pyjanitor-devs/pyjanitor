import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Processing French train data

    ## Background
    The SNCF (National Society of French Railways) is France's national state-owned railway company. Founded in 1938, it operates the country's national rail traffic along with Monaco, including the TGV, France's high-speed rail network. This dataset covers 2015-2018 with many different train stations. The dataset primarily covers aggregate trip times, delay times, cause for delay, etc., for each station there are 27 columns in total. A TGV route map can be seen [here](https://en.wikipedia.org/wiki/TGV#/media/File:France_TGV.png).

    ## The data
    The source data set is available from the [SNCF](https://ressources.data.sncf.com/explore/dataset/regularite-mensuelle-tgv-aqst/information/). Check out this [visualization](https://twitter.com/noccaea/status/1095735292206739456) of it. This has been used in a [tidy tuesday](https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-02-26) previously. The [full data set](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-02-26/full_trains.csv) is available but we will work with a [subset](https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-02-26/small_trains.csv).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Preliminaries
    """)
    return


@app.cell
def _():
    from collections import Counter

    import pandas as pd
    import seaborn as sns

    # allow plots to appear directly in the notebook
    # '%matplotlib inline' command supported automatically in marimo
    return Counter, pd, sns


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Call chaining example
    First, we run all the methods using pyjanitor's preferred call chaining approach. This code updates the column names, removes any empty rows/columns, and drops some unneeded columns in a very readable manner.
    """)
    return


@app.cell
def _(pd):
    chained_df = (
        # ingest raw data
        pd.read_csv(
            "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-02-26/small_trains.csv"
        )
        # removes whitespace, punctuation/symbols, capitalization
        .clean_names()
        # removes entirely empty rows / columns
        .remove_empty()
        # renames 1 column
        .rename_column("num_late_at_departure", "num_departing_late")
        # drops 3 unnecessary columns
        .drop(columns=["service", "delay_cause", "delayed_number"])
        # add 2 new columns with a calculation
        .join_apply(
            lambda df: df.num_departing_late / df.total_num_trips, "prop_late_departures"
        )
        .join_apply(
            lambda df: df.num_arriving_late / df.total_num_trips, "prop_late_arrivals"
        )
    )

    chained_df.head(3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Step by step through the methods
    Now, we will import the French data again and then use the methods from the call chain one at a time. Our subset of the train data has over 32000 rows and 13 columns.
    """)
    return


@app.cell
def _(pd):
    df = pd.read_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-02-26/small_trains.csv"
    )

    df.shape
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Cleaning column names
    The clean_names method converts the column names to lowercase and replaces all spaces with underscores. For this data set, it actually does not modify any of the names.
    """)
    return


@app.cell
def _(df):
    original_columns = df.columns
    df_1 = df.clean_names()
    new_columns = df_1.columns
    original_columns == new_columns
    return df_1, new_columns


@app.cell
def _(new_columns):
    new_columns
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Renaming columns
    We rename the "num_late_at_departure" column for consistency purposes with the rename_column method.
    """)
    return


@app.cell
def _(df_1):
    df_2 = df_1.rename_column('num_late_at_departure', 'num_departing_late')
    df_2.columns
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Dropping empty columns and rows
    The remove_empty method looks for empty columns and rows and drops them if found.
    """)
    return


@app.cell
def _(df_2):
    df_2.shape
    return


@app.cell
def _(df_2):
    df_3 = df_2.remove_empty()
    df_3.shape
    return (df_3,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Drop unneeded columns
    We identify 3 columns that we decided are unnecessary for the analysis and can quickly drop them with the aptly named drop_columns method.
    """)
    return


@app.cell
def _(df_3):
    df_4 = df_3.drop(columns=['service', 'delay_cause', 'delayed_number'])
    return (df_4,)


@app.cell
def _(df_4):
    df_4.columns
    return


@app.cell
def _(Counter, df_4):
    # gives us the top ten departure stations from that column
    Counter(df_4['departure_station']).most_common(10)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We use seaborn to quickly visualize how quickly departure and arrivals times were late versus the total number of trips for each of the over 30000 routes in the database.
    """)
    return


@app.cell
def _(df_4, sns):
    sns.pairplot(df_4, x_vars=['num_departing_late', 'num_arriving_late'], y_vars='total_num_trips', height=7, aspect=0.7)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Add additional statistics as new columns
    We can add columns containing additional statistics concerning the proportion of time each route is late either departing or arriving by using the add_columns method for each route.

    Note the difference between how we added the two columns below and the same code in the chained_df file creation at the top of the notebook. In order to operate on the df that was in the process of being created in the call chain, we had to use join_apply with a lambda function instead of the add_columns method. Alternatively, we could have split the chain into two separate chains with the df being created in the first chain and the add_columns method being used in the second chain.
    """)
    return


@app.cell
def _(df_4):
    df_prop = df_4.add_columns(prop_late_departures=df_4.num_departing_late / df_4.total_num_trips, prop_late_arrivals=df_4.num_arriving_late / df_4.total_num_trips)
    df_prop.head(3)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
