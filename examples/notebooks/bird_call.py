import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Processing Bird Call Data

    ## Background

    The following example was obtained by translating the R code from [TidyTuesday 2019-04-30](https://github.com/rfordatascience/tidytuesday/tree/47567cb80846739c8543d158c1f3ff226c7e5a5f/data/2019/2019-04-30)
    to Python using Pandas and PyJanitor. It provides a simple example of using pyjanitor for:
    - column renaming
    - column name cleaning
    - dataframe merging

    The data originates from a study of the effects of articifial light on bird behaviour. It is a subset of the original study for the Chicago area.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Citations

    This data set originates from the publication:

    >*Winger BM, Weeks BC, Farnsworth A, Jones AW, Hennen M, Willard DE (2019) Nocturnal flight-calling behaviour predicts vulnerability to artificial light in migratory birds. Proceedings of the Royal Society B 286(1900): 20190364.* https://doi.org/10.1098/rspb.2019.0364

    To reference only the data, please cite the Dryad data package:

    > *Winger BM, Weeks BC, Farnsworth A, Jones AW, Hennen M, Willard DE (2019) Data from: Nocturnal flight-calling behaviour predicts vulnerability to artificial light in migratory birds. Dryad Digital Repository.* https://doi.org/10.5061/dryad.8rr0498
    """)
    return


@app.cell
def _():
    import pandas as pd
    return (pd,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Get Raw Data

    Using pandas to import csv data.
    """)
    return


@app.cell
def _(pd):
    raw_birds = pd.read_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-04-30/raw/Chicago_collision_data.csv"
    )
    raw_call = pd.read_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-04-30/raw/bird_call.csv",
        sep=" ",
    )
    raw_light = pd.read_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-04-30/raw/Light_levels_dryad.csv"
    )
    return raw_birds, raw_call, raw_light


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Original DataFrames

    Taking a quick look at the three imported (raw) pandas dataframes.
    """)
    return


@app.cell
def _(raw_birds):
    raw_birds.head()
    return


@app.cell
def _(raw_call):
    raw_call.head()
    return


@app.cell
def _(raw_light):
    raw_light.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Cleaning Data Using Pyjanitor
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pyjanitor provides additional method calls to standard pandas dataframe objects. The *clean_names()* method is one example which removes whitespace and lowercases all column names.
    """)
    return


@app.cell
def _(raw_light):
    clean_light = raw_light.clean_names()
    return (clean_light,)


@app.cell
def _(clean_light):
    clean_light.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Pyjanitor champions the cleaning process using the **call chaining approach**. We use this here to provide multiple column renaming. As our dataframes have inconsistent column names we rename the columns in the raw_call dataframe.
    """)
    return


@app.cell
def _(raw_call):
    clean_call = (
        raw_call.rename_column(
            "Species", "Genus"
        ).rename_column(  # rename 'Species' column to 'Genus'
            "Family", "Species"
        )  # rename 'Family' columnto 'Species'
    )
    return (clean_call,)


@app.cell
def _(clean_call):
    clean_call.head()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can chain as many standard pandas commands as we like, along with any pyjanitor specific methods.
    """)
    return


@app.cell
def _(clean_call, raw_birds):
    clean_birds = (
        raw_birds
        # merge the raw_birds dataframe with clean_raw dataframe
        .merge(clean_call, how="left")
        .select_columns(
            [
                "Genus",
                "Species",
                "Date",
                "Locality",
                "Collisions",
                "Call",
                "Habitat",
                "Stratum",
            ]
        )  # include list of cols
        .clean_names()
        # rename 'collisions' column to 'family' in merged dataframe
        .rename_column("collisions", "family")
        .rename_column("call", "flight_call")
        # drop all rows which contain a NaN
        .dropna()
    )
    return (clean_birds,)


@app.cell
def _(clean_birds):
    clean_birds.head()
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
