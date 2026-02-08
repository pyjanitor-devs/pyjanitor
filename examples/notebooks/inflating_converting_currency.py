import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Inflating and Converting Currency
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Notice

    This notebook's section on `convert_currency` has been disabled, as `exchangeratesapi.io` has disabled pinging of its API without an API key.
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
    This notebook serves to show a brief and simple example of how to use the `convert_currency()` and `inflate_currency()` methods from pyjanitor's finance submodule.

    The data for this example notebook come from the [United States Department of Agriculture Economic Research Service](https://www.ers.usda.gov/data-products/food-expenditure-series/), and we are specifically going to download the data of nominal food and alcohol expenditures, with taxes and tips, for all purchasers.  The data set includes nominal expenditures for 1997-2018, and the expenditures are provided in **millions** of U.S. dollars for the year in the which the expenditures were made.  For example, the expenditure values for 1997 are in units of 1997 U.S. dollars, whereas expenditures for 2018 are in 2018 U.S. dollars.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Getting and Cleaning the Data
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    url = "https://www.ers.usda.gov/webdocs/DataFiles/50606/nominal_expenditures.csv?v=9289.4"
    # 1) Read in the data from .csv file
    # 2) Clean up the column names
    # 3) Remove any empty rows or columns
    # 4) Melt the dataframe (from wide to long) to obtain "tidy" format
    data = (
        pd.read_csv(
            url, usecols=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], nrows=22, thousands=","
        )
        .clean_names()
        .remove_empty()
        .melt(id_vars=["year"], var_name="store_type", value_name="expenditure")
    )
    data.head()
    return (data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use inflate_currency() to Inflate All Values to 2018$
    """)
    return


@app.cell
def _(data):
    # Use split-apply-combine strategy to obtain 2018$ values
    # Group the data frame by year
    grouped = data.groupby(["year"])
    # Apply the inflate_currency() method to each group
    # (Note that each group comes with a name; in this case,
    #  the name corresponds to the year)
    data_constant_dollar = grouped.apply(
        lambda x: x.inflate_currency(
            column_name="expenditure",
            country="USA",
            currency_year=int(x.name),
            to_year=2018,
            make_new_column=True,
        )
    )
    data_constant_dollar.head()
    return (data_constant_dollar,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot Time Series to Observe Currency Inflation
    """)
    return


@app.cell
def _(data_constant_dollar):
    # Plot time series of nominal and real (2018$) expenditures for grocery stores
    # Note that the 2018 values for both series should be equal
    (
        data_constant_dollar.loc[
            data_constant_dollar["store_type"].str.contains("grocery_stores"), :
        ]
        .set_index("year")
        .drop(columns="store_type")
        .plot()
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Use convert_currency() to Convert USD to British Pounds

    _Note: Disabled and commented out due to `exchangeratesapi.io` policies.
    We are working through the deprecation of the API._
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Plot Time Series to Observe Currency Conversion
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
