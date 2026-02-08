import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Replacing Bad Values

    This is US wind turbine data. The numeric fields use -9999 as a null value for missing data.
    Using -9999 as a null value in numeric fields will cause big problems for any summary statistics like totals, means, etc,
    we should change that to something else, like np.NaN which Pandas sum and mean functions will automatically filter out.
    You can see that the means for before and after replacing -9999 with np.NaN are very different.
    You can use Janitor's find_replace to easily replace them.
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
    ## Load Wind Turbine Data
    """)
    return


@app.cell
def _(pd):
    wind = pd.read_csv(
        "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2018/2018-11-06/us_wind.csv"
    )
    wind.head()
    return (wind,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check Mean
    """)
    return


@app.cell
def _(wind):
    wind.t_hh.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The t_hh column appears to be affected by -9999 values.
    What are all the columns that are affected?
    """)
    return


@app.cell
def _(wind):
    [col for col in wind.columns if -9999 in wind[col].values]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note:
    When replacing the -9999 values you can make a copy of the dataframe to prevent making modifications to the original dataframe.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    At first glance, it looks like the mean is negative, but this is only because of the bad values (-9999.0) throughout the column. To get the right mean, we should replace them!## Replace Bad Values with NaNs
    """)
    return


@app.cell
def _(np, wind):
    mapping = {-9999.0: np.nan}
    wind2 = wind.find_replace(
        usgs_pr_id=mapping,
        p_tnum=mapping,
        p_cap=mapping,
        t_cap=mapping,
        t_hh=mapping,
        t_rd=mapping,
        t_rsa=mapping,
        t_ttlh=mapping,
    )
    wind2.head()
    return (wind2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Check the Mean (again)
    """)
    return


@app.cell
def _(wind2):
    wind2.t_hh.mean()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And, now that the bad values were replaced by NaNs (which the mean() method ignores), the calculated mean is correct!
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Alternate method
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If we look at the description of the data (see [README](https://github.com/rfordatascience/tidytuesday/blob/master/data/2018/2018-11-06/readme.md)) we can find descriptions for our data values, for example:

    - `p_year`: Year project became operational
    - `t_hh`: Turbine hub height (meters)
    - `xlong`: Longitude

    Using our knowledge of the data, this would give us bounds we could use for these values. For example, the oldest electric wind turbine was built in 1887 and this document was written in 2018, so $1887 \leq \text{p_year} \leq 2018$. Turbine hub height should be positive, and a value above 1 km would be silly, so $0 < \text{t_hh} < 1000$. These are wind turbines in the United States, so $-161.76 < \text{xlong} < -68.01$.

    (Note that the README actually gives us minima and maxima for the data, so we could get much tighter bounds from that.)

    To filter out potential bad values, we will use `update_where` to remove values outside these ranges.
    """)
    return


@app.cell
def _(np, wind):
    # Note that update_where mutates the original dataframe
    (
        wind.update_where(
            (wind["p_year"] < 1887) | (wind["p_year"] > 2018), "p_year", np.nan
        )
        .update_where((wind["t_hh"] <= 0) | (wind["t_hh"] >= 1000), "t_hh", np.nan)
        .update_where(
            (wind["xlong"] < -161.76) | (wind["xlong"] > -68.01), "xlong", np.nan
        )
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Confirming this produces the same result
    """)
    return


@app.cell
def _(wind):
    wind.t_hh.mean()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
