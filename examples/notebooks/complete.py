import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Expose explicitly missing values with `complete`
    """)
    return


@app.cell
def _():
    import pandas as pd

    return (pd,)


@app.cell
def _(pd):
    # from http://imachordata.com/2016/02/05/you-complete-me/
    df = pd.DataFrame(
        {
            "Year": [1999, 2000, 2004, 1999, 2004],
            "Taxon": [
                "Saccharina",
                "Saccharina",
                "Saccharina",
                "Agarum",
                "Agarum",
            ],
            "Abundance": [4, 5, 2, 1, 8],
        }
    )

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that Year 2000 and Agarum pairing is missing in the DataFrame above. Let’s make it explicit:
    """)
    return


@app.cell
def _(df):
    df.complete("Year", "Taxon")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What if we wanted the explicit missing values for all the years from 1999 to 2004? Easy - simply pass a dictionary pairing the column name with the new values:
    """)
    return


@app.cell
def _(df):
    new_year_values = {"Year": range(df.Year.min(), df.Year.max() + 1)}

    df.complete(new_year_values, "Taxon", sort=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can pass a callable as values in the dictionary:
    """)
    return


@app.cell
def _(df):
    new_year_values_1 = lambda year: range(year.min(), year.max() + 1)  # noqa: E731
    df.complete({"Year": new_year_values_1}, "Taxon")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can get explicit rows, based only on existing data:
    """)
    return


@app.cell
def _(pd):
    # https://stackoverflow.com/q/62266057/7175713
    df_1 = {
        "Name": ("Bob", "Bob", "Emma"),
        "Age": (23, 23, 78),
        "Gender": ("Male", "Male", "Female"),
        "Item": ("house", "car", "house"),
        "Value": (5, 1, 3),
    }
    df_1 = pd.DataFrame(df_1)
    df_1
    return (df_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the DataFrame above, there is no `car` Item value for the `Name`, `Age`, `Gender`  combination -> `(Emma, 78, Female)`. Pass `(Name, Age, Gender)` and `Item` to explicitly expose the missing row:
    """)
    return


@app.cell
def _(df_1):
    df_1.complete(("Name", "Age", "Gender"), "Item")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The example above showed how to expose missing rows on a group basis. There is also the option of exposing missing rows with the `by` parameter:
    """)
    return


@app.cell
def _(pd):
    df_2 = pd.DataFrame(
        {
            "state": ["CA", "CA", "HI", "HI", "HI", "NY", "NY"],
            "year": [2010, 2013, 2010, 2012, 2016, 2009, 2013],
            "value": [1, 3, 1, 2, 3, 2, 5],
        }
    )
    df_2
    return (df_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's expose all the missing years, based on the minimum and maximum year, for each state:
    """)
    return


@app.cell
def _(df_2):
    new_year_values_2 = lambda year: range(year.min(), year.max() + 1)  # noqa: E731
    df_2.complete({"year": new_year_values_2}, by="state")
    return (new_year_values_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can fill the nulls with `fill_value`:
    """)
    return


@app.cell
def _(df_2, new_year_values_2):
    df_2.complete({"year": new_year_values_2}, by="state", fill_value=0)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
