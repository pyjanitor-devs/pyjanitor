import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Processing International Teacher Data
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Background

    This sample data comes from the UNESCO Institute of Statistics and can be found at [tidytuesdays' github repo](https://github.com/rfordatascience/tidytuesday/tree/master/data/2019/2019-05-07). This subset of the the data collected by the UNESCO Institute of Statistics contains country-level data on the number of teachers, teacher-to-student ratios, and related figures.
    """)
    return


@app.cell
def _():
    import pandas as pd
    import pandas_flavor as pf

    dirty_csv = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-07/EDULIT_DS_06052019101747206.csv"
    dirty_df = pd.read_csv(dirty_csv)
    dirty_df.head()
    return dirty_df, pd, pf


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Dictionary

    Below is the data dictionary from the tidytuesday github repo.


    |variable      |class     |description |
    |:---|:---|:-----------|
    |edulit_ind    | character | Unique ID|
    |indicator     | character | Education level group ("Lower Secondary Education", "Primary Education", "Upper Secondary Education", "Pre-Primary Education", "Secondary Education", "Tertiary Education", "Post-Secondary Non-Tertiary Education")|
    |country_code  | character |  Country code |
    |country       | character | Country Full name|
    |year          | integer (date)    | Year |
    |student_ratio | double    |Student to teacher ratio (lower = fewer students/teacher)|
    |flag_codes    | character | Code to indicate some metadata about exceptions |
    |flags         | character | Metadata about exceptions |


    The indicator variable describles the education level for each observation. Let's evaluate the actual values of the Indicator column in the data.
    """)
    return


@app.cell
def _(dirty_df):
    # The set of unique values in the indicator column
    set(dirty_df.Indicator)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Notice that strings in the dataframe each contain "Pupil-teach ratio in" & "(headcount basis)". We don't need all of this text to analyze the data.

     need some custom functions to clean up the strings. We'll need a function that removes a substring, given a pattern, from values in columns. Another function that removes trailing and leading characters from a value in a column. And finally, a function to make the first letter in each string upper case.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Data Cleaning
    """)
    return


@app.cell
def _(pf):
    @pf.register_dataframe_method
    def str_remove(df, column_name: str, pat: str, *args, **kwargs):
        """Remove a substring, given its pattern from a string value, in a given column"""
        df[column_name] = df[column_name].str.replace(pat, "", *args, **kwargs)
        return df


    @pf.register_dataframe_method
    def str_trim(df, column_name: str, *args, **kwargs):
        """Remove trailing and leading characters, in a given column"""
        df[column_name] = df[column_name].str.strip(*args, **kwargs)
        return df


    @pf.register_dataframe_method
    def str_title(df, column_name: str, *args, **kwargs):
        """Make the first letter in each word upper case"""
        df[column_name] = df[column_name].str.title(*args, **kwargs)
        return df


    @pf.register_dataframe_method
    def drop_duplicated_column(df, column_name: str, column_order: int = 0):
        """Remove duplicated columns
        and retain only a column given its order.
        Order 0 is to remove the first column,
        Order 1 is to remove the second column, and etc"""

        cols = list(df.columns)
        col_indexes = [
            col_idx for col_idx, col_name in enumerate(cols) if col_name == column_name
        ]

        # given that a column could be duplicated, user could opt based on its order
        removed_col_idx = col_indexes[column_order]
        # get the column indexes without column that is being removed
        filtered_cols = [c_i for c_i, c_v in enumerate(cols) if c_i != removed_col_idx]
        return df.iloc[:, filtered_cols]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note in the next example how we are able to chain our manipulations of the data into one process without losing our ability to explain what we are doing. The is the preferred framework for using pyjanitor
    """)
    return


@app.cell
def _(dirty_df):
    py_clean_df = (
        dirty_df.clean_names()
        # modify string values
        .str_remove("indicator", "Pupil-teacher ratio in")
        .str_remove("indicator", "(headcount basis)")
        .str_remove("indicator", "\\(\\)")
        .str_trim("indicator")
        .str_trim("country")
        .str_title("indicator")
        # remove `time` column (which is duplicated). The second `time` is being removed
        .drop_duplicated_column("time", 1)
        # renaming columns
        .rename_column("location", "country_code")
        .rename_column("value", "student_ratio")
        .rename_column("time", "year")
    )

    py_clean_df.head()
    return (py_clean_df,)


@app.cell
def _(pd, py_clean_df):
    # ensure that the output from janitor is similar with the clean r's janitor
    r_clean_csv = "https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2019/2019-05-07/student_teacher_ratio.csv"
    r_clean_df = pd.read_csv(r_clean_csv)

    pd.testing.assert_frame_equal(r_clean_df, py_clean_df)
    return


@app.cell
def _():
    import marimo as mo
    return (mo,)


if __name__ == "__main__":
    app.run()
