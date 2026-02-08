import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Pivot_Longer : One function to cover transformations from wide to long form.
    """)
    return


@app.cell
def _():
    import re

    import numpy as np
    import pandas as pd

    return np, pd, re


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Unpivoting(reshaping data from wide to long form) in Pandas is executed either through [pd.melt](https://pandas.pydata.org/docs/reference/api/pandas.melt.html), [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html), or [pd.DataFrame.stack](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.stack.html). However, there are scenarios where a few more steps are required to massage the data into the long form that we desire. Take the dataframe below, copied from [Stack Overflow](https://stackoverflow.com/questions/64061588/pandas-melt-multiple-columns-to-tabulate-a-dataset#64062002):
    """)
    return


@app.cell
def _(pd):
    df = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "M_start_date_1": [201709, 201709, 201709],
            "M_end_date_1": [201905, 201905, 201905],
            "M_start_date_2": [202004, 202004, 202004],
            "M_end_date_2": [202005, 202005, 202005],
            "F_start_date_1": [201803, 201803, 201803],
            "F_end_date_1": [201904, 201904, 201904],
            "F_start_date_2": [201912, 201912, 201912],
            "F_end_date_2": [202007, 202007, 202007],
        }
    )

    df
    return (df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is a [beautiful solution](https://stackoverflow.com/a/64062027/7175713), from Stack Overflow :
    """)
    return


@app.cell
def _(df):
    df1 = df.set_index("id")
    df1.columns = df1.columns.str.split("_", expand=True)
    df1 = (
        df1.stack(level=[0, 2, 3], future_stack=True)
        .sort_index(level=[0, 1], ascending=[True, False])
        .reset_index(level=[2, 3], drop=True)
        .sort_index(axis=1, ascending=False)
        .rename_axis(["id", "cod"])
        .reset_index()
    )

    df1
    return (df1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We propose an alternative, based on [pandas melt](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html) and [concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html), that abstracts the reshaping mechanism, allows the user to focus on the task, can be applied to other scenarios,  and is chainable :
    """)
    return


@app.cell
def _(df):
    df
    return


@app.cell
def _(df):
    result = df.pivot_longer(
        index="id",
        names_to=("cod", ".value", "date", "num"),
        names_sep="_",
        sort_by_appearance=True,
    ).drop(columns=["date", "num"])

    result
    return (result,)


@app.cell
def _(df1, result):
    _columns = ["id", "cod", "start", "end"]
    df1_1 = df1.sort_values(_columns, ignore_index=True)
    result_1 = result.sort_values(_columns, ignore_index=True)
    df1_1.equals(result_1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html#janitor.pivot_longer) is a combination of ideas from R's [tidyr](https://tidyr.tidyverse.org/reference/pivot_longer.html) and [data.table](https://rdatatable.gitlab.io/data.table/) and is built on the powerful pandas' [melt](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html) and [concat](https://pandas.pydata.org/docs/reference/api/pandas.concat.html) functions.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html#janitor.pivot_longer) can melt dataframes easily; It is just a wrapper around pandas' [melt](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html).

    [Source Data](https://pandas.pydata.org/pandas-docs/stable/user_guide/reshaping.html#reshaping-by-melt)
    """)
    return


@app.cell
def _(pd):
    index = pd.MultiIndex.from_tuples([("person", "A"), ("person", "B")])
    df_1 = pd.DataFrame(
        {
            "first": ["John", "Mary"],
            "last": ["Doe", "Bo"],
            "height": [5.5, 6.0],
            "weight": [130, 150],
        },
        index=index,
    )
    df_1
    return (df_1,)


@app.cell
def _(df_1):
    df_1.pivot_longer(index=["first", "last"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you want the data unpivoted in order of appearance, you can set `sort_by_appearance` to `True`:
    """)
    return


@app.cell
def _(df_1):
    df_1.pivot_longer(index=["first", "last"], sort_by_appearance=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you wish to reuse the original index, you can set `ignore_index` to `False`; note that the index labels will be repeated as necessary:
    """)
    return


@app.cell
def _(df_1):
    df_1.pivot_longer(index=["first", "last"], ignore_index=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can also unpivot MultiIndex columns, the same way you would with pandas' [melt](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html#pandas.melt):

    [Source Data](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html#pandas.melt)
    """)
    return


@app.cell
def _(pd):
    df_2 = pd.DataFrame(
        {
            "A": {0: "a", 1: "b", 2: "c"},
            "B": {0: 1, 1: 3, 2: 5},
            "C": {0: 2, 1: 4, 2: 6},
        }
    )
    df_2.columns = [list("ABC"), list("DEF")]
    df_2
    return (df_2,)


@app.cell
def _(df_2):
    df_2.pivot_longer(index=[("A", "D")], values_to="num")
    return


@app.cell
def _(df_2):
    df_2.pivot_longer(index=[("A", "D")], column_names=[("B", "E")])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    And just like [melt](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.melt.html#pandas.melt), you can unpivot on a specific level, with `column_level`:
    """)
    return


@app.cell
def _(df_2):
    df_2.pivot_longer(index="A", column_names="B", column_level=0)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that when unpivoting MultiIndex columns, you need to pass a list of tuples to the `index` or `column_names` parameters.


    Also, `names_sep` or `names_pattern` parameters (which we shall look at in subsequent examples) only work for single indexed columns.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    You can dynamically select columns, using regular expressions or some other form supported by [pyjanitor's](https://pyjanitor-devs.github.io/pyjanitor/) [select_columns](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.select_columns.html#janitor.select_columns), especially if it is a lot of column names, and you are *lazy* like me  😄
    """)
    return


@app.cell
def _(pd):
    url = (
        "https://raw.githubusercontent.com/tidyverse/tidyr/main/data-raw/billboard.csv"
    )
    df_3 = pd.read_csv(url)
    df_3
    return (df_3,)


@app.cell
def _(df_3, re):
    # unpivot all columns that start with 'wk'
    df_3.pivot_longer(column_names=re.compile("^(wk)"), names_to="week")
    return


@app.cell
def _(df_3):
    df_3.pivot_longer(column_names="wk*", names_to="week")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html#janitor.pivot_longer) can also unpivot paired columns.  In this regard, it is like pandas' [wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html), but with more flexibility and power. Let's look at an example from pandas' [wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html) docs :
    """)
    return


@app.cell
def _(pd):
    df_4 = pd.DataFrame(
        {
            "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
            "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
            "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
        }
    )
    df_4
    return (df_4,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the data above, the `height`(ht) is paired with `age`(numbers). [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html) can handle this easily:
    """)
    return


@app.cell
def _(df_4, pd):
    pd.wide_to_long(df_4, stubnames="ht", i=["famid", "birth"], j="age")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now let's see how [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) handles this:
    """)
    return


@app.cell
def _(df_4):
    df_4.pivot_longer(
        index=["famid", "birth"], names_to=(".value", "age"), names_pattern="(.+)(.)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The first observable difference is that [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) is method chainable, while [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html) is not. Now, let's learn more about the `.value` variable.


    When `.value` is used in `names_to`, a pairing is created between `names_to` and `names_pattern`. For the example above, we get this pairing :
    <br>
    <div align='center'> ".value" --> (.+), "age" --> (.) </div>

    This tells the [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) function to keep values associated with `.value`(`.+`) as the column name, while values not associated with `.value`, in this case, the numbers, will be collated under a new column `age`. Internally, pandas `str.extract` is used to get the capturing groups before reshaping. This level of abstraction, we believe, allows the user to focus on the task, and get things done faster.

    Note that if you want the data returned in order of appearance you can set `sort_by_appearance` to `True`:
    """)
    return


@app.cell
def _(df_4):
    df_4.pivot_longer(
        index=["famid", "birth"],
        names_to=(".value", "age"),
        names_pattern="(.+)(.)",
        sort_by_appearance=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note also that the values in the `age` column are of `object` dtype. You can change the dtype, using pandas' [astype](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.astype.html) method.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We've seen already that [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html) handles this already and very well, so why bother? Let's look at another scenario where [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html) would need a few more steps. [Source Data](https://community.rstudio.com/t/pivot-longer-on-multiple-column-sets-pairs/43958):
    """)
    return


@app.cell
def _(pd):
    df_5 = pd.DataFrame(
        {
            "off_loc": ["A", "B", "C", "D", "E", "F"],
            "pt_loc": ["G", "H", "I", "J", "K", "L"],
            "pt_lat": [
                100.07548220000001,
                75.191326,
                122.65134479999999,
                124.13553329999999,
                124.13553329999999,
                124.01028909999998,
            ],
            "off_lat": [
                121.271083,
                75.93845266,
                135.043791,
                134.51128400000002,
                134.484374,
                137.962195,
            ],
            "pt_long": [
                4.472089953,
                -144.387785,
                -40.45611048,
                -46.07156181,
                -46.07156181,
                -46.01594293,
            ],
            "off_long": [
                -7.188632000000001,
                -143.2288569,
                21.242563,
                40.937416999999996,
                40.78472,
                22.905889000000002,
            ],
        }
    )
    df_5
    return (df_5,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can unpivot with [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html) by first reorganising the columns :
    """)
    return


@app.cell
def _(df_5):
    df1_2 = df_5.copy()
    df1_2.columns = df1_2.columns.str.split("_").str[::-1].str.join("_")
    df1_2
    return (df1_2,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we can unpivot :
    """)
    return


@app.cell
def _(df1_2, pd):
    pd.wide_to_long(
        df1_2.reset_index(),
        stubnames=["loc", "lat", "long"],
        sep="_",
        i="index",
        j="set",
        suffix=".+",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can get the same transformed dataframe, with less lines, using [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html):
    """)
    return


@app.cell
def _(df_5):
    df_5.pivot_longer(names_to=["set", ".value"], names_pattern="(.+)_(.+)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again, the key here is the `.value` symbol. Pairing `names_to` with `names_pattern` and its results from [pd.str.extract](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.extract.html), we get :
    <br>
    <div align="center"> set--> (.+) --> [off, pt] and  .value--> (.+) --> [loc, lat, long] </div>
    <br>

    All values associated with `.value`(`loc, lat, long`) remain as column names, while values not associated with `.value`(`off, pt`) are lumped into a new column `set`.

    Notice that we did not have to reset the index - [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) takes care of that internally;  [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) allows you to focus on what you want, so you can get it and move on.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that the unpivoting could also have been executed with `names_sep`:
    """)
    return


@app.cell
def _(df_5):
    df_5.pivot_longer(
        names_to=["set", ".value"],
        names_sep="_",
        ignore_index=False,
        sort_by_appearance=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look at another example, from [Stack Overflow](https://stackoverflow.com/questions/45123924/convert-pandas-dataframe-from-wide-to-long/45124130) :
    """)
    return


@app.cell
def _(pd):
    df_6 = pd.DataFrame(
        [{"a_1": 2, "ab_1": 3, "ac_1": 4, "a_2": 5, "ab_2": 6, "ac_2": 7}]
    )
    df_6
    return (df_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The data above requires extracting `a`, `ab` and `ac` from `1` and `2`. This is another example of a paired column. We could solve this using [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html); in fact there is a very good solution from [Stack Overflow](https://stackoverflow.com/a/45124775/7175713)
    """)
    return


@app.cell
def _(df_6, pd):
    df1_3 = df_6.copy()
    df1_3["id"] = df1_3.index
    pd.wide_to_long(df1_3, ["a", "ab", "ac"], i="id", j="num", sep="_")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or you could simply pass the buck to [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html):
    """)
    return


@app.cell
def _(df_6):
    df_6.pivot_longer(names_to=(".value", "num"), names_sep="_")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the solution above, we used the `names_sep` argument, as it is more convenient. A few more examples to get you familiar with the `.value` symbol.

    [Source Data](https://stackoverflow.com/questions/55403008/pandas-partial-melt-or-group-melt)
    """)
    return


@app.cell
def _(pd):
    df_7 = pd.DataFrame(
        [[1, 1, 2, 3, 4, 5, 6], [2, 7, 8, 9, 10, 11, 12]],
        columns=["id", "ax", "ay", "az", "bx", "by", "bz"],
    )
    df_7
    return (df_7,)


@app.cell
def _(df_7):
    df_7.pivot_longer(index="id", names_to=("name", ".value"), names_pattern="(.)(.)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    For the code above `.value` is paired with `x`, `y`, `z`(which become the new column names), while `a`, `b` are unpivoted into the `name` column.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the dataframe below, we need to unpivot the data, keeping only the suffix `hi`, and pulling out the number between `A` and `g`.

    [Source Data](https://stackoverflow.com/questions/35929985/melt-a-data-table-with-a-column-pattern)
    """)
    return


@app.cell
def _(pd):
    df_8 = pd.DataFrame([{"id": 1, "A1g_hi": 2, "A2g_hi": 3, "A3g_hi": 4, "A4g_hi": 5}])
    df_8
    return (df_8,)


@app.cell
def _(df_8):
    df_8.pivot_longer(
        index="id", names_to=["time", ".value"], names_pattern="A(.)g_(.+)"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's see an example where we have multiple values in a paired column, and we wish to split them into separate columns. [Source Data](https://stackoverflow.com/questions/64107566/how-to-pivot-longer-and-populate-with-fields-from-column-names-at-the-same-tim?noredirect=1#comment113369419_64107566) :
    """)
    return


@app.cell
def _(pd):
    df_9 = pd.DataFrame(
        {
            "Sony | TV | Model | value": {0: "A222", 1: "A234", 2: "A4345"},
            "Sony | TV | Quantity | value": {0: 5, 1: 5, 2: 4},
            "Sony | TV | Max-quant | value": {0: 10, 1: 9, 2: 9},
            "Panasonic | TV | Model | value": {0: "T232", 1: "S3424", 2: "X3421"},
            "Panasonic | TV | Quantity | value": {0: 1, 1: 5, 2: 1},
            "Panasonic | TV | Max-quant | value": {0: 10, 1: 12, 2: 11},
            "Sanyo | Radio | Model | value": {0: "S111", 1: "S1s1", 2: "S1s2"},
            "Sanyo | Radio | Quantity | value": {0: 4, 1: 2, 2: 4},
            "Sanyo | Radio | Max-quant | value": {0: 9, 1: 9, 2: 10},
        }
    )
    df_9
    return (df_9,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The goal is to reshape the data into long format, with separate columns for `Manufacturer`(Sony,...), `Device`(TV, Radio), `Model`(S3424, ...), `maximum quantity` and `quantity`.

    Below is the [accepted solution](https://stackoverflow.com/a/64107688/7175713) on Stack Overflow :
    """)
    return


@app.cell
def _(df_9, pd):
    df1_4 = df_9.copy()
    # Create a multiIndex column header
    df1_4.columns = pd.MultiIndex.from_arrays(
        zip(*df1_4.columns.str.split("\\s?\\|\\s?"))
    )
    # Reshape the dataframe using
    # `set_index`, `droplevel`, and `stack`
    df1_4.stack([0, 1], future_stack=True).droplevel(1, axis=1).set_index(
        "Model", append=True
    ).rename_axis([None, "Manufacturer", "Device", "Model"]).sort_index(
        level=[1, 2, 3]
    ).reset_index().drop("level_0", axis=1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or, we could use [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html), along with `.value` in `names_to` and a regular expression in `names_pattern` :
    """)
    return


@app.cell
def _(df_9):
    df_9.pivot_longer(
        names_to=("Manufacturer", "Device", ".value"),
        names_pattern="(.+)\\|(.+)\\|(.+)\\|.*",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What if we are interested in unpivoting only a part of the entire dataframe?

    [Source Data](https://stackoverflow.com/questions/63044119/converting-wide-format-data-into-long-format-with-multiple-indices-and-grouped-d)
    """)
    return


@app.cell
def _(pd):
    df_10 = pd.DataFrame(
        {
            "time": [1, 2, 3],
            "factor": ["a", "a", "b"],
            "variable1": [0, 0, 0],
            "variable2": [0, 0, 1],
            "variable3": [0, 2, 0],
            "variable4": [2, 0, 1],
            "variable5": [1, 0, 1],
            "variable6": [0, 1, 1],
            "O1V1": [0, 0.2, -0.3],
            "O1V2": [0, 0.4, -0.9],
            "O1V3": [0.5, 0.2, -0.6],
            "O1V4": [0.5, 0.2, -0.6],
            "O1V5": [0, 0.2, -0.3],
            "O1V6": [0, 0.4, -0.9],
            "O1V7": [0.5, 0.2, -0.6],
            "O1V8": [0.5, 0.2, -0.6],
            "O2V1": [0, 0.5, 0.3],
            "O2V2": [0, 0.2, 0.9],
            "O2V3": [0.6, 0.1, -0.3],
            "O2V4": [0.5, 0.2, -0.6],
            "O2V5": [0, 0.5, 0.3],
            "O2V6": [0, 0.2, 0.9],
            "O2V7": [0.6, 0.1, -0.3],
            "O2V8": [0.5, 0.2, -0.6],
            "O3V1": [0, 0.7, 0.4],
            "O3V2": [0.9, 0.2, -0.3],
            "O3V3": [0.5, 0.2, -0.7],
            "O3V4": [0.5, 0.2, -0.6],
            "O3V5": [0, 0.7, 0.4],
            "O3V6": [0.9, 0.2, -0.3],
            "O3V7": [0.5, 0.2, -0.7],
            "O3V8": [0.5, 0.2, -0.6],
        }
    )
    df_10
    return (df_10,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What is the task? This is copied verbatim from the source:

    <blockquote>Each row of the data frame represents a time period. There are multiple 'subjects' being monitored, namely O1, O2, and O3. Each subject has 8 variables being measured. I need to convert this data into long format where each row contains the information for one subject at a given time period, but with only the first 4 subject variables, as well as the extra information about this time period in columns 2-4, but not columns 5-8.</blockquote>
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Below is the accepted solution, using [wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html):
    """)
    return


@app.cell
def _(df_10, pd):
    df1_5 = df_10.rename(
        columns={
            x: x[2:] + x[1:2] for x in df_10.columns[df_10.columns.str.startswith("O")]
        }
    )
    df1_5 = pd.wide_to_long(
        df1_5,
        i=["time", "factor"] + [f"variable{i}" for i in range(1, 7)],
        j="id",
        stubnames=[f"V{i}" for i in range(1, 9)],
        suffix=".*",
    )
    df1_5 = df1_5.reset_index().drop(
        columns=[f"V{i}" for i in range(5, 9)] + [f"variable{i}" for i in range(3, 7)]
    )
    df1_5
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can abstract the details and focus on the task with [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html):
    """)
    return


@app.cell
def _(df_10, re):
    df_10.pivot_longer(
        index=slice("time", "variable2"),
        column_names=re.compile(".+V[1-4]$"),
        names_to=("id", ".value"),
        names_pattern=".(.)(.+)$",
        sort_by_appearance=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One more example on the `.value` symbol for paired columns

    [Source Data](https://stackoverflow.com/questions/59477686/python-pandas-melt-single-column-into-two-seperate) :
    """)
    return


@app.cell
def _(pd):
    df_11 = pd.DataFrame({"id": [1, 2], "A_value": [50, 33], "D_value": [60, 45]})
    df_11
    return (df_11,)


@app.cell
def _(df_11):
    df_11.pivot_longer(index="id", names_to=("value_type", ".value"), names_sep="_")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    There are scenarios where we need to unpivot the data, and group values within the column names under new columns. The values in the columns will not become new column names, so we do not need the `.value` symbol. Let's see an example below:

    [Source Data](https://stackoverflow.com/questions/59550804/melt-column-by-substring-of-the-columns-name-in-pandas-python)
    """)
    return


@app.cell
def _(pd):
    df_12 = pd.DataFrame(
        {
            "subject": [1, 2],
            "A_target_word_gd": [1, 11],
            "A_target_word_fd": [2, 12],
            "B_target_word_gd": [3, 13],
            "B_target_word_fd": [4, 14],
            "subject_type": ["mild", "moderate"],
        }
    )
    df_12
    return (df_12,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the dataframe above, `A` and `B` represent conditions, while the suffixes `gd` and `fd` represent value types. We are not interested in the words in the middle (`_target_word`). We could solve it this way (this is the chosen solution, copied from [Stack Overflow](https://stackoverflow.com/a/59550967/7175713)) :
    """)
    return


@app.cell
def _(df_12, pd):
    new_df = pd.melt(
        df_12, id_vars=["subject_type", "subject"], var_name="abc"
    ).sort_values(by=["subject", "subject_type"])
    new_df["cond"] = new_df["abc"].apply(lambda x: x.split("_")[0])
    new_df["value_type"] = new_df.pop("abc").apply(lambda x: x.split("_")[-1])
    new_df
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or, we could just pass the buck to [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html):
    """)
    return


@app.cell
def _(df_12):
    df_12.pivot_longer(
        index=["subject", "subject_type"],
        names_to=("cond", "value_type"),
        names_pattern="([A-Z]).*(gd|fd)",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the code above, we pass in the new names of the columns to `names_to`('cond', 'value_type'), and pass the groups to be extracted as a regular expression to `names_pattern`.

    We could also reshape with `names_sep`:
    """)
    return


@app.cell
def _(df_12):
    df_12.pivot_longer(
        index=["subject", "subject_type"],
        names_to=("cond", "value_type"),
        names_sep="_target_word_",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here's another example where [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) abstracts the process and makes reshaping easy.


    In the dataframe below, we would like to unpivot the data and separate the column names into individual columns(`vault` should be in an `event` column, `2012` should be in a `year` column and `f` should be in a `gender` column).

    [Source Data](https://dcl-wrangle.stanford.edu/pivot-advanced.html)
    """)
    return


@app.cell
def _(pd):
    df_13 = pd.DataFrame(
        {
            "country": ["United States", "Russia", "China"],
            "vault_2012_f": [48.132, 46.366, 44.266],
            "vault_2012_m": [46.632, 46.866, 48.316],
            "vault_2016_f": [46.866, 45.733, 44.332],
            "vault_2016_m": [45.865, 46.033, 45.0],
            "floor_2012_f": [45.366, 41.599, 40.833],
            "floor_2012_m": [45.266, 45.308, 45.133],
            "floor_2016_f": [45.999, 42.032, 42.066],
            "floor_2016_m": [43.757, 44.766, 43.799],
        }
    )
    df_13
    return (df_13,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Here is one way to reshape the data:
    """)
    return


@app.cell
def _(df_13):
    reshape = df_13.set_index("country")
    reshape.columns = reshape.columns.str.split("_", expand=True)
    _columns = ["event", "year", "gender"]
    reshape.columns.names = _columns
    reshape.stack(level=_columns, future_stack=True).rename("score").reset_index(
        level=["country"] + _columns
    ).reset_index(drop=True)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is simplified with [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html):
    """)
    return


@app.cell
def _(df_13):
    df_13.pivot_longer(
        index="country",
        names_to=["event", "year", "gender"],
        names_sep="_",
        values_to="score",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Again, if you want the data returned in order of appearance, you can turn on the `sort_by_appearance` parameter:
    """)
    return


@app.cell
def _(df_13):
    df_13.pivot_longer(
        index="country",
        names_to=["event", "year", "gender"],
        names_sep="_",
        values_to="score",
        sort_by_appearance=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    One more feature that [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) offers is to pass a list of regular expressions to `names_pattern`. This comes in handy when one single regex cannot encapsulate similar columns for reshaping to long form. This idea is inspired by the [melt](https://rdatatable.gitlab.io/data.table/reference/melt.data.table.html) function in R's [data.table](https://rdatatable.gitlab.io/data.table/). A couple of examples should make this clear.

    [Source Data](https://stackoverflow.com/questions/61138600/tidy-dataset-with-pivot-longer-multiple-columns-into-two-columns)
    """)
    return


@app.cell
def _(pd):
    df_14 = pd.DataFrame(
        [
            {
                "title": "Avatar",
                "actor_1": "CCH_Pound…",
                "actor_2": "Joel_Davi…",
                "actor_3": "Wes_Studi",
                "actor_1_FB_likes": 1000,
                "actor_2_FB_likes": 936,
                "actor_3_FB_likes": 855,
            },
            {
                "title": "Pirates_of_the_Car…",
                "actor_1": "Johnny_De…",
                "actor_2": "Orlando_B…",
                "actor_3": "Jack_Daven…",
                "actor_1_FB_likes": 40000,
                "actor_2_FB_likes": 5000,
                "actor_3_FB_likes": 1000,
            },
            {
                "title": "The_Dark_Knight_Ri…",
                "actor_1": "Tom_Hardy",
                "actor_2": "Christian…",
                "actor_3": "Joseph_Gor…",
                "actor_1_FB_likes": 27000,
                "actor_2_FB_likes": 23000,
                "actor_3_FB_likes": 23000,
            },
            {
                "title": "John_Carter",
                "actor_1": "Daryl_Sab…",
                "actor_2": "Samantha_…",
                "actor_3": "Polly_Walk…",
                "actor_1_FB_likes": 640,
                "actor_2_FB_likes": 632,
                "actor_3_FB_likes": 530,
            },
            {
                "title": "Spider-Man_3",
                "actor_1": "J.K._Simm…",
                "actor_2": "James_Fra…",
                "actor_3": "Kirsten_Du…",
                "actor_1_FB_likes": 24000,
                "actor_2_FB_likes": 11000,
                "actor_3_FB_likes": 4000,
            },
            {
                "title": "Tangled",
                "actor_1": "Brad_Garr…",
                "actor_2": "Donna_Mur…",
                "actor_3": "M.C._Gainey",
                "actor_1_FB_likes": 799,
                "actor_2_FB_likes": 553,
                "actor_3_FB_likes": 284,
            },
        ]
    )
    df_14
    return (df_14,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Above, we have a dataframe of movie titles, actors, and their facebook likes. It would be great if we could transform this into a long form, with just the title, the actor names, and the number of likes. Let's look at a possible solution :

    First, we reshape the columns, so that the numbers appear at the end.
    """)
    return


@app.cell
def _(df_14, re):
    df1_6 = df_14.set_index("title")
    _header = [re.split("(_?\\d)", column) for column in df1_6]
    df1_6.columns = [f"{first}{last}{middle}" for first, middle, last in _header]
    df1_6
    return (df1_6,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we can reshape, using [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html) :
    """)
    return


@app.cell
def _(df1_6, pd):
    pd.wide_to_long(
        df1_6.reset_index(),
        stubnames=["actor", "actor_FB_likes"],
        i="title",
        j="group",
        sep="_",
    ).rename(columns={"actor_FB_likes": "num_likes"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We can achieve this by using `.value` multiple times, and then renaming the columns:
    """)
    return


@app.cell
def _(df_14):
    df_14.pivot_longer(
        index="title", names_to=(".value", ".value"), names_pattern="(.+)_\\d(.*)"
    ).rename(columns={"actor_FB_likes": "num_likes"})
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    What if we could just get our data in long form without the massaging? We know our data has a pattern to it --> it either ends in a number or *likes*.  Can't we take advantage of that? Yes, we can (I know, I know; it sounds like a campaign slogan 🤪)
    """)
    return


@app.cell
def _(df_14):
    df_14.pivot_longer(
        index="title", names_to=("actor", "num_likes"), names_pattern=("\\d$", "likes$")
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    A pairing of `names_to` and `names_pattern` results in:

        {"actor": '\d$', "num_likes": 'likes$'}

    The first regex looks for columns that end with a number, while the other looks for columns that end with *likes*. [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) will then look for columns that end with a number and lump all the values in those columns under the `actor` column, and also look for columns that end with *like* and combine all the values in those columns into a new column -> `num_likes`. Underneath the hood, [numpy select](https://numpy.org/doc/stable/reference/generated/numpy.select.html) and [pd.Series.str.contains](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.str.contains.html) are used to pull apart the columns into the new columns.

    Again, it is about the goal; we are not interested in the numbers (1,2,3), we only need the names of the actors, and their facebook likes. [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) aims to give as much flexibility as possible, in addition to ease of use, to allow the end user focus on the task.

    Let's take a look at another example. [Source Data](https://stackoverflow.com/questions/60439749/pair-wise-melt-in-pandas-dataframe) :
    """)
    return


@app.cell
def _(np, pd):
    df_15 = pd.DataFrame(
        {
            "id": [0, 1],
            "Name": ["ABC", "XYZ"],
            "code": [1, 2],
            "code1": [4, np.nan],
            "code2": ["8", 5],
            "type": ["S", "R"],
            "type1": ["E", np.nan],
            "type2": ["T", "U"],
        }
    )
    df_15
    return (df_15,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    We cannot directly use [pd.wide_to_long](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html) here without some massaging, as there is no definite suffix(the first `code` does not have a suffix), neither can we use `.value` here, again because there is no suffix. However, we can see a pattern where some columns start with `code`, and others start with `type`. Let's see how [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) solves this, using a sequence of regular expressions in the `names_pattern` argument :
    """)
    return


@app.cell
def _(df_15):
    df_15.pivot_longer(
        index=["id", "Name"],
        names_to=("code_all", "type_all"),
        names_pattern=("^code", "^type"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The key here is passing the right regular expression, and ensuring the names in `names_to` is paired with the right regex in `names_pattern`; as such, every column that starts with `code` will be included in the new `code_all` column; the same happens to the `type_all` column. Easy and flexible, right?

    Let's explore another example, from [Stack Overflow](https://stackoverflow.com/questions/12466493/reshaping-multiple-sets-of-measurement-columns-wide-format-into-single-columns) :
    """)
    return


@app.cell
def _(pd):
    df_16 = pd.DataFrame(
        [
            {
                "ID": 1,
                "DateRange1Start": "1/1/90",
                "DateRange1End": "3/1/90",
                "Value1": 4.4,
                "DateRange2Start": "4/5/91",
                "DateRange2End": "6/7/91",
                "Value2": 6.2,
                "DateRange3Start": "5/5/95",
                "DateRange3End": "6/6/96",
                "Value3": 3.3,
            }
        ]
    )
    df_16
    return (df_16,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    In the dataframe above, we need to reshape the data to have a start date, end date and value. For the `DateRange` columns, the numbers are embedded within the string, while for `value` it is appended at the end. One possible solution is to reshape the columns so that the numbers are at the end :
    """)
    return


@app.cell
def _(df_16, re):
    df1_7 = df_16.set_index("ID")
    _header = [re.split("(\\d)", column) for column in df1_7]
    df1_7.columns = [f"{first}{last}{middle}" for first, middle, last in _header]
    df1_7
    return (df1_7,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Now, we can unpivot:
    """)
    return


@app.cell
def _(df1_7, pd):
    pd.wide_to_long(
        df1_7.reset_index(),
        stubnames=["DateRangeStart", "DateRangeEnd", "Value"],
        i="ID",
        j="num",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or, we could allow pivot_longer worry about the massaging; simply pass to `names_pattern` a list of regular expressions that match what we are after :
    """)
    return


@app.cell
def _(df_16):
    df_16.pivot_longer(
        index="ID",
        names_to=("DateRangeStart", "DateRangeEnd", "Value"),
        names_pattern=("Start$", "End$", "^Value"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The code above looks for columns that end with *Start*(`Start$`), aggregates all the values in those columns into `DateRangeStart` column, looks for columns that end with *End*(`End$`), aggregates all the values within those columns into `DateRangeEnd` column, and finally looks for columns that start with *Value*(`^Value`), and aggregates the values in those columns into the `Value` column. Just know the patterns, and pair them accordingly. Again, the goal is a focus on the task, to make it simple for the end user.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Note that for the example above, you can use multiple `.value` to get the data out:
    """)
    return


@app.cell
def _(df_16):
    df_16.pivot_longer("ID", names_to=(".value", ".value"), names_pattern="(.+)\\d(.*)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Let's look at another example [Source Data](https://stackoverflow.com/questions/64316129/how-to-efficiently-melt-multiple-columns-using-the-module-melt-in-pandas/64316306#64316306) :
    """)
    return


@app.cell
def _(pd):
    df_17 = pd.DataFrame(
        {
            "Activity": ["P1", "P2"],
            "General": ["AA", "BB"],
            "m1": ["A1", "B1"],
            "t1": ["TA1", "TB1"],
            "m2": ["A2", "B2"],
            "t2": ["TA2", "TB2"],
            "m3": ["A3", "B3"],
            "t3": ["TA3", "TB3"],
        }
    )
    df_17
    return (df_17,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This is a [solution](https://stackoverflow.com/a/64316306/7175713) provided by yours truly :
    """)
    return


@app.cell
def _(df_17, pd):
    pd.wide_to_long(
        df_17, i=["Activity", "General"], stubnames=["t", "m"], j="number"
    ).set_axis(["Task", "M"], axis="columns").droplevel(-1).reset_index()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Or, we could use [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html), abstract the details, and focus on the task :
    """)
    return


@app.cell
def _(df_17):
    df_17.pivot_longer(
        index=["Activity", "General"],
        names_pattern=["^m", "^t"],
        names_to=["M", "Task"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alright, one last example :


    [Source Data](https://stackoverflow.com/questions/64159054/how-do-you-pivot-longer-columns-in-groups)
    """)
    return


@app.cell
def _(pd):
    df_18 = pd.DataFrame(
        {
            "Name": ["John", "Chris", "Alex"],
            "activity1": ["Birthday", "Sleep Over", "Track Race"],
            "number_activity_1": [1, 2, 4],
            "attendees1": [14, 18, 100],
            "activity2": ["Sleep Over", "Painting", "Birthday"],
            "number_activity_2": [4, 5, 1],
            "attendees2": [10, 8, 5],
        }
    )
    df_18
    return (df_18,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    The task here is to unpivot the data, and group the data under three new columns ("activity", "number_activity", and "attendees").

    We can see that there is a pattern to the data; let's create a list of regular expressions that match the patterns and pass to `names_pattern``:
    """)
    return


@app.cell
def _(df_18):
    df_18.pivot_longer(
        index="Name",
        names_to=("activity", "number_activity", "attendees"),
        names_pattern=("^activity", "^number_activity", "^attendees"),
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Alright, let's look at one final example:


    [Source Data](https://stackoverflow.com/questions/60387077/reshaping-and-melting-dataframe-whilst-picking-up-certain-regex)
    """)
    return


@app.cell
def _(pd):
    df_19 = pd.DataFrame(
        {
            "Location": ["Madrid", "Madrid", "Rome", "Rome"],
            "Account": ["ABC", "XYX", "ABC", "XYX"],
            "Y2019:MTD:January:Expense": [4354, 769867, 434654, 632556456],
            "Y2019:MTD:January:Income": [56456, 32556456, 5214, 46724423],
            "Y2019:MTD:February:Expense": [235423, 6785423, 235423, 46588],
        }
    )
    df_19
    return (df_19,)


@app.cell
def _(df_19):
    df_19.pivot_longer(
        index=["Location", "Account"],
        names_to=("year", "month", ".value"),
        names_pattern="Y(.+):MTD:(.{3}).+(Income|Expense)",
        sort_by_appearance=True,
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    [pivot_longer](https://pyjanitor-devs.github.io/pyjanitor/reference/janitor.functions/janitor.pivot_longer.html) does not solve all problems; no function does. Its aim is to make it easy to unpivot dataframes from wide to long form, while offering a lot of flexibility and power.
    """)
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
