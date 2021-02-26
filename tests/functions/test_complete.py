import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


# from http://imachordata.com/2016/02/05/you-complete-me/
@pytest.fixture
def df1():
    return pd.DataFrame(
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


def test_empty_column(df1):
    "Return dataframe if `columns` is empty."
    assert_frame_equal(df1.complete(), df1)


def test_MultiIndex_column(df1):
    "Raise ValueError if column is a MultiIndex."
    df = df1
    df.columns = [["A", "B", "C"], list(df.columns)]
    with pytest.raises(ValueError):
        df1.complete(["Year", "Taxon"])


def test_column_duplicated(df1):
    "Raise ValueError if column is duplicated in `columns`"
    with pytest.raises(ValueError):
        df1.complete(
            columns=[
                "Year",
                "Taxon",
                {"Year": lambda x: range(x.Year.min().x.Year.max() + 1)},
            ]
        )


def test_type_columns(df1):
    "Raise error if columns is not a list object."
    with pytest.raises(TypeError):
        df1.complete(columns="Year")


def test_fill_value_is_a_dict(df1):
    "Raise error if fill_value is not a dictionary"
    with pytest.raises(TypeError):
        df1.complete(columns=["Year", "Taxon"], fill_value=0)


def test_wrong_column_fill_value(df1):
    "Raise ValueError if column in `fill_value` does not exist."
    with pytest.raises(ValueError):
        df1.complete(columns=["Taxon", "Year"], fill_value={"year": 0})


frame = pd.DataFrame(
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
wrong_columns = (
    (frame, ["b", "Year"]),
    (frame, [{"Yayay": range(7)}]),
    (frame, ["Year", ["Abundant", "Taxon"]]),
    (frame, ["Year", ("Abundant", "Taxon")]),
)

empty_sub_columns = [
    (frame, ["Year", []]),
    (frame, ["Year", {}]),
    (frame, ["Year", ()]),
]


@pytest.mark.parametrize("frame,wrong_columns", wrong_columns)
def test_wrong_columns(frame, wrong_columns):
    """Test that ValueError is raised if wrong column is supplied."""
    with pytest.raises(ValueError):
        frame.complete(columns=wrong_columns)


@pytest.mark.parametrize("frame,empty_sub_cols", empty_sub_columns)
def test_empty_subcols(frame, empty_sub_cols):
    """Raise ValueError for an empty group in columns"""
    with pytest.raises(ValueError):
        frame.complete(columns=empty_sub_cols)


def test_fill_value(df1):
    """Test fill_value argument."""
    output1 = pd.DataFrame(
        {
            "Year": [1999, 1999, 2000, 2000, 2004, 2004],
            "Taxon": [
                "Agarum",
                "Saccharina",
                "Agarum",
                "Saccharina",
                "Agarum",
                "Saccharina",
            ],
            "Abundance": [1, 4.0, 0, 5, 8, 2],
        }
    )

    result = df1.complete(
        columns=["Year", "Taxon"], fill_value={"Abundance": 0}
    )
    assert_frame_equal(result, output1)


@pytest.fixture
def df1_output():
    return pd.DataFrame(
        {
            "Year": [
                1999,
                1999,
                2000,
                2000,
                2001,
                2001,
                2002,
                2002,
                2003,
                2003,
                2004,
                2004,
            ],
            "Taxon": [
                "Agarum",
                "Saccharina",
                "Agarum",
                "Saccharina",
                "Agarum",
                "Saccharina",
                "Agarum",
                "Saccharina",
                "Agarum",
                "Saccharina",
                "Agarum",
                "Saccharina",
            ],
            "Abundance": [1.0, 4, 0, 5, 0, 0, 0, 0, 0, 0, 8, 2],
        }
    )


def test_fill_value_all_years(df1, df1_output):
    """
    Test the complete function accurately replicates for
    all the years from 1999 to 2004.
    """

    result = df1.complete(
        columns=[
            {"Year": lambda x: range(x.Year.min(), x.Year.max() + 1)},
            "Taxon",
        ],
        fill_value={"Abundance": 0},
    )
    assert_frame_equal(result, df1_output)


def test_dict_series(df1, df1_output):
    """
    Test the complete function if a dictionary containing a Series
    is present in `columns`.
    """

    result = df1.complete(
        columns=[
            {
                "Year": lambda x: pd.Series(
                    range(x.Year.min(), x.Year.max() + 1)
                )
            },
            "Taxon",
        ],
        fill_value={"Abundance": 0},
    )
    assert_frame_equal(result, df1_output)


def test_dict_series_duplicates(df1, df1_output):
    """
    Test the complete function if a dictionary containing a
    Series (with duplicates) is present in `columns`.
    """

    result = df1.complete(
        columns=[
            {
                "Year": pd.Series(
                    [1999, 2000, 2000, 2001, 2002, 2002, 2002, 2003, 2004]
                )
            },
            "Taxon",
        ],
        fill_value={"Abundance": 0},
    )
    assert_frame_equal(result, df1_output)


# adapted from https://tidyr.tidyverse.org/reference/complete.html
complete_parameters = [
    (
        pd.DataFrame(
            {
                "group": [1, 2, 1],
                "item_id": [1, 2, 2],
                "item_name": ["a", "b", "b"],
                "value1": [1, 2, 3],
                "value2": [4, 5, 6],
            }
        ),
        ["group", "item_id", "item_name"],
        pd.DataFrame(
            {
                "group": [1, 1, 1, 1, 2, 2, 2, 2],
                "item_id": [1, 1, 2, 2, 1, 1, 2, 2],
                "item_name": ["a", "b", "a", "b", "a", "b", "a", "b"],
                "value1": [
                    1.0,
                    np.nan,
                    np.nan,
                    3.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    2.0,
                ],
                "value2": [
                    4.0,
                    np.nan,
                    np.nan,
                    6.0,
                    np.nan,
                    np.nan,
                    np.nan,
                    5.0,
                ],
            }
        ),
    ),
    (
        pd.DataFrame(
            {
                "group": [1, 2, 1],
                "item_id": [1, 2, 2],
                "item_name": ["a", "b", "b"],
                "value1": [1, 2, 3],
                "value2": [4, 5, 6],
            }
        ),
        ["group", ("item_id", "item_name")],
        pd.DataFrame(
            {
                "group": [1, 1, 2, 2],
                "item_id": [1, 2, 1, 2],
                "item_name": ["a", "b", "a", "b"],
                "value1": [1.0, 3.0, np.nan, 2.0],
                "value2": [4.0, 6.0, np.nan, 5.0],
            }
        ),
    ),
]


@pytest.mark.parametrize("df,columns,output", complete_parameters)
def test_complete(df, columns, output):
    "Test the complete function, with and without groupings."
    assert_frame_equal(df.complete(columns), output)


@pytest.fixture
def duplicates():
    return pd.DataFrame(
        {
            "row": [
                "21.08.2020",
                "21.08.2020",
                "21.08.2020",
                "21.08.2020",
                "22.08.2020",
                "22.08.2020",
                "22.08.2020",
                "22.08.2020",
            ],
            "column": ["A", "A", "B", "C", "A", "B", "B", "C"],
            "value": [43.0, 36, 36, 28, 16, 40, 34, 0],
        }
    )


# https://stackoverflow.com/questions/63541729/
# pandas-how-to-include-all-columns-for-all-rows-although-value-is-missing-in-a-d
# /63543164#63543164
def test_duplicates(duplicates):
    """Test that the complete function works for duplicate values."""
    df = pd.DataFrame(
        {
            "row": {
                0: "21.08.2020",
                1: "21.08.2020",
                2: "21.08.2020",
                3: "21.08.2020",
                4: "22.08.2020",
                5: "22.08.2020",
                6: "22.08.2020",
            },
            "column": {0: "A", 1: "A", 2: "B", 3: "C", 4: "A", 5: "B", 6: "B"},
            "value": {0: 43, 1: 36, 2: 36, 3: 28, 4: 16, 5: 40, 6: 34},
        }
    )

    result = df.complete(columns=["row", "column"], fill_value={"value": 0})

    assert_frame_equal(result, duplicates)


def test_unsorted_duplicates(duplicates):
    """Test output for unsorted duplicates."""

    df = pd.DataFrame(
        {
            "row": {
                0: "22.08.2020",
                1: "22.08.2020",
                2: "21.08.2020",
                3: "21.08.2020",
                4: "21.08.2020",
                5: "21.08.2020",
                6: "22.08.2020",
            },
            "column": {
                0: "B",
                1: "B",
                2: "A",
                3: "A",
                4: "B",
                5: "C",
                6: "A",
            },
            "value": {0: 40, 1: 34, 2: 43, 3: 36, 4: 36, 5: 28, 6: 16},
        }
    )

    result = df.complete(columns=["row", "column"], fill_value={"value": 0})

    assert_frame_equal(result, duplicates)


# https://stackoverflow.com/questions/32874239/
# how-do-i-use-tidyr-to-fill-in-completed-rows-within-each-value-of-a-grouping-var
def test_grouping_first_columns():
    """
    Test complete function when the first entry
    in columns is a grouping.
    """

    df2 = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "choice": [5, 6, 7],
            "c": [9.0, np.nan, 11.0],
            "d": [
                pd.NaT,
                pd.Timestamp("2015-09-30 00:00:00"),
                pd.Timestamp("2015-09-29 00:00:00"),
            ],
        }
    )
    output2 = pd.DataFrame(
        {
            "id": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "c": [9.0, 9.0, 9.0, np.nan, np.nan, np.nan, 11.0, 11.0, 11.0],
            "d": [
                pd.NaT,
                pd.NaT,
                pd.NaT,
                pd.Timestamp("2015-09-30 00:00:00"),
                pd.Timestamp("2015-09-30 00:00:00"),
                pd.Timestamp("2015-09-30 00:00:00"),
                pd.Timestamp("2015-09-29 00:00:00"),
                pd.Timestamp("2015-09-29 00:00:00"),
                pd.Timestamp("2015-09-29 00:00:00"),
            ],
            "choice": [5, 6, 7, 5, 6, 7, 5, 6, 7],
        }
    )
    result = df2.complete(columns=[("id", "c", "d"), "choice"])
    assert_frame_equal(result, output2)


# https://stackoverflow.com/questions/48914323/tidyr-complete-cases-nesting-misunderstanding
def test_complete_multiple_groupings():
    """Test that `complete` gets the correct output for multiple groupings."""
    df3 = pd.DataFrame(
        {
            "project_id": [1, 1, 1, 1, 2, 2, 2],
            "meta": ["A", "A", "B", "B", "A", "B", "C"],
            "domain1": ["d", "e", "h", "i", "d", "i", "k"],
            "question_count": [3, 3, 3, 3, 2, 2, 2],
            "tag_count": [2, 1, 3, 2, 1, 1, 2],
        }
    )

    output3 = pd.DataFrame(
        {
            "meta": ["A", "A", "A", "A", "B", "B", "B", "B", "C", "C"],
            "domain1": ["d", "d", "e", "e", "h", "h", "i", "i", "k", "k"],
            "project_id": [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],
            "question_count": [3, 2, 3, 2, 3, 2, 3, 2, 3, 2],
            "tag_count": [2.0, 1.0, 1.0, 0.0, 3.0, 0.0, 2.0, 1.0, 0.0, 2.0],
        }
    )

    result = df3.complete(
        columns=[("meta", "domain1"), ("project_id", "question_count")],
        fill_value={"tag_count": 0},
    )
    assert_frame_equal(result, output3)


@pytest.fixture
def output_dict_tuple():
    return pd.DataFrame(
        [
            {"Year": 1999, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 1999, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 1999, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 1999, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 1999, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2000, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2000, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2000, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2000, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2000, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2001, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2001, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2001, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2001, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2001, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2002, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2002, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2002, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2002, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2002, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2003, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2003, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2003, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2003, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2003, "Taxon": "Saccharina", "Abundance": 5},
            {"Year": 2004, "Taxon": "Agarum", "Abundance": 1},
            {"Year": 2004, "Taxon": "Agarum", "Abundance": 8},
            {"Year": 2004, "Taxon": "Saccharina", "Abundance": 2},
            {"Year": 2004, "Taxon": "Saccharina", "Abundance": 4},
            {"Year": 2004, "Taxon": "Saccharina", "Abundance": 5},
        ]
    )


def test_dict_tuple(df1, output_dict_tuple):
    """
    Test if a dictionary and a tuple/list
    are included in the `columns` parameter.
    """

    result = df1.complete(
        columns=[
            {"Year": lambda x: range(x.Year.min(), x.Year.max() + 1)},
            ("Taxon", "Abundance"),
        ]
    )

    assert_frame_equal(result, output_dict_tuple)


def test_complete_groupby():
    """Test output in the presence of a groupby."""
    df = pd.DataFrame(
        {
            "state": ["CA", "CA", "HI", "HI", "HI", "NY", "NY"],
            "year": [2010, 2013, 2010, 2012, 2016, 2009, 2013],
            "value": [1, 3, 1, 2, 3, 2, 5],
        }
    )

    result = df.complete(
        columns=[{"year": lambda x: range(x.year.min(), x.year.max() + 1)},],
        by="state",
    )

    expected = pd.DataFrame(
        [
            {"state": "CA", "year": 2010, "value": 1.0},
            {"state": "CA", "year": 2011, "value": np.nan},
            {"state": "CA", "year": 2012, "value": np.nan},
            {"state": "CA", "year": 2013, "value": 3.0},
            {"state": "HI", "year": 2010, "value": 1.0},
            {"state": "HI", "year": 2011, "value": np.nan},
            {"state": "HI", "year": 2012, "value": 2.0},
            {"state": "HI", "year": 2013, "value": np.nan},
            {"state": "HI", "year": 2014, "value": np.nan},
            {"state": "HI", "year": 2015, "value": np.nan},
            {"state": "HI", "year": 2016, "value": 3.0},
            {"state": "NY", "year": 2009, "value": 2.0},
            {"state": "NY", "year": 2010, "value": np.nan},
            {"state": "NY", "year": 2011, "value": np.nan},
            {"state": "NY", "year": 2012, "value": np.nan},
            {"state": "NY", "year": 2013, "value": 5.0},
        ]
    )

    assert_frame_equal(result, expected)


import janitor

df = pd.DataFrame(
    {
        "state": ["CA", "CA", "HI", "HI", "HI", "NY", "NY"],
        "year": [2010, 2013, 2010, 2012, 2016, 2009, 2013],
        "value": [1, 3, 1, 2, 3, 2, 5],
    }
)
result = df.complete(columns=[{"year": lambda df: np.arange(df.year.min(), df.year.max() + 1)}], by='state')

print(df, end="\n\n")
print(result)
