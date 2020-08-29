import itertools

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# from https://tidyr.tidyverse.org/reference/complete.html
df = pd.DataFrame(
    {
        "group": [1, 2, 1],
        "item_id": [1, 2, 2],
        "item_name": ["a", "b", "b"],
        "value1": [1, 2, 3],
        "value2": [4, 5, 6],
    }
)

columns = [
    ["group", "item_id", "item_name"],
    ["group", ("item_id", "item_name")],
]

expected_output = [
    pd.DataFrame(
        {
            "group": [1, 1, 1, 1, 2, 2, 2, 2],
            "item_id": [1, 1, 2, 2, 1, 1, 2, 2],
            "item_name": ["a", "b", "a", "b", "a", "b", "a", "b"],
            "value1": [1.0, np.nan, np.nan, 3.0, np.nan, np.nan, np.nan, 2.0],
            "value2": [4.0, np.nan, np.nan, 6.0, np.nan, np.nan, np.nan, 5.0],
        }
    ),
    pd.DataFrame(
        {
            "group": [1, 1, 2, 2],
            "item_id": [1, 2, 1, 2],
            "item_name": ["a", "b", "a", "b"],
            "value1": [1.0, 3.0, np.nan, 2.0],
            "value2": [4.0, 6.0, np.nan, 5.0],
        }
    ),
]
complete_parameters = [
    (dataframe, columns, output)
    for dataframe, (columns, output) in itertools.product(
        [df], zip(columns, expected_output)
    )
]


@pytest.mark.parametrize("df,columns,output", complete_parameters)
def test_complete(df, columns, output):
    """Test the complete function, with and without groupings."""
    assert_frame_equal(df.complete(columns), output)


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


def test_fill_value_all_years(df1):
    """
    Test the complete function accurately replicates for all the years
    from 1999 to 2004.
    """

    output1 = pd.DataFrame(
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

    result = df1.complete(
        columns=[
            {"Year": range(df1.Year.min(), df1.Year.max() + 1)},
            "Taxon",
        ],
        fill_value={"Abundance": 0},
    )
    assert_frame_equal(result, output1)


def test_type_columns(df1):
    """Raise error if columns is not a list object.'"""
    with pytest.raises(TypeError):
        df1.complete(columns="Year")


def test_empty_columns(df1):
    """Raise error if columns is empty'"""
    with pytest.raises(ValueError):
        df1.complete(columns=[])


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
    (frame, ["Year", set()]),
]


@pytest.mark.parametrize("frame,wrong_columns", wrong_columns)
def test_wrong_columns(frame, wrong_columns):
    """Test that KeyError is raised if wrong column is supplied."""
    with pytest.raises(KeyError):
        frame.complete(columns=wrong_columns)


@pytest.mark.parametrize("frame,empty_sub_cols", empty_sub_columns)
def test_empty_subcols(frame, empty_sub_cols):
    """Raise ValueError for an empty container in columns'"""
    with pytest.raises(ValueError):
        frame.complete(columns=empty_sub_cols)


# https://stackoverflow.com/questions/32874239/
# how-do-i-use-tidyr-to-fill-in-completed-rows-within-each-value-of-a-grouping-var
def test_grouping_first_columns():
    """Test complete function when the first entry in columns is
        a grouping."""

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
            "project_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
            "meta": ["A", "A", "B", "B", "C", "A", "A", "B", "B", "C"],
            "domain1": ["d", "e", "h", "i", "k", "d", "e", "h", "i", "k"],
            "question_count": [3, 3, 3, 3, 3, 2, 2, 2, 2, 2],
            "tag_count": [2.0, 1.0, 3.0, 2.0, 0.0, 1.0, 0.0, 0.0, 1.0, 2.0],
        }
    )

    result = (
        df3.complete(
            columns=[("meta", "domain1"), ("project_id", "question_count")],
            fill_value={"tag_count": 0},
        )
        # this part is not necessary for the test
        # however, I wanted to match the result from Stack Overflow,
        # hence the extra steps
        .sort_values(["project_id", "meta"], ignore_index=True).reindex(
            columns=df3.columns
        )
    )

    assert_frame_equal(result, output3)


# https://stackoverflow.com/questions/63541729/
# pandas-how-to-include-all-columns-for-all-rows-although-value-is-missing-in-a-d
# /63543164#63543164
def test_duplicate_index():
    """Test that the complete function works for duplicate index."""
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

    dup_expected_output = pd.DataFrame(
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

    result = df.complete(columns=["row", "column"], fill_value={"value": 0})

    assert_frame_equal(result, dup_expected_output)
