import numpy as np
import pandas as pd
import pytest
import itertools
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

list_of_columns = [
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
        [df], zip(list_of_columns, expected_output)
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
        list_of_columns=["Year", "Taxon"], fill_value={"Abundance": 0}
    )
    assert_frame_equal(result, output1)


def test_fill_value_all_years(df1):
    """Test the complete function accurately replicates for all the years
       from 1999 to 2004."""

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
        list_of_columns=[
            {"Year": range(df1.Year.min(), df1.Year.max() + 1)},
            "Taxon",
        ],
        fill_value={"Abundance": 0},
    )
    assert_frame_equal(result, output1)


def test_type_list_of_columns(df1):
    """Raise error if list_of_columns is not a list object.'"""
    with pytest.raises(TypeError):
        df1.complete(list_of_columns="Year")


def test_empty_list_of_columns(df1):
    """Raise error if list_of_columns is empty'"""
    with pytest.raises(ValueError):
        df1.complete(list_of_columns=[])


def test_wrong_column_names_string(df1):
    """Raise error if wrong column names is passed as strings'"""
    with pytest.raises(KeyError):
        df1.complete(list_of_columns=["b", "Year"])


def test_wrong_column_names_dict(df1):
    """Raise error if wrong column name is in dictionary in list_of_columns'"""
    with pytest.raises(KeyError):
        df1.complete(list_of_columns=[{"Yayay": range(7)}])


def test_wrong_column_names_sublist(df1):
    """Raise error if wrong column name is in list grouping in list_of_columns'"""
    with pytest.raises(KeyError):
        df1.complete(list_of_columns=["Year", ["Abundant", "Taxon"]])


def test_wrong_column_names_tuple(df1):
    """Raise error if wrong column name is in tuple grouping in list_of_columns'"""
    with pytest.raises(KeyError):
        df1.complete(list_of_columns=["Year", ("Abundant", "Taxon")])


def test_empty_sublist(df1):
    """Raise error for an empty sublist in list_of_columns'"""
    with pytest.raises(ValueError):
        df1.complete(list_of_columns=["Year", []])


def test_empty_dict(df1):
    """Raise error for any empty dictionary in list_of_columns'"""
    with pytest.raises(ValueError):
        df1.complete(list_of_columns=["Year", {}])


def test_empty_tuple(df1):
    """Raise error for any empty tuple in list_of_columns'"""
    with pytest.raises(ValueError):
        df1.complete(list_of_columns=["Year", ()])


def test_wrong_column_type(df1):
    """Raise error if entry in list_of_columns is not a string/list/tuple/dict'"""
    with pytest.raises(ValueError):
        df1.complete(list_of_columns=["Year", set()])


# https://stackoverflow.com/questions/32874239/how-do-i-use-tidyr-to-fill-in-completed-rows-within-each-value-of-a-grouping-var
def test_grouping_first_columns():
    """Test complete function when the first entry in list_of_columns is 
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
    result = df2.complete(list_of_columns=[("id", "c", "d"), "choice"])
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
            list_of_columns=[
                ("meta", "domain1"),
                ("project_id", "question_count"),
            ],
            fill_value={"tag_count": 0},
        )
    # this part is not necessary for the test
    # however, I wanted to match the result from Stack Overflow,
    # hence the extra steps
        .sort_values(["project_id", "meta"], ignore_index=True)
        .reindex(columns=df3.columns)
    )

    assert_frame_equal(result, output3)
