import numpy as np
import pandas as pd
import pytest
import itertools
from pandas.testing import assert_frame_equal

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
        fill_value={"Abundance" : 0},
    )
    assert_frame_equal(result, output1)


# test for wrong column names
# test for wrong grouping pattern ... dictionary/set/dataframe...
# add more tests in parametrize
