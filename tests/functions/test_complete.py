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

#test for wrong column names
#test for fill value
#test for wrong grouping pattern ... dictionary/set/dataframe...
#add more tests in parametrize

