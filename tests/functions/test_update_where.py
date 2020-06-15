import pandas as pd
import numpy as np
import pytest
from pandas._testing import assert_frame_equal

from janitor.functions import update_where

# @pytest.mark.functions
# def test_update_where(dataframe):
# not sure what is going on here @zbarry
# would love to learn how this works
# if you could explain it to me

# @pytest.mark.functions
# def test_update_where(dataframe):
#   """
#   Test that it accepts conditional parameters
#    """
#    pd.testing.assert_frame_equal(
#        dataframe.update_where(
#            (dataframe["decorated-elephant"] == 1)
#            & (dataframe["animals@#$%^"] == "rabbit"),
#            "cities",
#            "Durham",
#        ),
#        dataframe.replace("Cambridge", "Durham"),
#  )


def test_update_where():
    """
    Test that function works with expression
    """
    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [5, 6, 7, 8],
        "c": [0, 0, 0, 0]
    })
    expected = pd.DataFrame({
        'a': [1, 2, 3, 4],
        'b': [5, 6, 7, 8],
        'c': [0, 0, 10, 0]
    })
    result = (df.update_where(conditions="a > 2 and b < 8",
                              target_column_name='c',
                              target_val=10))

    assert_frame_equal(result, expected)


def test_update_where_1():
    """
    Raise an error if the conditions argument
    is not a string
    """

    df = pd.DataFrame({
        "a": [1, 2, 3, 4],
        "b": [5, 6, 7, 8],
        "c": [0, 0, 0, 0]
    })

    with pytest.raises(TypeError):
        update_where(df=df,
                     conditions=(df['a'] > 2) & (df['b'] < 8),
                     target_column_name='c',
                     target_val=10)


def test_update_where_2():
    """
    Test with null entries
    """

    df = pd.DataFrame({
        'A': [1, 2, 3, 4, np.nan],
        'B': range(10, 0, -2),
        'C': range(10, 5, -1)
    })

    expected = pd.DataFrame({
        'A': [1.0, 2.0, 3.0, 4.0, np.nan],
        'B': [10, 8, 6, 4, 2],
        'C': [10, 9, 8, 7, 10]
    })

    #set A not equal to A, since NaN != NaN
    #this excludes the null rows
    result = (df.update_where(conditions='A!=A and B==2',
                              target_column_name='C',
                              target_val=10))

    assert_frame_equal(result, expected)
