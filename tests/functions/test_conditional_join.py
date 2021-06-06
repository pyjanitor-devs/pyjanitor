import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
import janitor
from janitor import le_join
from dataclasses import dataclass


df1 = pd.DataFrame({'col_a': [1,2,3], 'col_b': ["A", "B", "C"]})
df3 = pd.DataFrame({'col_a': [0, 2, 3], 'col_c': ["Z", "X", "Y"]})
df_3 = df3.astype('category')
df2 = [1,2,3]
df_2 = pd.Series(df2)
@dataclass
class lte_join:
    left : str 
    right : str

def test_type_right():
    """Raise TypeError if wrong type is provided for `right`."""
    with pytest.raises(TypeError):
        df1.conditional_join(df2, le_join("col_a", "col_c"))


def test_right_unnamed_Series():
    """Raise ValueError if `right` is an unnamed Series."""
    with pytest.raises(ValueError):
        df1.conditional_join(df_2, le_join("col_a", "col_c"))

def test_wrong_condition_type():
    """Raise TypeError if wrong type is provided for condition."""
    with pytest.raises(TypeError):      
        df1.conditional_join(df3, lte_join("col_a", "col_c"))

def test_wrong_column_type():
    """Raise TypeError if wrong type is provided for the columns."""
    with pytest.raises(TypeError):      
        df1.conditional_join(df3, le_join(1, "col_c"))

def test_wrong_column_presence():
    """Raise ValueError if column is not found in the dataframe."""
    with pytest.raises(ValueError):      
        df1.conditional_join(df3, le_join("col_a", "col_b"))

def test_no_condition():
    """Raise ValueError if no condition is provided."""
    with pytest.raises(ValueError):      
        df1.conditional_join(df3)

def test_wrong_column_dtype():
    """
    Raise ValueError if dtypes of columns 
    is not one of numeric, date, or string.
    """
    with pytest.raises(ValueError):      
        df1.conditional_join(df_3, le_join("col_a", "col_c"))


# df1 = pd.DataFrame({'col_a': [1,2,3], 'col_b': ["A", "B", "C"]})
# # df2 = pd.DataFrame({'col_a': [0, 2, 3], 'col_c': ["Z", "X", "Y"]})
# df2 = pd.Series([1,2,3], name='ragnar').astype('category')


# print(df1.conditional_join(df2, le_join("col_a", "ragnar")))