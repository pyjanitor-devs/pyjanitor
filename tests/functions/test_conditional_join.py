import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
import janitor
from janitor import le_join, lt_join, ge_join, gt_join
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

df_multi = df1.copy()
df_multi.columns = [list('AB'), list('CD')]

multi_index_columns = [(df_multi, le_join(("A", "B"), "col_c"), df2), (df1, le_join("col_a", ("A", "B")), df_multi)]

@pytest.mark.parametrize("df_left,condition,df_right", multi_index_columns)
def test_multiIndex_columns(df_left,condition, df_right):
    """Raise ValueError if columns are MultiIndex."""
    with pytest.raises(ValueError):
        df_left.conditional_join(df_right, condition)

@pytest.fixture
def df_left():
    return pd.DataFrame([{'x': 'b', 'y': 1, 'v': 1},
 {'x': 'b', 'y': 3, 'v': 2},
 {'x': 'b', 'y': 6, 'v': 3},
 {'x': 'a', 'y': 1, 'v': 4},
 {'x': 'a', 'y': 3, 'v': 5},
 {'x': 'a', 'y': 6, 'v': 6},
 {'x': 'c', 'y': 1, 'v': 7},
 {'x': 'c', 'y': 3, 'v': 8},
 {'x': 'c', 'y': 6, 'v': 9}]
)

@pytest.fixture
def df_right():
    return pd.DataFrame([{'x': 'c', 'v': 8, 'foo': 4}, {'x': 'b', 'v': 7, 'foo': 2}])

# def test_less_than_join(df_left, df_right):
#     """Test output of less than join."""
#     result = df_left.conditional_join(df_right, le_join("y", "foo"))
#     pass
# df1 = pd.DataFrame({'col_a': [1,2,3], 'col_b': ["A", "B", "C"]})
# df2 = pd.DataFrame({'col_a': [0, 2, 3], 'col_c': ["Z", "X", "Y"]})
# df2 = pd.Series([1,2,3], name='ragnar').astype('category')

# df1 = pd.DataFrame(dict(col_a = [1,2,5,np.nan], col_b=pd.Series(['A','B','B','C'], dtype='string')))
# df2 = pd.DataFrame({'col_a': [2,0, 3,np.nan], 'col_c': pd.Series(["Z", "X", "Y","A"], dtype='string')})
# print(df1)
# print(df2)
# print(df1.dtypes)
# print(df2.dtypes)
# print(df1.conditional_join(df2, le_join("col_a", "col_a")))


df1 = pd.DataFrame([{'x': 'b', 'y': 1, 'v': 1},
 {'x': 'b', 'y': 3, 'v': 2},
 {'x': 'b', 'y': 6, 'v': 3},
 {'x': 'a', 'y': 1, 'v': 4},
 {'x': 'a', 'y': 3, 'v': 5},
 {'x': 'a', 'y': 6, 'v': 6},
 {'x': 'c', 'y': 1, 'v': 7},
 {'x': 'c', 'y': 3, 'v': 8},
 {'x': 'c', 'y': 6, 'v': 9}]
)

df2 = pd.DataFrame([{'x': 'c', 'v': 8, 'foo': 4}, {'x': 'b', 'v': 7, 'foo': 2}])

result = df1.conditional_join(df2, le_join("v", "v"))

print(df1, end = '\n')
print(df2, end='\n')
print(result)