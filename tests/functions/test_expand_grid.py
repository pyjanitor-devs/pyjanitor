import pandas as pd
import pandas_flavor as pf
import pytest
import numpy as np
from itertools import chain
from functools import reduce
from numpy.lib import recfunctions as rfn
from string import ascii_lowercase, ascii_uppercase
from pandas._testing import assert_frame_equal
from pandas._testing import assert_series_equal
from typing import Dict


@pf.register_dataframe_method
def expand_grid(df: pd.DataFrame = None,
                df_key: str = None,
                others: Dict = None):
    #check if others is a dictionary
    if not isinstance(others, dict):
        #strictly name value pairs
        #same idea as in R and tidyverse implementation
        #probably take this out,
        #as it is covered in expand_grid function
        raise ValueError("others must be a dictionary")
    #if there is a dataframe, for the method chaining,
    #it must have a key, to create a name value pair
    if df is not None:
        if not df_key:
            raise ValueError("dataframe requires a name")
        else:
            others.update({df_key: df})
    dfs, dicts = check_instance(others)

    return grid_computation(dfs, dicts)


def check_instance(entry):

    if not entry:
        raise ValueError("passed dictionary cannot be empty")
    #if it is a number, convert to list
    #as numbers are not iterable
    entry = {
        key: [value] if isinstance(value,
                                   (type(None), int, float, bool)) else value
        for key, value in entry.items()
    }

    #convert to list if value is a string or a set or a tuple
    #just the safety that a list brings in
    #if a string is supplied and not within a list
    #it is expected that the string will be chunked into individual letters
    #and iterated through
    entry = {
        key: list(value) if isinstance(value,
                                       (str, set, tuple, range)) else value
        for key, value in entry.items()
    }

    #collect dataframes here
    dfs = []

    #collect non dataframes here, proper dicts ... key value pair where the value is a list of scalars
    dicts = {}

    for key, value in entry.items():

        #exclude dicts:
        if isinstance(value, dict):
            raise ValueError("nested dicts not allowed")

        #process arrays
        if isinstance(value, np.ndarray):
            if value.size == 0:
                raise ValueError("array cannot be empty")
            elif value.ndim == 1:
                dfs.append(pd.DataFrame(value, columns=[key]))
            elif value.ndim == 2:
                dfs.append(pd.DataFrame(value).add_prefix(f"{key}_"))
            else:
                raise ValueError(
                    "expand_grid works with only vector and matrix arrays")
        #process series
        if isinstance(value, pd.Series):
            if value.empty:
                raise ValueError("passed Series cannot be empty")
            if not isinstance(value.index, pd.MultiIndex):
                if value.name:
                    value = value.to_frame(name=f"{key}_{value.name}")
                    dfs.append(value)
                else:
                    value = value.to_frame(name=f"{key}")
                    dfs.append(value)
            else:
                raise ValueError(
                    "expand_grid does not work with pd.MultiIndex")
        #process dataframe
        if isinstance(value, pd.DataFrame):
            if value.empty:
                raise ValueError("passed DataFrame cannot be empty")
            if not (isinstance(value.index, pd.MultiIndex)
                    or isinstance(value.columns, pd.MultiIndex)):
                #add key to dataframe columns
                value = value.add_prefix(f"{key}_")
                dfs.append(value)
            else:
                raise ValueError(
                    "expand grid does not work with pd.MultiIndex")
        #process lists
        if isinstance(value, list):
            if not value:
                raise ValueError("passed value cannot be empty")
            elif np.array(value).ndim == 1:
                checklist = (type(None), str, int, float, bool)
                check = (isinstance(internal, checklist) for internal in value)
                if all(check):
                    dicts.update({key: value})
                else:
                    raise ValueError("values in iterable must be scalar")
            elif np.array(value).ndim == 2:
                value = pd.DataFrame(value).add_prefix(f"{key}_")
                dfs.append(value)
            else:
                raise ValueError("sequence's dimension should be 1d or 2d")

    return dfs, dicts


#computation for values that are not arrays/dataframes/series
#these are collected into dictionary and processed with numpy meshgrid
def grid_computation_dict(dicts):
    #actual computation
    if len(dicts) == 1:
        key = list(dicts.keys())[0]
        value = list(dicts.values())[0]
        final = pd.DataFrame(value, columns=[key])
    else:
        res = np.meshgrid(*dicts.values())
        #create structured array
        #keeps data type of each value in the dict
        outcome = np.core.records.fromarrays(res, names=",".join(dicts))
        #reshape into a 1 column array
        #using the size of any of the arrays obtained from the meshgrid computation
        outcome = np.reshape(outcome, (np.size(res[0]), 1))
        #flatten structured array into 1d array
        outcome = np.concatenate(outcome)
        #sort array
        outcome.sort(axis=0, order=list(dicts))
        #create dataframe
        final = pd.DataFrame.from_records(outcome)
    return final


#this is for dataframes/series
#this should be the final output if there are lists or lists and dicts
#returned from check instance
def compute_two_dfs(df1, df2):
    #get lengths of dataframes(number of rows) and swap
    # essentially we'll pair one dataframe with the other's length:
    lengths = reversed([ent.index.size for ent in (df1, df2)])
    #grab the maximum string length
    string_cols = [
        frame.select_dtypes(include="object").columns for frame in (df1, df2)
    ]

    #pair max string length with col
    #will be passed into frame.to_records, to get dtype in numpy recarray
    string_cols = [{col: f"<U{frame[col].str.len().max()}"
                    for col in ent}
                   for ent, frame in zip(string_cols, (df1, df2))]

    (len_first, col_dtypes,
     first), (len_last, col_dtypes,
              last) = list(zip(lengths, string_cols, (df1, df2)))

    #export to numpy as recarray
    first = first.to_records(column_dtypes=col_dtypes, index=False)
    #tile first with len_first
    #remember, len_first is the length of the other dataframe
    first = np.tile(first, (len_first, 1))
    #get a 1d array
    first = np.concatenate(first)
    #sorting here ensures we get each row of the first
    #with the entire rows of the other dataframe
    np.recarray.sort(first, order=first.dtype.names[0])

    #same process as first, except there'll be no sorting
    last = last.to_records(column_dtypes=col_dtypes, index=False)
    last = np.tile(last, (len_last, 1))
    last = np.concatenate(last)
    result = rfn.merge_arrays((first, last), flatten=True, asrecarray=True)
    return pd.DataFrame.from_records(result)


def grid_computation_list(dfs):
    return reduce(compute_two_dfs, dfs)


def grid_computation(dfs, dicts):
    if not dicts:
        result = grid_computation_list(dfs)
    elif not dfs:
        result = grid_computation_dict(dicts)
    else:
        dfs.append(grid_computation_dict(dicts))
        result = grid_computation_list(dfs)
    return result


########################### Tests


def test_not_a_dict():
    """Test that entry(list) is not a dictionary"""
    data = [60, 70]
    with pytest.raises(ValueError):
        assert expand_grid(data)


def test_not_a_dict_1():
    """Test that entry (dataframe) is not a dictionary"""
    data = pd.DataFrame([60, 70])
    with pytest.raises(ValueError):
        assert expand_grid(data)


def test_empty_dict():
    """Test that entry should not be empty"""
    data = {}
    with pytest.raises(ValueError):
        assert expand_grid(data)


def test_scalar_to_list():
    """
    Test that dictionary values are all converted to lists.
    """
    data = {
        "x": 1,
        "y": "string",
        "z": set((2, 3, 4)),
        "a": tuple((26, 50)),
        "b": None,
        "c": 1.2,
        "d": True,
        "e": False
    }
    expected = ([], {
        'x': [1],
        'y': ['s', 't', 'r', 'i', 'n', 'g'],
        'z': [2, 3, 4],
        'a': [26, 50],
        'b': [None],
        'c': [1.2],
        'd': [True],
        'e': [False]
    })
    assert check_instance(data) == expected


def test_nested_dict():
    data = {"x": {"y": 2}}
    with pytest.raises(ValueError):
        assert expand_grid(data)


def test_numpy():
    data = {"x": np.array([])}
    with pytest.raises(ValueError):
        assert expand_grid(data)


def test_numpy_1d():
    data = {"x": np.array([2, 3])}
    expected = pd.DataFrame(np.array([2, 3]), columns=['x'])
    assert_frame_equal(expand_grid(others=data), expected)


def test_numpy_2d():
    data = {"x": np.array([[2, 3]])}
    expected = pd.DataFrame(np.array([[2, 3]])).add_prefix("x_")
    assert_frame_equal(expand_grid(others=data), expected)


def test_numpy_gt_2d():
    data = {"x": np.array([[[2, 3]]])}
    with pytest.raises(ValueError):
        assert expand_grid(data)


def test_series_empty():
    """Test that values in key value pair should not be empty ... for Series"""
    data = {"x": pd.Series([], dtype='int')}
    with pytest.raises(ValueError):
        assert expand_grid(others=data)


def test_series_not_multiIndex_no_name():
    """Test for single index series"""
    data = {"x": pd.Series([2, 3])}
    expected = pd.DataFrame([2, 3], columns=["x"])
    assert_frame_equal(check_instance(data)[0][0], expected)


def test_series_not_multiIndex_with_name():
    """Test for single index series with name"""
    data = {"x": pd.Series([2, 3], name="y")}
    expected = pd.DataFrame([2, 3], columns=["x_y"])
    assert_frame_equal(check_instance(data)[0][0], expected)


def test_series_multiIndex():
    """Test that multiIndexed series trigger error"""
    data = {
        "x": pd.Series([2, 3],
                       index=pd.MultiIndex.from_arrays([[1, 2], [3, 4]]))
    }
    with pytest.raises(ValueError):
        assert expand_grid(data)


def test_dataframe_empty():
    """Test for empty dataframes"""
    data = {"x": pd.DataFrame([])}
    with pytest.raises(ValueError):
        assert expand_grid(data)


def test_dataframe_single_index():
    """Test for single indexed dataframes"""
    data = {"x": pd.DataFrame([[2, 3], [6, 7]])}
    expected = pd.DataFrame([[2, 3], [6, 7]]).add_prefix("x_")
    assert_frame_equal(expand_grid(others=data), expected)


def test_dataframe_multi_index_index():
    """Test for multiIndex dataframe"""
    data = {
        "x":
        pd.DataFrame([[2, 3], [6, 7]],
                     index=pd.MultiIndex.from_arrays([['a', 'b'], ['y', 'z']]))
    }

    with pytest.raises(ValueError):
        assert expand_grid(others=data)


def test_dataframe_multi_index_column():
    data = {
        "x":
        pd.DataFrame([[2, 3], [6, 7]],
                     columns=pd.MultiIndex.from_arrays([['m', 'n'], ['p',
                                                                     'q']]))
    }

    with pytest.raises(ValueError):
        assert expand_grid(others=data)


def test_dataframe_multi_index_index_column():
    data = {
        "x":
        pd.DataFrame([[2, 3], [6, 7]],
                     index=pd.MultiIndex.from_arrays([['a', 'b'], ['y', 'z']]),
                     columns=pd.MultiIndex.from_arrays([['m', 'n'], ['p',
                                                                     'q']]))
    }
    with pytest.raises(ValueError):
        assert expand_grid(others=data)


def test_list_empty():
    data = {"x": [], "y": [2, 3]}
    with pytest.raises(ValueError):
        assert expand_grid(others=data)


def test_lists():
    data = {"x": [[2, 3], [4, 3]]}
    expected = pd.DataFrame([[2, 3], [4, 3]]).add_prefix("x_")
    assert_frame_equal(expand_grid(others=data), expected)


def test_lists_all_scalar():
    data = {"x": [2, 3, 4, 5, "ragnar"]}
    expected = {"x": [2, 3, 4, 5, "ragnar"]}
    assert check_instance(data)[-1] == expected


def test_lists_not_all_scalar():
    data = {"x": [[2, 3], 4, 5, "ragnar"]}
    with pytest.raises(ValueError):
        assert expand_grid(others=data)


def test_computation_output_1():
    """Test output if entry contains no dataframes/series"""
    data = {"x": range(1, 4), "y": [1, 2]}
    expected = pd.DataFrame({
        'x': {
            0: 1,
            1: 1,
            2: 2,
            3: 2,
            4: 3,
            5: 3
        },
        'y': {
            0: 1,
            1: 2,
            2: 1,
            3: 2,
            4: 1,
            5: 2
        }
    })
    assert_frame_equal(expand_grid(others=data), expected)


def test_computation_output_2():
    """Test output if entry contains only dataframes/series"""
    data = {
        "df": pd.DataFrame({
            "x": range(1, 6),
            "y": [5, 4, 3, 2, 1]
        }),
        "df1": pd.DataFrame({
            "x": range(4, 7),
            "y": [6, 5, 4]
        })
    }

    expected = pd.DataFrame([{
        'df_x': 1,
        'df_y': 5,
        'df1_x': 4,
        'df1_y': 6
    }, {
        'df_x': 1,
        'df_y': 5,
        'df1_x': 5,
        'df1_y': 5
    }, {
        'df_x': 1,
        'df_y': 5,
        'df1_x': 6,
        'df1_y': 4
    }, {
        'df_x': 2,
        'df_y': 4,
        'df1_x': 4,
        'df1_y': 6
    }, {
        'df_x': 2,
        'df_y': 4,
        'df1_x': 5,
        'df1_y': 5
    }, {
        'df_x': 2,
        'df_y': 4,
        'df1_x': 6,
        'df1_y': 4
    }, {
        'df_x': 3,
        'df_y': 3,
        'df1_x': 4,
        'df1_y': 6
    }, {
        'df_x': 3,
        'df_y': 3,
        'df1_x': 5,
        'df1_y': 5
    }, {
        'df_x': 3,
        'df_y': 3,
        'df1_x': 6,
        'df1_y': 4
    }, {
        'df_x': 4,
        'df_y': 2,
        'df1_x': 4,
        'df1_y': 6
    }, {
        'df_x': 4,
        'df_y': 2,
        'df1_x': 5,
        'df1_y': 5
    }, {
        'df_x': 4,
        'df_y': 2,
        'df1_x': 6,
        'df1_y': 4
    }, {
        'df_x': 5,
        'df_y': 1,
        'df1_x': 4,
        'df1_y': 6
    }, {
        'df_x': 5,
        'df_y': 1,
        'df1_x': 5,
        'df1_y': 5
    }, {
        'df_x': 5,
        'df_y': 1,
        'df1_x': 6,
        'df1_y': 4
    }])

    assert_frame_equal(expand_grid(others=data), expected)


def test_computation_output_3():
    """Test mix of dataframes and lists"""
    data = {
        "df": pd.DataFrame({
            "x": range(1, 3),
            "y": [2, 1]
        }),
        "z": range(1, 4)
    }
    expected = pd.DataFrame(
        {
            "df_x": [1, 1, 1, 2, 2, 2,],
            "df_y": [2, 2, 2, 1, 1, 1,],
            "z": [1, 2, 3, 1, 2, 3,],
        }
    )
    assert_frame_equal(expand_grid(others=data), expected)


def test_computation_output_4():
    """ output from list of strings"""
    data = {"l1": ascii_lowercase[:3], "l2": ascii_uppercase[:3]}
    expected = pd.DataFrame([{
        'l1': 'a',
        'l2': 'A'
    }, {
        'l1': 'a',
        'l2': 'B'
    }, {
        'l1': 'a',
        'l2': 'C'
    }, {
        'l1': 'b',
        'l2': 'A'
    }, {
        'l1': 'b',
        'l2': 'B'
    }, {
        'l1': 'b',
        'l2': 'C'
    }, {
        'l1': 'c',
        'l2': 'A'
    }, {
        'l1': 'c',
        'l2': 'B'
    }, {
        'l1': 'c',
        'l2': 'C'
    }])
    assert_frame_equal(expand_grid(others=data), expected)


def test_df_key():
    """ Raise error if dataframe key is not supplied"""
    df = pd.DataFrame({"x": [2, 3]})
    others = {"df": pd.DataFrame({"x": range(1, 6), "y": [5, 4, 3, 2, 1]})}

    with pytest.raises(ValueError):
        assert (expand_grid(df, others))


def test_df_others():
    """ Raise error if others is not a dict"""
    df = pd.DataFrame({"x": [2, 3]})
    others = [5, 4, 3, 2, 1]

    with pytest.raises(ValueError):
        assert (expand_grid(df, others))


def test_df_output():
    """Test output from chaining method to a dataframe"""
    #example is from tidyverse's expand_grid page
    #https://tidyr.tidyverse.org/reference/expand_grid.html#compared-to-expand-grid
    df = pd.DataFrame({"x": range(1, 3), "y": [2, 1]})
    others = {"z": range(1, 4)}
    expected = pd.DataFrame([{
        'df_x': 1,
        'df_y': 2,
        'z': 1
    }, {
        'df_x': 1,
        'df_y': 2,
        'z': 2
    }, {
        'df_x': 1,
        'df_y': 2,
        'z': 3
    }, {
        'df_x': 2,
        'df_y': 1,
        'z': 1
    }, {
        'df_x': 2,
        'df_y': 1,
        'z': 2
    }, {
        'df_x': 2,
        'df_y': 1,
        'z': 3
    }])
    result = expand_grid(df, df_key="df", others=others)
    assert_frame_equal(result, expected)
