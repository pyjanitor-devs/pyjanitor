from string import ascii_lowercase, ascii_uppercase

import numpy as np
import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from janitor.functions import expand_grid
from janitor.utils import _check_instance


def test_not_a_dict():
    """Test that entry(list) is not a dictionary"""
    data = [60, 70]
    with pytest.raises(TypeError):
        assert expand_grid(others=data)


def test_not_a_dict_1():
    """Test that entry (dataframe) is not a dictionary"""
    data = pd.DataFrame([60, 70])
    with pytest.raises(TypeError):
        expand_grid(others=data)


def test_empty_dict():
    """Test that entry should not be empty"""
    data = {}
    with pytest.raises(ValueError):
        expand_grid(others=data)


def test_scalar_to_list():
    """
    Test that dictionary values are all converted to lists.
    """
    data = {
        "x": 1,
        "y": "string",
        "z": set([2, 3, 4]),
        "a": tuple([26, 50]),
        "b": None,
        "c": 1.2,
        "d": True,
        "e": False,
    }
    expected = (
        [],
        {
            "x": [1],
            "y": ["string"],
            "z": [2, 3, 4],
            "a": [26, 50],
            "b": [None],
            "c": [1.2],
            "d": [True],
            "e": [False],
        },
    )
    assert _check_instance(data) == expected


def test_nested_dict():
    """Raise error if dictionary is nested in a dictionary's values"""
    data = {"x": {"y": 2}}
    with pytest.raises(TypeError):
        expand_grid(others=data)


def test_numpy():
    """Raise error if numpy array in dictionary's values is empty"""
    data = {"x": np.array([])}
    with pytest.raises(ValueError):
        expand_grid(others=data)


def test_numpy_1d():
    """Test output from a 1d numpy array """
    data = {"x": np.array([2, 3])}
    expected = pd.DataFrame(np.array([2, 3]), columns=["x"])
    assert_frame_equal(expand_grid(others=data), expected)


def test_numpy_2d():
    """Test output from a 2d numpy array"""
    data = {"x": np.array([[2, 3]])}
    expected = pd.DataFrame(np.array([[2, 3]])).add_prefix("x_")
    assert_frame_equal(expand_grid(others=data), expected)


def test_numpy_gt_2d():
    """Raise error if numpy array dimension is greater than 2"""
    data = {"x": np.array([[[2, 3]]])}
    with pytest.raises(TypeError):
        expand_grid(others=data)


def test_series_empty():
    """Test that values in key value pair should not be empty ... for Series"""
    data = {"x": pd.Series([], dtype="int")}
    with pytest.raises(ValueError):
        expand_grid(others=data)


def test_series_not_multi_index_no_name():
    """Test for single index series"""
    data = {"x": pd.Series([2, 3])}
    expected = pd.DataFrame([2, 3], columns=["x"])
    assert_frame_equal(_check_instance(data)[0][0], expected)


def test_series_not_multi_index_with_name():
    """Test for single index series with name"""
    data = {"x": pd.Series([2, 3], name="y")}
    expected = pd.DataFrame([2, 3], columns=["x_y"])
    assert_frame_equal(_check_instance(data)[0][0], expected)


def test_series_multi_index():
    """Test that multiIndexed series trigger error"""
    data = {
        "x": pd.Series(
            [2, 3], index=pd.MultiIndex.from_arrays([[1, 2], [3, 4]])
        )
    }
    with pytest.raises(TypeError):
        expand_grid(others=data)


def test_dataframe_empty():
    """Trigger error for empty dataframes"""
    data = {"x": pd.DataFrame([])}
    with pytest.raises(ValueError):
        expand_grid(others=data)


def test_dataframe_single_index():
    """Test for single indexed dataframes"""
    data = {"x": pd.DataFrame([[2, 3], [6, 7]])}
    expected = pd.DataFrame([[2, 3], [6, 7]]).add_prefix("x_")
    assert_frame_equal(expand_grid(others=data), expected)


def test_dataframe_multi_index_index():
    """Trigger error if dataframe has a MultiIndex index"""
    data = {
        "x": pd.DataFrame(
            [[2, 3], [6, 7]],
            index=pd.MultiIndex.from_arrays([["a", "b"], ["y", "z"]]),
        )
    }

    with pytest.raises(TypeError):
        expand_grid(others=data)


def test_dataframe_multi_index_column():
    """Trigger error if dataframe has a MultiIndex column"""
    data = {
        "x": pd.DataFrame(
            [[2, 3], [6, 7]],
            columns=pd.MultiIndex.from_arrays([["m", "n"], ["p", "q"]]),
        )
    }

    with pytest.raises(TypeError):
        expand_grid(others=data)


def test_dataframe_multi_index_index_and_column():
    """Trigger error if dataframe has a MultiIndex column or index"""
    data = {
        "x": pd.DataFrame(
            [[2, 3], [6, 7]],
            index=pd.MultiIndex.from_arrays([["a", "b"], ["y", "z"]]),
            columns=pd.MultiIndex.from_arrays([["m", "n"], ["p", "q"]]),
        )
    }
    with pytest.raises(TypeError):
        expand_grid(others=data)


def test_list_empty():
    """Raise error if any list in dictionary's values is empty"""
    data = {"x": [], "y": [2, 3]}
    with pytest.raises(ValueError):
        expand_grid(others=data)


def test_lists():
    """
    Test expected output
    from one level nested lists
    in a dictionary's values
    """
    data = {"x": [[2, 3], [4, 3]]}
    expected = pd.DataFrame([[2, 3], [4, 3]]).add_prefix("x_")
    assert_frame_equal(expand_grid(others=data), expected)


def test_lists_all_scalar():
    """
    Test that all values in a list
    in dictionary's values are scalar
    """
    data = {"x": [2, 3, 4, 5, "ragnar"]}
    expected = {"x": [2, 3, 4, 5, "ragnar"]}
    assert _check_instance(data)[-1] == expected


def test_lists_not_all_scalar():
    """
    Trigger error if values in a list
    in the dictionary's values are not scalar
    """
    data = {"x": [[2, 3], 4, 5, "ragnar"]}
    with pytest.raises(ValueError):
        expand_grid(others=data)


def test_computation_output_1():
    """Test output if entry contains no dataframes/series"""
    data = {"x": range(1, 4), "y": [1, 2]}
    expected = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3], "y": [1, 2, 1, 2, 1, 2]})
    assert_frame_equal(expand_grid(others=data), expected)


def test_computation_output_2():
    """Test output if entry contains only dataframes/series"""
    data = {
        "df": pd.DataFrame({"x": range(1, 6), "y": [5, 4, 3, 2, 1]}),
        "df1": pd.DataFrame({"x": range(4, 7), "y": [6, 5, 4]}),
    }

    expected = pd.DataFrame(
        {
            "df_x": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "df_y": [5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1],
            "df1_x": [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],
            "df1_y": [6, 5, 4, 6, 5, 4, 6, 5, 4, 6, 5, 4, 6, 5, 4],
        }
    )

    assert_frame_equal(expand_grid(others=data), expected)


def test_computation_output_3():
    """Test mix of dataframes and lists"""
    data = {
        "df": pd.DataFrame({"x": range(1, 3), "y": [2, 1]}),
        "z": range(1, 4),
    }
    expected = pd.DataFrame(
        {
            "df_x": [1, 1, 1, 2, 2, 2],
            "df_y": [2, 2, 2, 1, 1, 1],
            "z": [1, 2, 3, 1, 2, 3],
        }
    )
    assert_frame_equal(expand_grid(others=data), expected)


def test_computation_output_4():
    """ Test output from list of strings"""
    data = {"l1": list(ascii_lowercase[:3]), "l2": list(ascii_uppercase[:3])}
    expected = pd.DataFrame(
        {
            "l1": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "l2": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
        }
    )
    assert_frame_equal(expand_grid(others=data), expected)


def test_df_key():
    """ Raise error if dataframe key is not supplied"""
    df = pd.DataFrame({"x": [2, 3]})
    others = {"df": pd.DataFrame({"x": range(1, 6), "y": [5, 4, 3, 2, 1]})}

    with pytest.raises(KeyError):
        expand_grid(df, others=others)


def test_df_others():
    """ Raise error if others is not a dict"""
    df = pd.DataFrame({"x": [2, 3]})
    others = [5, 4, 3, 2, 1]
    with pytest.raises(TypeError):
        expand_grid(df, others=others)


def test_df_output():
    """
    Test output from chaining method to a dataframe.
    Example is from tidyverse's expand_grid page -
    https://tidyr.tidyverse.org/reference/expand_grid.html#compared-to-expand-grid
    """
    df = pd.DataFrame({"x": range(1, 3), "y": [2, 1]})
    others = {"z": range(1, 4)}
    expected = pd.DataFrame(
        {
            "df_x": [1, 1, 1, 2, 2, 2],
            "df_y": [2, 2, 2, 1, 1, 1],
            "z": [1, 2, 3, 1, 2, 3],
        }
    )
    result = expand_grid(df, df_key="df", others=others)
    assert_frame_equal(result, expected)


def test_df_multi_index():
    """Test that datafarme is not a multiIndex"""
    df = {
        "x": pd.DataFrame(
            [[2, 3], [6, 7]],
            columns=pd.MultiIndex.from_arrays([["m", "n"], ["p", "q"]]),
        )
    }

    others = [5, 4, 3, 2, 1]

    with pytest.raises(TypeError):
        expand_grid(df, others=others)
