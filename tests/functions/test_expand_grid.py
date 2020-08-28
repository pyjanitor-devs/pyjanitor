from string import ascii_lowercase, ascii_uppercase

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions import expand_grid
from janitor.utils import _check_instance


def test_not_a_dict():
    """Test that entry(list) is not a dictionary."""
    data = [60, 70]
    with pytest.raises(TypeError):
        expand_grid(others=data)


def test_not_a_dict_1():
    """Test that entry (dataframe) is not a dictionary."""
    data = pd.DataFrame([60, 70])
    with pytest.raises(TypeError):
        expand_grid(others=data)


empty_containers = [
    {},
    {"x": pd.Series([], dtype="int")},
    {"x": pd.DataFrame([])},
    {"x": [], "y": [2, 3]},
    {"x": [2, 3], "y": set()},
    {"x": np.array([])},
]


@pytest.mark.parametrize("empty", empty_containers)
def test_empty(empty):
    """Test that entry should not be empty."""
    with pytest.raises(ValueError):
        expand_grid(others=empty)


frames_series = [
    {"x": pd.Series([2, 3])},
    {"x": pd.Series([2, 3], name="y")},
    {"x": pd.DataFrame([[2, 3], [6, 7]])},
]

frames_series_output = [
    pd.DataFrame([2, 3], columns=["x"]),
    pd.DataFrame([2, 3], columns=["x_y"]),
    pd.DataFrame([[2, 3], [6, 7]]).add_prefix("x_"),
]

zip_frame_series = zip(frames_series, frames_series_output)


@pytest.mark.parametrize("frames_series,outputs", zip_frame_series)
def test_frames_series_single_index(frames_series, outputs):
    """Test for single indexed dataframe and series."""
    assert_frame_equal(expand_grid(others=frames_series), outputs)


multiIndex_pandas = [
    {
        "x": pd.Series(
            [2, 3, 4],
            name="mika",
            index=pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)]),
        )
    },
    {
        "x": pd.DataFrame(
            [[2, 3, 4]],
            columns=pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)]),
        )
    },
]

multiIndex_output = [
    pd.DataFrame({"x_mika": [2, 3, 4]}),
    pd.DataFrame({"x_0": [2], "x_1": [3], "x_2": [4]}),
]

zip_multiIndex = zip(multiIndex_pandas, multiIndex_output)


@pytest.mark.parametrize("multiIndex_data,multiIndex_outputs", zip_multiIndex)
def test_frames_series_multi_iIdex(multiIndex_data, multiIndex_outputs):
    """Test for multiIndex dataframe and series."""
    assert_frame_equal(expand_grid(others=multiIndex_data), multiIndex_outputs)


def test_scalar_to_list():
    """Test that scalars are converted to lists."""
    data = {
        "x": 1,
        "y": "string",
        "z": {2, 3, 4},
        "a": (26, 50),
        "b": None,
        "c": 1.2,
        "d": True,
        "e": False,
        "f": range(2, 12),
    }
    expected = {
        "x": [1],
        "y": ["string"],
        "z": {2, 3, 4},
        "a": (26, 50),
        "b": [None],
        "c": [1.2],
        "d": [True],
        "e": [False],
        "f": range(2, 12),
    }

    assert _check_instance(data) == expected


def test_numpy_1d():
    """Test output from a 1d numpy array."""
    data = {"x": np.array([2, 3])}
    expected = pd.DataFrame(np.array([2, 3]), columns=["x_0"])
    assert_frame_equal(expand_grid(others=data), expected)


def test_numpy_2d():
    """Test output from a 2d numpy array."""
    data = {"x": np.array([[2, 3]])}
    expected = pd.DataFrame(np.array([[2, 3]]), columns=["x_0", "x_1"])
    assert_frame_equal(expand_grid(others=data), expected)


def test_numpy_gt_2d():
    """Raise error if numpy array dimension is greater than 2."""
    data = {"x": np.array([[[2, 3]]])}
    with pytest.raises(ValueError):
        expand_grid(others=data)


def test_lists():
    """
    Test expected output from one level nested lists in a dictionary's values.
    """
    data = {"x": [[2, 3], [4, 3]]}
    expected = pd.DataFrame({"x": [[2, 3], [4, 3]]})
    assert_frame_equal(expand_grid(others=data), expected)


def test_computation_output_1():
    """Test output if entry contains no dataframes/series."""
    data = {"x": range(1, 4), "y": [1, 2]}
    expected = pd.DataFrame({"x": [1, 1, 2, 2, 3, 3], "y": [1, 2, 1, 2, 1, 2]})
    # h/t to @hectormz for picking it up
    # check_dtype is set to False for this test -
    # on Windows systems integer dtype default is int32
    # while on Ubuntu it comes up as int64
    # when the test is executed, the dtype on the left is int32
    # while the expected dataframe has a dtype of int64
    # And this was causing this particular test to fail on Windows
    # pandas has a minimum of int64 for integer columns
    # My suspicion is that because the resulting dataframe was
    # created solely from a dictionary via numpy
    # pandas simply picked up the dtype supplied from numpy
    # whereas the expected dataframe was created within Pandas
    # and got assigned the minimum dtype of int64
    # hence the error and the need to set check_dtype to False
    assert_frame_equal(expand_grid(others=data), expected, check_dtype=False)


def test_computation_output_2():
    """Test output if entry contains only dataframes/series."""
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
    """Test mix of dataframes and lists."""
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
    """Test output from list of strings."""
    data = {"l1": list(ascii_lowercase[:3]), "l2": list(ascii_uppercase[:3])}
    expected = pd.DataFrame(
        {
            "l1": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "l2": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
        }
    )
    assert_frame_equal(expand_grid(others=data), expected)


def test_df_key():
    """Raise error if dataframe key is not supplied."""
    df = pd.DataFrame({"x": [2, 3]})
    others = {"df": pd.DataFrame({"x": range(1, 6), "y": [5, 4, 3, 2, 1]})}

    with pytest.raises(KeyError):
        expand_grid(df, others=others)


def test_df_others():
    """Raise error if others is not a dict."""
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


def test_2d_arrays_multiple_columns():
    """Test on 2d arrays with multiple columns and rows."""
    # example referenced from tidyr page
    # https://tidyr.tidyverse.org/reference/expand_grid.html
    data = {
        "x": np.reshape(np.arange(1, 5), (2, -1), order="F"),
        "y": np.reshape(np.arange(5, 9), (2, -1), order="F"),
    }

    expected = pd.DataFrame(
        {
            "x_0": [1, 1, 2, 2],
            "x_1": [3, 3, 4, 4],
            "y_0": [5, 6, 5, 6],
            "y_1": [7, 8, 7, 8],
        }
    )

    assert_frame_equal(expand_grid(others=data), expected)


def test_null_entries():
    """Test on null entries"""
    data = {"V1": (5, np.nan, 1), "V2": (1, 3, 2)}
    expected = pd.DataFrame(
        {
            "V1": [5.0, 5.0, 5.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0],
            "V2": [1, 3, 2, 1, 3, 2, 1, 3, 2],
        }
    )
    assert_frame_equal(expand_grid(others=data), expected)
