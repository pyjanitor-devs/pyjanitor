from string import ascii_lowercase, ascii_uppercase

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor.functions import expand_grid
from janitor.utils import _grid_computation

not_others = [
    (None, [60, 70]),
    (None, pd.DataFrame([60, 70])),
    (pd.DataFrame({"x": [2, 3]}), [5, 4, 3, 2, 1]),
]


@pytest.mark.parametrize(
    """
    frame, others
    """,
    not_others,
)
def test_not_a_dict(
    frame, others,
):
    """Raise if `others` is not a dictionary."""
    with pytest.raises(TypeError):
        expand_grid(df=frame, others=others)


def test_df_key():
    """Raise error if dataframe key is not supplied."""
    df = pd.DataFrame({"x": [2, 3]})
    others = {"df": pd.DataFrame({"x": range(1, 6), "y": [5, 4, 3, 2, 1]})}

    with pytest.raises(KeyError):
        expand_grid(df, others=others)


def test_numpy_gt_2d():
    """Raise error if numpy array dimension is greater than 2."""
    data = {"x": np.array([[[2, 3]]])}
    with pytest.raises(ValueError):
        expand_grid(others=data)


# no reason to check if entry is empty
empty_containers = [
    ({}),
    ({"x": pd.Series([], dtype="int")}),
    ({"x": pd.DataFrame([])}),
    ({"x": [], "y": [2, 3]}),
    ({"x": [2, 3], "y": set()}),
    ({"x": np.array([])}),
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
    """Test when the entry contains is a single indexed dataframe/series."""
    assert_frame_equal(expand_grid(others=frames_series), outputs)


multiIndex_pandas = [
    (
        {
            "x": pd.Series(
                [2, 3, 4],
                name="mika",
                index=pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)]),
            )
        },
        pd.DataFrame({"x_mika": [2, 3, 4]}),
    ),
    pytest.param(
        {  # expand_grid cant deal with MultiIndex columns
            "x": pd.DataFrame(
                [[2, 3, 4]],
                columns=pd.MultiIndex.from_tuples([(1, 2), (3, 4), (5, 6)]),
            )
        },
        pd.DataFrame({"x_0": [2], "x_1": [3], "x_2": [4]}),
        marks=pytest.mark.xfail,
    ),
]

multiIndex_output = [
    pd.DataFrame({"x_mika": [2, 3, 4]}),
    pd.DataFrame({"x_0": [2], "x_1": [3], "x_2": [4]}),
]

zip_multiIndex = zip(multiIndex_pandas, multiIndex_output)


@pytest.mark.parametrize(
    "multiIndex_data,multiIndex_outputs", multiIndex_pandas
)
def test_frames_series_multi_iIdex(multiIndex_data, multiIndex_outputs):
    """
    Test that expand_grid works with multiIndex dataframe and series,
    and that the returned dataframe has a single index and column.
    """
    assert_frame_equal(expand_grid(others=multiIndex_data), multiIndex_outputs)


@pytest.mark.xfail
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

    assert _grid_computation(data) == expected


def test_computation_output():
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


def test_df_chaining():
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


input_others = [
    {
        "df": pd.DataFrame({"x": range(1, 6), "y": [5, 4, 3, 2, 1]}),
        "df1": pd.DataFrame({"x": range(4, 7), "y": [6, 5, 4]}),
    },
    {"df": pd.DataFrame({"x": range(1, 3), "y": [2, 1]}), "z": range(1, 4)},
    {"l1": list(ascii_lowercase[:3]), "l2": list(ascii_uppercase[:3])},
    {"x": [[2, 3], [4, 3]]},
    {"x": np.array([2, 3])},
    {"x": np.array([[2, 3]])},
    {
        "x": np.reshape(np.arange(1, 5), (2, -1), order="F"),
        "y": np.reshape(np.arange(5, 9), (2, -1), order="F"),
    },
    {"V1": (5, np.nan, 1), "V2": (1, 3, 2)},
]

output_others = [
    pd.DataFrame(
        {
            "df_x": [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5],
            "df_y": [5, 5, 5, 4, 4, 4, 3, 3, 3, 2, 2, 2, 1, 1, 1],
            "df1_x": [4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6],
            "df1_y": [6, 5, 4, 6, 5, 4, 6, 5, 4, 6, 5, 4, 6, 5, 4],
        }
    ),
    pd.DataFrame(
        {
            "df_x": [1, 1, 1, 2, 2, 2],
            "df_y": [2, 2, 2, 1, 1, 1],
            "z": [1, 2, 3, 1, 2, 3],
        }
    ),
    pd.DataFrame(
        {
            "l1": ["a", "a", "a", "b", "b", "b", "c", "c", "c"],
            "l2": ["A", "B", "C", "A", "B", "C", "A", "B", "C"],
        }
    ),
    pd.DataFrame({"x": [[2, 3], [4, 3]]}),
    pd.DataFrame(np.array([2, 3]), columns=["x"]),
    pd.DataFrame(np.array([[2, 3]]), columns=["x_0", "x_1"]),
    pd.DataFrame(
        {
            "x_0": [1, 1, 2, 2],
            "x_1": [3, 3, 4, 4],
            "y_0": [5, 6, 5, 6],
            "y_1": [7, 8, 7, 8],
        }
    ),
    pd.DataFrame(
        {
            "V1": [5.0, 5.0, 5.0, np.nan, np.nan, np.nan, 1.0, 1.0, 1.0],
            "V2": [1, 3, 2, 1, 3, 2, 1, 3, 2],
        }
    ),
]

zip_others_only = zip(input_others, output_others)


@pytest.mark.parametrize("grid_input,grid_output", zip_others_only)
def test_expand_grid_others_only(grid_input, grid_output):
    """
    Tests that expand_grid output is correct when only the `others` argument
    is supplied.
    """
    assert_frame_equal(expand_grid(others=grid_input), grid_output)


df = {"x": pd.Series([2, 3])}
others = {
    "x": pd.DataFrame([[2, 3], [6, 7]]),
    "y": pd.Series([2, 3]),
    "z": range(1, 4),
    "k": ("a", "b", "c"),
}

result = expand_grid(
    others={
        "x": np.reshape(np.arange(1, 5), (2, -1), order="F"),
        "y": np.reshape(np.arange(5, 9), (2, -1), order="F"),
        "z": [0, 1, 2, 3],
    }
)

print(result)
