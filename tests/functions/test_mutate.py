import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

# @pytest.fixture
# def df_mutate():
#     return pd.DataFrame(
#         {
#             "col1": [5, 10, 15],
#             "col2": [3, 6, 9],
#             "col3": [10, 100, 1_000],
#         }
#     )


@pytest.fixture
def df_mutate():
    data = {
        "avg_jump": [3, 4, 1, 2, 3, 4],
        "avg_run": [3, 4, 1, 3, 2, 4],
        "combine_id": [100200, 100200, 101200, 101200, 102201, 103202],
    }
    return pd.DataFrame(data)


def test_mutate_wrong_arg(df_mutate):
    """
    Raise if wrong arg is provided
    """
    with pytest.raises(
        NotImplementedError, match="janitor.mutate is not supported for.+"
    ):
        df_mutate.mutate(1)


def test_mutate_dict_df_str(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate({"avg_run": "sqrt"})
    expected = df_mutate.assign(avg_run=df_mutate["avg_run"].transform("sqrt"))
    assert_frame_equal(actual, expected)


def test_mutate_dict_by_str(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate({"avg_run": "mean"}, by="combine_id")
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("mean")
    )
    assert_frame_equal(actual, expected)


def test_mutate_dict_df_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate({"avg_run": lambda df: df.sum()})
    expected = df_mutate.assign(avg_run=df_mutate["avg_run"].sum())
    assert_frame_equal(actual, expected)


def test_mutate_dict_by_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate(
        {"avg_run": lambda df: df.sum()}, by="combine_id"
    )
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual, expected)


def test_mutate_dict_by_transform_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate(
        {"avg_run": lambda df: df.transform("sum")}, by="combine_id"
    )
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual, expected)


def test_mutate_dict_df_tuple(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate({"avg_run_sqrt": ("avg_run", "sqrt")})
    expected = df_mutate.assign(
        avg_run_sqrt=df_mutate["avg_run"].transform("sqrt")
    )
    assert_frame_equal(actual, expected)


def test_mutate_dict_by_tuple(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate(
        {"avg_run_mean": ("avg_run", "mean")}, by="combine_id"
    )
    expected = df_mutate.assign(
        avg_run_mean=df_mutate.groupby("combine_id")["avg_run"].transform(
            "mean"
        )
    )
    assert_frame_equal(actual, expected)
