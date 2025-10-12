import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df_mutate():
    data = {
        "avg_jump": [3, 4, 1, 2, 3, 4],
        "avg_run": [3, 4, 1, 3, 2, 4],
        "combine_id": [100200, 100200, 101200, 101200, 102201, 103202],
    }
    return pd.DataFrame(data)


def test_mutate_callable_dataframe(df_mutate):
    """Test output for callable"""
    expected = df_mutate.mutate(lambda df: df.add(1))
    actual = df_mutate.add(1)
    assert_frame_equal(actual, expected)


def test_mutate_callable_series(df_mutate):
    """Test output for callable"""
    expected = df_mutate.mutate(lambda df: df.sum(axis=1).rename("new_column"))
    actual = df_mutate.assign(new_column=lambda df: df.sum(axis=1))
    assert_frame_equal(actual, expected)


def test_mutate_callable_unnamed_series(df_mutate):
    """Raise if Series is unnamed"""
    with pytest.raises(
        ValueError, match="Ensure the pandas Series object has a name"
    ):
        df_mutate.mutate(lambda df: df.sum(axis=1))


def test_mutate_callable(df_mutate):
    "Raise if output of callable is not a pandas Series/DataFrame"
    with pytest.raises(
        TypeError,
        match="The output from the mutation should be a named Series or a DataFrame",
    ):
        df_mutate.mutate(lambda df: np.sum(df["avg_run"]))


def test_mutate_wrong_arg(df_mutate):
    """
    Raise if wrong arg is provided
    """
    with pytest.raises(
        TypeError,
        match="The output from the mutation should be a named Series or a DataFrame",
    ):
        df_mutate.mutate(1)


def test_mutate_dict_df_str(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate({"avg_run": "sqrt"})
    expected = df_mutate.assign(avg_run=df_mutate["avg_run"].transform("sqrt"))
    assert_frame_equal(actual, expected)


def test_mutate_dict_df_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate({"avg_run": lambda df: df.sum()})
    expected = df_mutate.assign(avg_run=df_mutate["avg_run"].sum())
    assert_frame_equal(actual, expected)


def test_mutate_dict_df_tuple(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate({"avg_run_sqrt": ("avg_run", "sqrt")})
    expected = df_mutate.assign(
        avg_run_sqrt=df_mutate["avg_run"].transform("sqrt")
    )
    assert_frame_equal(actual, expected)


def test_mutate_tuple_count_not_eq_2(df_mutate):
    """Raise error if length of tuple is not 2"""
    with pytest.raises(ValueError, match="the tuple has to be a length of 2"):
        df_mutate.mutate(("avg_run",))


def test_mutate_df_tuple(df_mutate):
    "Test output for a tuple"
    actual = df_mutate.mutate(("avg_run", "sqrt"))
    expected = df_mutate.assign(avg_run=df_mutate["avg_run"].transform("sqrt"))
    assert_frame_equal(actual, expected)


def test_mutate_tuple_df_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.mutate(("avg_run", lambda df: df.sum()))
    expected = df_mutate.assign(avg_run=df_mutate["avg_run"].sum())
    assert_frame_equal(actual, expected)


def test_mutate_callable_by_grouped_object(df_mutate):
    """Test output for callable"""

    actual = df_mutate.groupby("combine_id").mutate(
        lambda df: df.avg_run.transform("sum")
    )
    grp = df_mutate.groupby("combine_id")
    expected = df_mutate.assign(avg_run=grp["avg_run"].transform("sum"))
    assert_frame_equal(actual.obj, expected)


def test_mutate_dict_by_str(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate({"avg_run": "mean"})
    grp = df_mutate.groupby("combine_id")["avg_run"]
    expected = df_mutate.assign(avg_run=grp.transform("mean"))
    assert_frame_equal(actual.obj, expected)


def test_mutate_dict_by_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(
        {"avg_run": lambda df: df.sum()}
    )
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.obj, expected)


def test_mutate_dict_by_transform_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(
        {"avg_run": lambda df: df.transform("sum")}
    )
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.obj, expected)


def test_mutate_dict_by_tuple(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(
        {"avg_run_mean": ("avg_run", "mean")}
    )
    expected = df_mutate.assign(
        avg_run_mean=df_mutate.groupby("combine_id")["avg_run"].transform(
            "mean"
        )
    )
    assert_frame_equal(actual.obj, expected)


def test_mutate_by_tuple(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(("avg_run", "mean"))
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("mean")
    )
    assert_frame_equal(actual.obj, expected)


def test_mutate_tuple_by_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(
        ("avg_run", lambda df: df.sum())
    )
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.obj, expected)


def test_mutate_tuple_by_grouped_object(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(
        ("avg_run", lambda df: df.sum())
    )
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.obj, expected)
