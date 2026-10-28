import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal


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
    with pytest.raises(ValueError, match="Ensure the pandas Series object has a name"):
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
    expected = df_mutate.assign(avg_run_sqrt=df_mutate["avg_run"].transform("sqrt"))
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
    assert_frame_equal(actual.get_columns("*"), expected)


def test_mutate_dict_by_str(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate({"avg_run": "mean"})
    grp = df_mutate.groupby("combine_id")["avg_run"]
    expected = df_mutate.assign(avg_run=grp.transform("mean"))
    assert_frame_equal(actual.get_columns("*"), expected)


def test_mutate_dict_by_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate({"avg_run": lambda df: df.sum()})
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.get_columns("*"), expected)


def test_mutate_dict_by_transform_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(
        {"avg_run": lambda df: df.transform("sum")}
    )
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.get_columns("*"), expected)


def test_mutate_dict_by_tuple(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(
        {"avg_run_mean": ("avg_run", "mean")}
    )
    expected = df_mutate.assign(
        avg_run_mean=df_mutate.groupby("combine_id")["avg_run"].transform("mean")
    )
    assert_frame_equal(actual.get_columns("*"), expected)


def test_mutate_by_tuple(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(("avg_run", "mean"))
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("mean")
    )
    assert_frame_equal(actual.get_columns("*"), expected)


def test_mutate_tuple_by_callable(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(("avg_run", lambda df: df.sum()))
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.get_columns("*"), expected)


def test_mutate_tuple_by_grouped_object(df_mutate):
    """Test output for a dictionary"""
    actual = df_mutate.groupby("combine_id").mutate(("avg_run", lambda df: df.sum()))
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.get_columns("*"), expected)


def test_assign_by(df_mutate):
    """Test output"""
    actual = df_mutate.groupby("combine_id").assign(
        avg_run=lambda df: df.avg_run.transform("sum")
    )
    expected = df_mutate.assign(
        avg_run=df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    )
    assert_frame_equal(actual.get_columns("*"), expected)


def test_assign_expressions_by(df_mutate):
    """Test output"""
    actual = df_mutate.groupby("combine_id").assign(total=pd.col("avg_run").mul(2))
    expected = df_mutate.assign(total=pd.col("avg_run").mul(2))
    assert_frame_equal(actual.get_columns("*"), expected)


def test_assign_by_after_aggregation(df_mutate):
    """New columns are visible even if the groupby was already aggregated."""
    grp = df_mutate.groupby("combine_id")
    grp.sum()
    actual = grp.assign(total=pd.col("avg_run").mul(2)).sum()
    expected = (
        df_mutate.assign(total=pd.col("avg_run").mul(2)).groupby("combine_id").sum()
    )
    assert_frame_equal(actual, expected)


def test_assign_by_after_get_columns(df_mutate):
    """Accessors used on the original groupby do not leak into the copy."""
    grp = df_mutate.groupby("combine_id")
    grp.get_columns("*")
    actual = grp.assign(total=pd.col("avg_run").mul(2)).get_columns("*")
    expected = df_mutate.assign(total=pd.col("avg_run").mul(2))
    assert_frame_equal(actual, expected)


def test_assign_by_chained(df_mutate):
    """Chained assigns see columns created by earlier assigns."""
    actual = (
        df_mutate.groupby("combine_id")
        .assign(total=pd.col("avg_run").mul(2))
        .assign(total_sum=lambda df: df["total"].transform("sum"))
    )
    total = df_mutate["avg_run"].mul(2)
    expected = df_mutate.assign(
        total=total, total_sum=total.groupby(df_mutate["combine_id"]).transform("sum")
    )
    assert_frame_equal(actual.get_columns("*"), expected)
    assert_frame_equal(actual.sum(), expected.groupby("combine_id").sum())


def test_assign_by_aggregation_between_columns(df_mutate):
    """A whole-frame callable does not hide columns assigned after it."""
    actual = df_mutate.groupby("combine_id").assign(
        jump_sum=lambda df: df.transform("sum")["avg_jump"],
        total=pd.col("avg_run").mul(2),
    )
    expected = df_mutate.assign(
        jump_sum=df_mutate.groupby("combine_id")["avg_jump"].transform("sum"),
        total=pd.col("avg_run").mul(2),
    )
    assert_frame_equal(actual.get_columns("*"), expected)
    assert_frame_equal(actual.sum(), expected.groupby("combine_id").sum())


def test_assign_by_leaves_original_groupby(df_mutate):
    """The original groupby and DataFrame are unchanged."""
    grp = df_mutate.groupby("combine_id")
    expected = grp.sum()
    grp.assign(total=pd.col("avg_run").mul(2)).sum()
    assert_frame_equal(grp.sum(), expected)
    assert list(df_mutate.columns) == ["avg_jump", "avg_run", "combine_id"]


def test_assign_by_after_mutate(df_mutate):
    """Groupbys returned by the deprecated mutate carry no stale accessors."""
    grp = df_mutate.groupby("combine_id")
    grp.assign(total=1)
    with pytest.warns(FutureWarning):
        mutated = grp.mutate({"run_sum": ("avg_run", "sum")})
    actual = mutated.assign(total=pd.col("run_sum").add(1))
    run_sum = df_mutate.groupby("combine_id")["avg_run"].transform("sum")
    expected = df_mutate.assign(run_sum=run_sum, total=run_sum.add(1))
    assert_frame_equal(actual.get_columns("*"), expected)


def test_assign_by_column_named_obj():
    """A column named `_obj` is not mistaken for a cached accessor."""
    df = pd.DataFrame({"a": [1, 1, 2], "_obj": [1, 2, 3]})
    actual = df.groupby("a").assign(total=pd.col("_obj").mul(2))
    expected = df.assign(total=pd.col("_obj").mul(2))
    assert_frame_equal(actual.get_columns("*"), expected)
    assert_frame_equal(actual.sum(), expected.groupby("a").sum())


def test_assign_by_group_key_keeps_groups(df_mutate):
    """Re-assigning a grouping column keeps the existing groups."""
    grp = df_mutate.groupby("combine_id")
    actual = grp.assign(combine_id=0)["avg_run"].sum()
    expected = grp["avg_run"].sum()
    assert_series_equal(actual, expected)


def test_mutate_by_whole_frame_callable_then_column(df_mutate):
    """Columns added after a whole-frame callable reach aggregations."""
    with pytest.warns(FutureWarning):
        actual = df_mutate.groupby("combine_id").mutate(
            lambda df: df.transform("sum").add_suffix("_s"),
            {"run_sum": ("avg_run", "sum")},
        )
    grp = df_mutate.groupby("combine_id")
    expected = df_mutate.assign(
        avg_jump_s=grp["avg_jump"].transform("sum"),
        avg_run_s=grp["avg_run"].transform("sum"),
        run_sum=grp["avg_run"].transform("sum"),
    )
    assert_frame_equal(actual.obj, expected)
    assert_frame_equal(actual.sum(), expected.groupby("combine_id").sum())
