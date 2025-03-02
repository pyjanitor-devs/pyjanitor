import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal


@pytest.fixture
def df_summarise():
    data = {
        "avg_jump": [3, 4, 1, 2, 3, 4],
        "avg_run": [3, 4, 1, 3, 2, 4],
        "combine_id": [100200, 100200, 101200, 101200, 102201, 103202],
    }
    return pd.DataFrame(data)


def test_summarise_wrong_arg(df_summarise):
    """
    Raise if wrong arg is provided
    """
    with pytest.raises(
        NotImplementedError, match="janitor.summarise is not supported for.+"
    ):
        df_summarise.summarise(1)


def test_mutate_callable_series_unnamed(df_summarise):
    """Test output for callable"""
    with pytest.raises(
        ValueError, match="Ensure the pandas Series object has a name"
    ):
        df_summarise.summarise(lambda df: df.mean())


def test_mutate_callable_series(df_summarise):
    """Test output for callable"""
    expected = df_summarise.summarise(lambda df: df.mean().rename("mean"))
    actual = df_summarise.mean().to_frame("mean")
    assert_frame_equal(actual, expected)


def test_summarise_by_callable_grp(df_summarise):
    """Test output for a callable"""
    grp = df_summarise.groupby("combine_id")
    actual = df_summarise.summarise(lambda df: df.sum(), by=grp)
    expected = grp.sum()
    assert_frame_equal(actual, expected)


def test_summarise_dict_df_str(df_summarise):
    """Test output for a dictionary"""
    actual = df_summarise.summarise({"avg_run": "mean"})
    expected = df_summarise.agg({"avg_run": "mean"})
    assert_series_equal(actual, expected)


def test_summarise_dict_by_str(df_summarise):
    """Test output for a dictionary"""
    actual = df_summarise.summarise(
        {"avg_run": "mean"}, by={"by": "combine_id"}
    )
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "mean"})
    assert_frame_equal(actual, expected)


def test_summarise_dict_df_callable(df_summarise):
    """Test output for a dictionary"""
    actual = df_summarise.summarise({"avg_run": lambda df: df.sum()})
    expected = df_summarise.agg({"avg_run": "sum"})
    assert_series_equal(actual, expected)


def test_summarise_dict_by_callable(df_summarise):
    """Test output for a dictionary"""
    actual = df_summarise.summarise(
        {"avg_run": lambda df: df.sum()}, by="combine_id"
    )
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "sum"})
    assert_frame_equal(actual, expected)


def test_summarise_dict_by_agg_callable(df_summarise):
    """Test output for a dictionary"""
    actual = df_summarise.summarise(
        {"avg_run": lambda df: df.agg("sum")}, by="combine_id"
    )
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "sum"})
    assert_frame_equal(actual, expected)


def test_summarise_dict_df_tuple(df_summarise):
    """Test output for a dictionary"""
    actual = (
        df_summarise.summarise({"avg_run_sum": ("avg_run", "sum")})
        .rename("avg_run")
        .to_frame()
    )
    expected = df_summarise.agg(avg_run_sum=("avg_run", "sum"))
    assert_frame_equal(actual, expected)


def test_summarise_dict_by_tuple(df_summarise):
    """Test output for a dictionary"""
    actual = df_summarise.summarise(
        {"avg_run_mean": ("avg_run", "mean")}, by="combine_id"
    )
    expected = df_summarise.groupby("combine_id").agg(
        avg_run_mean=("avg_run", "mean")
    )
    assert_frame_equal(actual, expected)


def test_summarise_tuple_count_not_eq_2(df_summarise):
    """Raise error if length of tuple is not 2"""
    with pytest.raises(ValueError, match="the tuple has to be a length of 2"):
        df_summarise.summarise(("avg_run",))


def test_summarise_df_tuple(df_summarise):
    "Test output for a tuple"
    actual = df_summarise.summarise(("avg_run", "sum"))
    expected = df_summarise.agg({"avg_run": "sum"})
    assert_series_equal(actual, expected)


def test_summarise_by_tuple(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.summarise(("avg_run", "mean"), by="combine_id")
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "mean"})
    assert_frame_equal(actual, expected)


def test_summarise_tuple_df_callable(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.summarise(("avg_run", lambda df: df.sum()))
    expected = df_summarise.agg({"avg_run": "sum"})
    assert_series_equal(actual, expected)


def test_summarise_tuple_by_callable(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.summarise(
        ("avg_run", lambda df: df.sum()), by="combine_id"
    )
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "sum"})
    assert_frame_equal(actual, expected)


def test_summarise_tuple_by_callable_dataframe(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.summarise(
        ("avg_run", lambda df: df.agg(["sum", "mean"])), by="combine_id"
    )
    expected = df_summarise.groupby("combine_id").agg(
        {"avg_run": ["sum", "mean"]}
    )
    assert_frame_equal(actual, expected)


def test_summarise_tuple_grouped_object(df_summarise):
    """Test output for a tuple"""
    grp = df_summarise.groupby("combine_id")
    actual = df_summarise.summarise(
        ("avg_run", lambda df: df.agg(["sum", "mean"])), by=grp
    )
    expected = grp.agg({"avg_run": ["sum", "mean"]})
    assert_frame_equal(actual, expected)
