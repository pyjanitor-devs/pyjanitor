import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df_summarise():
    data = {
        "avg_jump": [3, 4, 1, 2, 3, 4],
        "avg_run": [3, 4, 1, 3, 2, 4],
        "combine_id": [100200, 100200, 101200, 101200, 102201, 103202],
    }
    return pd.DataFrame(data)


@pytest.fixture
def dfmi():
    """Create a MultiIndex DataFrame"""

    # https://pandas.pydata.org/docs/user_guide/advanced.html#using-slicers
    def mklbl(prefix, n):
        return ["%s%s" % (prefix, i) for i in range(n)]

    miindex = pd.MultiIndex.from_product(
        [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
    )

    micolumns = pd.MultiIndex.from_tuples(
        [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")],
        names=["lvl0", "lvl1"],
    )
    dfmi = (
        pd.DataFrame(
            np.arange(len(miindex) * len(micolumns)).reshape(
                (len(miindex), len(micolumns))
            ),
            index=miindex,
            columns=micolumns,
        )
        .sort_index()
        .sort_index(axis=1)
    )
    dfmi.index.names = list("ABCD")
    return dfmi


def test_summarise_callable_series_unnamed(df_summarise):
    """Test output for callable"""
    with pytest.raises(
        ValueError, match="Ensure the pandas Series object has a name"
    ):
        df_summarise.summarise(lambda df: df.mean())


def test_summarise_callable_not_pandas_object(df_summarise):
    """Raise if output is not a pandas Series/DataFrame"""
    with pytest.raises(
        TypeError,
        match="The output from the aggregation should be a named Series or a DataFrame",
    ):
        df_summarise.summarise(lambda df: df.avg_jump.mean())


def test_summarise_callable_series(df_summarise):
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


def test_summarise_by_callable_grp_grouped(df_summarise):
    """Test output for a callable"""
    grp = df_summarise.groupby("combine_id")
    actual = grp.summarise(lambda df: df.sum())
    expected = grp.sum()
    assert_frame_equal(actual, expected)


def test_summarise_dict_df_str(df_summarise):
    """Test output for a dictionary"""
    actual = df_summarise.summarise({"avg_run": "mean"})
    expected = df_summarise.agg({"avg_run": ["mean"]})
    assert_frame_equal(actual, expected)


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
    expected = df_summarise.agg({"avg_run": [lambda df: df.sum()]})
    assert_frame_equal(actual, expected)


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
    actual = df_summarise.summarise({"avg_run_sum": ("avg_run", "sum")})
    expected = df_summarise.agg(avg_run_sum=("avg_run", "sum")).T.rename(
        index={"avg_run": "sum"}
    )
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


def test_summarise_dict_tuple_count_not_eq_2(df_summarise):
    """Raise error if length of tuple is not 2"""
    with pytest.raises(ValueError, match="the tuple has to be a length of 2"):
        df_summarise.summarise({"avg_run_sum": ("avg_run",)})


def test_summarise_tuple_count_not_eq_2(df_summarise):
    """Raise error if length of tuple is not 2"""
    with pytest.raises(ValueError, match="the tuple has to be a length of 2"):
        df_summarise.summarise(("avg_run",))


def test_summarise_df_tuple(df_summarise):
    "Test output for a tuple"
    actual = df_summarise.summarise(("avg_run", "sum"))
    expected = df_summarise.agg({"avg_run": ["sum"]})
    assert_frame_equal(actual, expected)


def test_summarise_by_tuple(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.summarise(("avg_run", "mean"), by="combine_id")
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "mean"})
    assert_frame_equal(actual, expected)


def test_summarise_by_tuple_grouped(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.groupby("combine_id").summarise(("avg_run", "mean"))
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "mean"})
    assert_frame_equal(actual, expected)


def test_summarise_tuple_df_callable(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.summarise(("avg_run", lambda df: df.sum()))
    expected = df_summarise.agg({"avg_run": [lambda df: df.sum()]})
    assert_frame_equal(actual, expected)


def test_summarise_tuple_by_callable(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.summarise(
        ("avg_run", lambda df: df.sum()), by="combine_id"
    )
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "sum"})
    assert_frame_equal(actual, expected)


def test_summarise_tuple_by_callable_grouped(df_summarise):
    """Test output for a tuple"""
    actual = df_summarise.groupby("combine_id").summarise(
        ("avg_run", lambda df: df.sum())
    )
    expected = df_summarise.groupby("combine_id").agg({"avg_run": "sum"})
    assert_frame_equal(actual, expected)


def test_summarise_MI(dfmi):
    """Test summarise on a MultiIndex"""
    actual = dfmi.summarise(("a", "min"))
    expected = dfmi.agg(
        {("a", "bar"): "min", ("a", "foo"): ["min"]}
    ).rename_axis(columns=[None, None])
    assert_frame_equal(actual, expected)


def test_summarise_MI_different_levels(dfmi):
    """Test summarise on a MultiIndex"""
    actual = dfmi.summarise(
        {("a", "bar"): "sum", ("rar", "bla", "gah"): (("a", "foo"), "mean")},
        ("b", "min"),
        by={"level": "A"},
    )
    actual.columns = ["A", "B", "C", "D"]
    grp = dfmi.groupby(level="A")
    expected = grp.agg(
        {
            ("a", "bar"): "sum",
            ("a", "foo"): "mean",
            ("b", "bah"): "min",
            ("b", "foo"): "min",
        }
    )
    expected.columns = ["A", "B", "C", "D"]
    assert_frame_equal(actual, expected)


def test_summarise_MI_different_levels_dict(dfmi):
    """Test summarise on a MultiIndex"""
    actual = dfmi.summarise(
        {("a", "bar"): "sum", "rar": (("a", "foo"), "mean")},
        ("b", "min"),
        by={"level": "A"},
    )
    actual.columns = ["A", "B", "C", "D"]
    grp = dfmi.groupby(level="A")
    expected = grp.agg(
        {
            ("a", "bar"): "sum",
            ("a", "foo"): "mean",
            ("b", "bah"): "min",
            ("b", "foo"): "min",
        }
    )
    expected.columns = ["A", "B", "C", "D"]
    assert_frame_equal(actual, expected)


def test_summarise_MI_different_levels_tuple(dfmi):
    """Test summarise on a MultiIndex"""
    actual = dfmi.summarise(
        {("a", "bar"): "sum", ("rar",): (("a", "foo"), "mean")},
        ("b", "min"),
        by={"level": "A"},
    )
    actual.columns = ["A", "B", "C", "D"]
    grp = dfmi.groupby(level="A")
    expected = grp.agg(
        {
            ("a", "bar"): "sum",
            ("a", "foo"): "mean",
            ("b", "bah"): "min",
            ("b", "foo"): "min",
        }
    )
    expected.columns = ["A", "B", "C", "D"]
    assert_frame_equal(actual, expected)


def test_summarise_MI_different_levels_tuple_grouped(dfmi):
    """Test summarise on a MultiIndex"""
    actual = dfmi.groupby(level="A").summarise(
        {("a", "bar"): "sum", ("rar",): (("a", "foo"), "mean")},
        ("b", "min"),
    )
    actual.columns = ["A", "B", "C", "D"]
    grp = dfmi.groupby(level="A")
    expected = grp.agg(
        {
            ("a", "bar"): "sum",
            ("a", "foo"): "mean",
            ("b", "bah"): "min",
            ("b", "foo"): "min",
        }
    )
    expected.columns = ["A", "B", "C", "D"]
    assert_frame_equal(actual, expected)


def test_summarise_MI_different_levels_dataframe(dfmi):
    """raise if dictionary value is a tuple and the returned aggregate is a DataFrame"""
    with pytest.raises(TypeError, match="Expected a pandas Series object;.+"):
        dfmi.summarise(
            {"rar": (("a", "foo"), ["min", "mean"])},
            by={"level": "A"},
        )
