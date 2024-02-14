from functools import reduce

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_collapse_levels_sanity(multiindex_with_missing_dataframe):
    with pytest.raises(TypeError):
        multiindex_with_missing_dataframe.collapse_levels(sep=3)


@pytest.mark.functions
def test_collapse_levels_non_multilevel(multiindex_with_missing_dataframe):
    # an already single-level DataFrame is not distorted
    assert_frame_equal(
        multiindex_with_missing_dataframe.copy().collapse_levels(),
        multiindex_with_missing_dataframe.collapse_levels().collapse_levels(),
    )


@pytest.mark.functions
def test_collapse_levels_functionality_2level(
    multiindex_with_missing_dataframe,
):
    assert all(
        multiindex_with_missing_dataframe.copy()  # noqa: PD011
        .collapse_levels()
        .columns.values
        == ["a", "Normal  Distribution", "decorated-elephant_r.i.p-rhino :'("]
    )
    assert all(
        multiindex_with_missing_dataframe.copy()  # noqa: PD011
        .collapse_levels(sep="AsDf")
        .columns.values
        == [
            "a",
            "Normal  Distribution",
            "decorated-elephantAsDfr.i.p-rhino :'(",
        ]
    )


@pytest.mark.functions
def test_collapse_levels_functionality_3level(
    multiindex_with_missing_3level_dataframe,
):
    assert all(
        multiindex_with_missing_3level_dataframe.copy()  # noqa: PD011
        .collapse_levels()
        .columns.values
        == [
            "a",
            "Normal  Distribution_Hypercuboid (???)",
            "decorated-elephant_r.i.p-rhino :'(_deadly__flamingo",
        ]
    )
    assert all(
        multiindex_with_missing_3level_dataframe.copy()  # noqa: PD011
        .collapse_levels(sep="AsDf")
        .columns.values
        == [
            "a",
            "Normal  DistributionAsDfHypercuboid (???)",
            "decorated-elephantAsDfr.i.p-rhino :'(AsDfdeadly__flamingo",
        ]
    )


@pytest.fixture
def mi_index():
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


def test_sep_and_glue(mi_index):
    """raise if sep and glue are provided."""
    msg = "Only one of sep or glue should be provided."
    with pytest.raises(ValueError, match=msg):
        mi_index.collapse_levels(sep="_", glue="_")


def test_glue_type(mi_index):
    """raise if glue is not str."""
    with pytest.raises(TypeError, match="glue should be one of.+"):
        mi_index.collapse_levels(glue=3)


def test_axis_type(mi_index):
    """raise if axis is not str."""
    with pytest.raises(TypeError, match="axis should be one of.+"):
        mi_index.collapse_levels(axis=3)


def test_axis_value(mi_index):
    """raise if axis is not index/columns."""
    msg = "axis argument should be either 'index' or 'columns'."
    with pytest.raises(ValueError, match=msg):
        mi_index.collapse_levels(axis="INDEX")


def test_glue_output(mi_index):
    """test output if glue is provided."""
    expected = mi_index.collapse_levels(glue="{A}{B}{C}{D}", axis="index")
    index = reduce(
        lambda x, y: x + y,
        [
            mi_index.index.get_level_values(num)
            for num in range(mi_index.index.nlevels)
        ],
    )
    actual = mi_index.copy()
    actual.index = index
    assert_frame_equal(expected, actual)
