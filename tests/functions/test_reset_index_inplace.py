import pandas as pd
import pytest


@pytest.mark.functions
def test_reset_index_inplace_obj_equivalence(dataframe):
    """ Make sure operation is indeed in place. """

    df_riip = dataframe.reset_index_inplace()

    assert df_riip is dataframe


@pytest.mark.functions
def test_reset_index_inplace_after_group(dataframe):
    """ Make sure equivalent output to non-in place. """

    df_sum = dataframe.groupby(["animals@#$%^", "cities"]).sum()

    df_sum_ri = df_sum.reset_index()
    df_sum.reset_index_inplace()

    pd.testing.assert_frame_equal(df_sum_ri, df_sum)


@pytest.mark.functions
def test_reset_index_inplace_drop(dataframe):
    """ Test that correctly accepts `reset_index()` parameters. """

    pd.testing.assert_frame_equal(
        dataframe.reset_index(drop=True),
        dataframe.reset_index_inplace(drop=True),
    )
