import pandas as pd
import pytest


@pytest.mark.functions
def test_dropnotnull(missingdata_df):
    df = missingdata_df.clean_names()
    df_drop = df.dropnotnull("bell_chart")

    assert pd.isnull(df_drop["bell_chart"]).all()

    pd.testing.assert_frame_equal(df.loc[df_drop.index], df_drop)
