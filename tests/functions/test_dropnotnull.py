import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_dropnotnull(missingdata_df):
    df = missingdata_df.clean_names()
    df_drop = df.dropnotnull("bell_chart")

    assert pd.isna(df_drop["bell_chart"]).all()

    assert_frame_equal(df.loc[df_drop.index], df_drop)
