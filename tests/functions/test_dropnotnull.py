import pandas as pd
import pytest


@pytest.mark.functions
def test_dropnotnull(missingdata_df):
    df = missingdata_df.clean_names().dropnotnull("bell_chart")

    assert pd.isnull(df["bell_chart"]).all()
