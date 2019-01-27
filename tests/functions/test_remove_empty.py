import pandas as pd
import pytest
from hypothesis import given

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.tmp
@given(df=df_strategy())
def test_remove_empty(df):
    # This test ensures that there are no columns that are completely null.
    df = df.remove_empty()
    for col in df.columns:
        assert not pd.isnull(df[col]).all()
    for r, d in df.iterrows():
        assert not pd.isnull(d).all()
