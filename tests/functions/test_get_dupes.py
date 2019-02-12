import pandas as pd
import pytest


@pytest.mark.functions
def test_get_dupes():
    df = pd.DataFrame()
    df["a"] = [1, 2, 1]
    df["b"] = [1, 2, 1]
    df_dupes = df.get_dupes()
    assert df_dupes.shape == (2, 2)

    df2 = pd.DataFrame()
    df2["a"] = [1, 2, 3]
    df2["b"] = [1, 2, 3]
    df2_dupes = df2.get_dupes()
    assert df2_dupes.shape == (0, 2)
