import pytest
from hypothesis import given

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(df=df_strategy())
def test_rename_column(df):
    df = df.clean_names().rename_column("a", "index")
    assert set(df.columns) == set(
        ["index", "bell_chart", "decorated_elephant", "animals@#$%^", "cities"]
    )
