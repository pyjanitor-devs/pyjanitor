import pytest
from hypothesis import given
from hypothesis import settings

from janitor.testing_utils.strategies import df_strategy


@pytest.mark.functions
@given(df=df_strategy())
@settings(deadline=None)
def test_reorder_columns(df):
    # NOTE: This test essentially has four different tests underneath it.
    # WE should be able to refactor this using pytest.mark.parametrize.

    # sanity checking of inputs

    # input is not a list or pd.Index
    with pytest.raises(TypeError):
        df.reorder_columns("a")

    # one of the columns is not present in the DataFrame
    with pytest.raises(IndexError):
        df.reorder_columns(["notpresent"])

    # reordering functionality

    # sanity check when desired order matches current order
    # this also tests whether the function can take Pandas Index objects
    assert all(df.reorder_columns(df.columns).columns == df.columns)

    # when columns are list & not all columns of DataFrame are included
    assert all(
        df.reorder_columns(["animals@#$%^", "Bell__Chart"]).columns
        == ["animals@#$%^", "Bell__Chart", "a", "decorated-elephant", "cities"]
    )
