import pytest


@pytest.mark.functions
def test_shuffle_without_index_reset(dataframe):
    """
    Test the shuffle function.

    This test checks that the set of indices in the shuffled dataframe are
    identical to the set of indices in the original.
    """
    df = dataframe.shuffle(reset_index=False)
    assert set(df.index) == set(dataframe.index)


@pytest.mark.functions
def test_shuffle(dataframe):
    """
    Test the shuffle function.

    This test checks that the set of dataframes has identical columns and
    number of rows.
    """
    df = dataframe.shuffle()
    assert len(df) == len(dataframe)
    assert set(df.columns) == set(dataframe.columns)
