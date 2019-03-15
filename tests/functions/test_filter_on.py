import pytest


@pytest.mark.functions
@pytest.mark.parametrize("complement,expected", [(True, 6), (False, 3)])
def test_filter_on(dataframe, complement, expected):
    df = dataframe.filter_on("a == 3", complement=complement)
    assert len(df) == expected
