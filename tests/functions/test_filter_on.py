import pytest


@pytest.mark.functions
@pytest.mark.parametrize("complement,expected", [(True, 6), (False, 3)])
def test_filter_on(dataframe, complement, expected):
    df = dataframe.filter_on("a == 3", complement=complement)
    assert len(df) == expected


@pytest.mark.functions
@pytest.mark.parametrize("complement,expected", [(True, 3), (False, 6)])
def test_filter_on_with_multiple_criteria(dataframe, complement, expected):
    df = dataframe.filter_on("(a == 3) | (a == 1)", complement=complement)
    assert len(df) == expected
