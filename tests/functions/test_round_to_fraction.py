import pytest


@pytest.mark.functions
def test_round_to_nearest_half(dataframe):
    """Checks output for rounding to the nearest 1/2."""
    df = dataframe.round_to_fraction("Bell__Chart", 2)
    assert df.iloc[0, 1] == 1.0
    assert df.iloc[1, 1] == 2.5
    assert df.iloc[2, 1] == 3.0
    assert df.iloc[3, 1] == 1.0
    assert df.iloc[4, 1] == 2.5
    assert df.iloc[5, 1] == 3.0
    assert df.iloc[6, 1] == 1.0
    assert df.iloc[7, 1] == 2.5
    assert df.iloc[8, 1] == 3.0


@pytest.mark.functions
def test_round_digits(dataframe):
    """Checks rounding to the specified number of digits."""
    df = dataframe.round_to_fraction("Bell__Chart", 7, digits=3)
    assert df.iloc[0, 1] == 1.286
    assert df.iloc[1, 1] == 2.429
    assert df.iloc[2, 1] == 3.286


@pytest.mark.functions
@pytest.mark.parametrize(
    "denominator",
    [0, -5, -0.25],
)
def test_invalid_denominator_args(dataframe, denominator):
    """Ensure ValueError's are raised if denominator value passed in
    is invalid.
    """
    with pytest.raises(ValueError):
        dataframe.round_to_fraction("Bell__Chart", denominator)
