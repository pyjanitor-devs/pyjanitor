import pytest


@pytest.mark.functions
def test_min_max_scale(dataframe):
    df = dataframe.min_max_scale(column_name="a")
    assert df["a"].min() == 0
    assert df["a"].max() == 1


@pytest.mark.functions
def test_min_max_scale_custom_new_min_max(dataframe):
    df = dataframe.min_max_scale(column_name="a", feature_range=(1, 2))
    assert df["a"].min() == 1
    assert df["a"].max() == 2


@pytest.mark.functions
@pytest.mark.parametrize(
    "feature_range",
    [
        range(2),
        (1, 2, 3),
        ("1", 2),
        [1, "2"],
        ["1", "2"],
        [2, 1],
    ],
)
def test_min_max_new_min_max_errors(dataframe, feature_range):
    with pytest.raises(ValueError):
        dataframe.min_max_scale(feature_range=feature_range)
