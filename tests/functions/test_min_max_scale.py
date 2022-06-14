import pandas as pd
import pytest


@pytest.mark.functions
@pytest.mark.parametrize(
    "df, column_name, jointly, excepted",
    [
        # test default parameter
        (
            pd.DataFrame({"a": [5, 10], "b": [0, 5]}),
            None,
            True,
            pd.DataFrame({"a": [0.5, 1], "b": [0, 0.5]}),
        ),
        # test default parameter
        (
            pd.DataFrame({"a": [5, 10], "b": [0, 5]}),
            None,
            False,
            pd.DataFrame({"a": [0, 1.0], "b": [0, 1.0]}),
        ),
        # test list condition
        (
            pd.DataFrame({"a": [5, 10], "b": [0, 5]}),
            ["a", "b"],
            True,
            pd.DataFrame({"a": [0.5, 1.0], "b": [0, 0.5]}),
        ),
        # test list condition
        (
            pd.DataFrame({"a": [5, 10], "b": [0, 5]}),
            ["a", "b"],
            False,
            pd.DataFrame({"a": [0, 1.0], "b": [0, 1.0]}),
        ),
        # test Index condition
        (
            pd.DataFrame({"a": [5, 10], "b": [0, 5]}),
            pd.Index(["a", "b"]),
            False,
            pd.DataFrame({"a": [0, 1.0], "b": [0, 1.0]}),
        ),
        # test Index condition
        (
            pd.DataFrame({"a": [5, 10], "b": [0, 5]}),
            pd.Index(["a", "b"]),
            True,
            pd.DataFrame({"a": [0.5, 1], "b": [0, 0.5]}),
        ),
        # test str condition
        (
            pd.DataFrame({"a": [5, 10], "b": [0, 5]}),
            "a",
            True,
            pd.DataFrame({"a": [0, 1.0], "b": [0, 5]}),
        ),
        (
            pd.DataFrame({"a": [5, 10], "b": [0, 5]}),
            "a",
            False,
            pd.DataFrame({"a": [0, 1.0], "b": [0, 5]}),
        ),
        # test int condition
        (
            pd.DataFrame({1: [5, 10], "b": [0, 5]}),
            1,
            True,
            pd.DataFrame({1: [0, 1.0], "b": [0, 5]}),
        ),
        # test int condition
        (
            pd.DataFrame({1: [5, 10], "b": [0, 5]}),
            1,
            False,
            pd.DataFrame({1: [0, 1.0], "b": [0, 5]}),
        ),
    ],
)
def test_min_max_scale_column_name_type(df, column_name, jointly, excepted):
    result = df.min_max_scale(column_name=column_name, jointly=jointly)

    assert result.equals(excepted)


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
