import pandas as pd
import pytest

from janitor.functions import jitter  # noqa: F401


@pytest.mark.functions
def test_datatypes_check(dataframe):
    # `scale` should be a numeric value > 0
    with pytest.raises(TypeError):
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale="x"
        )

    # `random_state` should be an integer or 1-d array
    # (see documentation for np.random.seed)
    with pytest.raises(TypeError):
        assert dataframe.jitter(
            column_name="a",
            dest_column_name="a_jitter",
            scale=1,
            random_state="x",
        )

    # `clip` should only contain numeric values
    with pytest.raises(TypeError):
        assert dataframe.jitter(
            column_name="a",
            dest_column_name="a_jitter",
            scale=1,
            clip=["x", 2],
        )

    # The column to jitter should be numeric
    with pytest.raises(TypeError):
        assert dataframe.jitter(
            column_name="cities", dest_column_name="cities_jitter", scale=1
        )

    # `scale` should be greater than 0
    with pytest.raises(ValueError):
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=-5
        )

    # `clip` should be a size-2 tuple of numeric values
    with pytest.raises(ValueError):
        assert dataframe.jitter(
            column_name="a",
            dest_column_name="a_jitter",
            scale=1,
            clip=[-10, 10, 5],
        )

    # `clip[0]` should be less than `clip[1]`
    with pytest.raises(ValueError):
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=1, clip=[10, 5]
        )


@pytest.mark.functions
def test_jitter(dataframe):
    # Functional test to ensure jitter runs without error
    dataframe.jitter(column_name="a", dest_column_name="a_jitter", scale=1.0)


@pytest.mark.functions
def test_jitter_with_nans(missingdata_df):
    # Functional test to ensure jitter runs without error if NaNs are present
    missingdata_df.jitter(
        column_name="a", dest_column_name="a_jitter", scale=1.0
    )


@pytest.mark.functions
def test_jitter_random_state(dataframe):
    # Functional test to ensure jitter runs when setting random seed
    dataframe.jitter(
        column_name="a",
        dest_column_name="a_jitter",
        scale=1.0,
        random_state=77,
    )


@pytest.mark.functions
def test_jitter_clip(dataframe):
    # Ensure clip works as intended
    df = dataframe.jitter(
        column_name="a",
        dest_column_name="a_jitter",
        scale=1.0,
        clip=[1.5, 2.5],
    )
    assert (min(df["a_jitter"]) >= 1.5) & (max(df["a_jitter"]) <= 2.5)


@pytest.mark.functions
def test_jitter_results():
    """Ensure the mean of the jittered values is approximately
    equal to the mean of the original values, and that the
    standard deviation of the jittered value is approximately
    equal to the `scale` parameter."""
    error_tolerance = 0.05  # 5%
    scale = 2.0

    df = pd.DataFrame({"original": [1] * 1000})
    results = df.jitter(
        column_name="original", dest_column_name="jittered", scale=scale
    )
    assert (
        abs(
            (results["jittered"].mean() - results["original"].mean())
            / results["original"].mean()
        )
        <= error_tolerance
    )
    assert abs((results["jittered"].std() - scale) / scale) <= error_tolerance
