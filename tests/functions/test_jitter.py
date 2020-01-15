import pytest

from janitor.functions import jitter


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
            column_name="a",
            dest_column_name="a_jitter",
            scale=1,
            clip=[10, 5],
        )


@pytest.mark.functions
def test_jitter(dataframe):
    # Functional test to ensure jitter runs without error
    dataframe.jitter(column_name="a", dest_column_name="a_jitter", scale=1.0)


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
