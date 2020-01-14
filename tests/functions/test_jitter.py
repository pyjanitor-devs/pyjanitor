import pytest

from janitor.functions import jitter


@pytest.mark.functions
def test_datatypes_check(dataframe):
    with pytest.raises(TypeError):
        # `scale` should be a numeric value > 0
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale="x"
        )
        # `random_state` should be an integer or 1-d array (see documentation for np.random.seed)
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=1, random_state="x"
        )
        # `clip` should only contain numeric values
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=1,
            clip=["x", 2]
        )
    with pytest.raises(ValueError):
        # `scale` should be greater than 0
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=-5
        )
        # `clip` should be a size-2 tuple of numeric values
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=1,
            clip=[-10, 10, 5]
        )
        