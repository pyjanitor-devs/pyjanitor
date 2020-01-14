import pytest

from janitor.functions import jitter


@pytest.mark.functions
def test_datatypes_check(dataframe):
    with pytest.raises(TypeError):
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale="x"
        )
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=1, random_state="x"
        )
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=1,
            clip=["x", 2]
        )
    with pytest.raises(ValueError):
        assert dataframe.jitter(
            column_name="a", dest_column_name="a_jitter", scale=1,
            clip=[-10, 10, 5]
        )
        