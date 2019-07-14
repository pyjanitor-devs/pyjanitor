import pytest
from hypothesis import given

from janitor.utils import check_column


@pytest.mark.utils
def test_check_column(dataframe):
    """
    rename_column should return true if column exist
    """
    assert check_column(dataframe, ["a"]) is None


@pytest.mark.utils
def test_check_column_absent_column(dataframe):
    """
    rename_column should raise an error if the column is absent.
    """
    with pytest.raises(ValueError):
        check_column(dataframe, ["b"])
