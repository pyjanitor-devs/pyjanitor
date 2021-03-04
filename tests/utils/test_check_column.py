import pytest

from janitor.utils import check_column


@pytest.mark.utils
def test_check_column(dataframe):
    """
    check_column should return if column exist
    """
    assert check_column(dataframe, ["a"]) is None


@pytest.mark.utils
def test_check_column_single(dataframe):
    """
    Check works with a single input
    """

    assert check_column(dataframe, "a") is None

    with pytest.raises(ValueError):
        check_column(dataframe, "b")

    # should also work with non-string inputs

    with pytest.raises(ValueError):
        check_column(dataframe, 2)

    dataframe[2] = "asdf"

    assert check_column(dataframe, 2) is None


@pytest.mark.utils
def test_check_column_absent_column(dataframe):
    """
    check_column should raise an error if the column is absent.
    """
    with pytest.raises(ValueError):
        check_column(dataframe, ["b"])


@pytest.mark.utils
def test_check_column_excludes(dataframe):
    """
    check_column should return if column is absent and present is False
    """
    assert check_column(dataframe, ["b"], present=False) is None


@pytest.mark.utils
def test_check_column_absent_column_excludes(dataframe):
    """
    check_column should raise an error if the column is absent and present is
    False
    """
    with pytest.raises(ValueError):
        check_column(dataframe, ["a"], present=False)
