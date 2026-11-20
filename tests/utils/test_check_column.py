import pytest

from janitor.utils import check_column


@pytest.mark.utils
def test_check_column(dataframe):
    """
    check_column should return if column exist
    """
    assert check_column(dataframe, ["a"]) == ["a"]


@pytest.mark.utils
def test_check_column_single(dataframe):
    """
    Check works with a single input
    """

    assert check_column(dataframe, "a") == ["a"]

    with pytest.raises(ValueError):
        check_column(dataframe, "b")

    # should also work with non-string inputs

    with pytest.raises(ValueError):
        check_column(dataframe, 2)

    dataframe[2] = "asdf"

    assert check_column(dataframe, 2) == [2]


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
    assert check_column(dataframe, ["b"], present=False) == ["b"]


@pytest.mark.utils
def test_check_column_absent_column_excludes(dataframe):
    """
    check_column should raise an error if the column is absent and present is
    False
    """
    with pytest.raises(ValueError):
        check_column(dataframe, ["a"], present=False)


@pytest.mark.utils
def test_check_column_generator_is_reusable(dataframe):
    """One-shot iterables should be returned as a reusable list."""
    columns_map = map(lambda x: x, ["a"])
    result = check_column(dataframe, columns_map)

    assert result == ["a"]
    assert list(result) == ["a"]

    names = (name for name in ["missing", "a"])
    with pytest.raises(ValueError, match="missing not present"):
        check_column(dataframe, names)
    assert list(names) == ["a"]

    names = (name for name in ["a", "missing"])
    with pytest.raises(ValueError, match="a already present"):
        check_column(dataframe, names, present=False)
    assert check_column(dataframe, names, present=False) == ["missing"]
