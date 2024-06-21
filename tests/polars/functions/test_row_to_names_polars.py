import polars as pl
import pytest

import janitor.polars  # noqa: F401

df = pl.DataFrame(
    {
        "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
        "decorated-elephant": [1, 2, 3] * 3,
        "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
        "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    }
)


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_separator_type(df):
    """
    Raise if separator is not a string
    """
    with pytest.raises(TypeError, match="separator should be.+"):
        df.row_to_names([1, 2], separator=1)


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_numbers_type(df):
    """
    Raise if row_numbers is not an int/list
    """
    with pytest.raises(TypeError, match="row_numbers should be.+"):
        df.row_to_names({1, 2})


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_numbers_list_type(df):
    """
    Raise if row_numbers is a list
    and one of the entries is not an integer.
    """
    with pytest.raises(
        TypeError, match="entry in the row_numbers argument should be.+"
    ):
        df.row_to_names(["1", 2])


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names(df):
    df = df.row_to_names(2)
    assert df.columns[0] == "3.2346125"
    assert df.columns[1] == "3"
    assert df.columns[2] == "lion"
    assert df.columns[3] == "Basel"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_single_list(df):
    "Test output if row_numbers is a list, and contains a single item."
    df = df.row_to_names([2])
    assert df.columns[0] == "3.2346125"
    assert df.columns[1] == "3"
    assert df.columns[2] == "lion"
    assert df.columns[3] == "Basel"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_list(df):
    "Test output if row_numbers is a list."
    df = df.row_to_names([1, 2])
    assert df.columns[0] == "2.456234_3.2346125"
    assert df.columns[1] == "2_3"
    assert df.columns[2] == "leopard_lion"
    assert df.columns[3] == "Shanghai_Basel"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_delete_this_row(df):
    df = df.row_to_names(2, remove_rows=True)
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    assert df.to_series(0)[0] == 1.234_523_45
    assert df.to_series(1)[0] == 1
    assert df.to_series(2)[0] == "rabbit"
    assert df.to_series(3)[0] == "Cambridge"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_list_delete_this_row(df):
    df = df.row_to_names([2], remove_rows=True)
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    assert df.to_series(0)[0] == 1.234_523_45
    assert df.to_series(1)[0] == 1
    assert df.to_series(2)[0] == "rabbit"
    assert df.to_series(3)[0] == "Cambridge"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_delete_above(df):
    df = df.row_to_names(2, remove_rows_above=True)
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    assert df.to_series(0)[0] == 3.234_612_5
    assert df.to_series(1)[0] == 3
    assert df.to_series(2)[0] == "lion"
    assert df.to_series(3)[0] == "Basel"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_delete_above_list(df):
    "Test output if row_numbers is a list"
    df = df.row_to_names([2, 3], remove_rows_above=True)
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    assert df.to_series(0)[0] == 3.234_612_5
    assert df.to_series(1)[0] == 3
    assert df.to_series(2)[0] == "lion"
    assert df.to_series(3)[0] == "Basel"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_delete_above_delete_rows(df):
    """
    Test output for remove_rows=True
    and remove_rows_above=True
    """
    df = df.row_to_names([2, 3], remove_rows=True, remove_rows_above=True)
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    assert df.to_series(0)[0] == 2.456234
    assert df.to_series(1)[0] == 2
    assert df.to_series(2)[0] == "leopard"
    assert df.to_series(3)[0] == "Shanghai"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_delete_above_delete_rows_scalar(df):
    """
    Test output for remove_rows=True
    and remove_rows_above=True
    """
    df = df.row_to_names(2, remove_rows=True, remove_rows_above=True)
    if isinstance(df, pl.LazyFrame):
        df = df.collect()
    assert df.to_series(0)[0] == 1.23452345
    assert df.to_series(1)[0] == 1
    assert df.to_series(2)[0] == "rabbit"
    assert df.to_series(3)[0] == "Cambridge"


@pytest.mark.parametrize("df", [df, df.lazy()])
def test_row_to_names_delete_above_list_non_consecutive(df):
    "Raise if row_numbers is a list, but non consecutive"
    msg = "The remove_rows_above argument is applicable "
    msg += "only if the row_numbers argument is an integer, "
    msg += "or the integers in a list are consecutive increasing, "
    msg += "with a difference of 1."
    with pytest.raises(ValueError, match=msg):
        df.row_to_names([1, 3], remove_rows_above=True)
