import pytest

from janitor.errors import JanitorError


@pytest.mark.functions
def test_concatenate_columns(dataframe):
    df = dataframe.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    )
    assert "index" in df.columns


@pytest.mark.functions
def test_concatenate_columns_null_values(missingdata_df):
    df = missingdata_df.concatenate_columns(
        column_names=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
        ignore_empty=True,
    )
    expected_values = ["1.0-1", "2.0-2", "3"] * 3
    assert expected_values == df["index"].tolist()


@pytest.mark.functions
@pytest.mark.parametrize("column_names", [["a"], []])
def test_concatenate_columns_errors(dataframe, column_names):
    with pytest.raises(
        JanitorError, match="At least two columns must be specified"
    ):
        dataframe.concatenate_columns(
            column_names=column_names, new_column_name="index"
        )
