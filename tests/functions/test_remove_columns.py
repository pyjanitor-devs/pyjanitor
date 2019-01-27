from janitor.testing_utils.fixtures import dataframe


def test_remove_columns(dataframe):
    df = dataframe.remove_columns(columns=["a"])
    assert len(df.columns) == 4
