from janitor.testing_utils.fixtures import null_df


def test_fill_empty(null_df):
    df = null_df.fill_empty(columns=["2"], value=3)
    assert set(df.loc[:, "2"]) == set([3])


def test_fill_empty_column_string(null_df):
    df = null_df.fill_empty(columns="2", value=3)
    assert set(df.loc[:, "2"]) == set([3])
