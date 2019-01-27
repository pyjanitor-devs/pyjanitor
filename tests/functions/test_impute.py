from janitor.testing_utils.fixtures import missingdata_df

def test_impute_single_value(missingdata_df):
    df = missingdata_df.impute("a", 5)
    assert set(df["a"]) == set([1, 2, 5])


@pytest.mark.parametrize(
    "statistic,expected",
    [
        ("mean", set([1, 2, 1.5])),
        ("average", set([1, 2, 1.5])),
        ("median", set([1, 2, 1.5])),
        ("mode", set([1, 2])),
        ("min", set([1, 2])),
        ("minimum", set([1, 2])),
        ("max", set([1, 2])),
        ("maximum", set([1, 2])),
    ],
)
def test_impute_statistical(missingdata_df, statistic, expected):
    df = missingdata_df.impute("a", statistic=statistic)
    assert set(df["a"]) == expected
