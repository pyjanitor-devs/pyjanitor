import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import janitor


@pytest.fixture
def uniontest_df1():
    return pd.DataFrame(
        dict(
            jerbs=pd.Categorical(["fireman", "programmer", "astronaut"]),
            fruits=pd.Categorical(["apple", "banana", "orange"]),
            data=[1, 2, 3],
            df1_exclusive="somedata",
        )
    )


@pytest.fixture
def uniontest_df2():
    return pd.DataFrame(
        dict(
            jerbs=pd.Categorical(["fireman", "actor", "astronaut"]),
            fruits=pd.Categorical(["grape", "strawberry", "cherry"]),
            data=[4, 5, 6],
            df2_exclusive="otherdata",
            animals=pd.Categorical(["bear", "tiger", "sloth"]),
        )
    )


@pytest.fixture
def uniontest_df3():
    return pd.DataFrame(
        dict(
            jerbs=pd.Categorical(["salesman", "actor", "programmer"]),
            fruits=pd.Categorical(["grape", "banana", "cherry"]),
            data=[7, 8, 9],
            df3_exclusive="evenmoredata",
            animals=pd.Categorical(["bear", "capybara", "sloth"]),
        )
    )


def test_unionize_dataframe_categories_type(uniontest_df1):
    with pytest.raises(TypeError):
        janitor.unionize_dataframe_categories(
            uniontest_df1, uniontest_df1["jerbs"]
        )


def test_unionize_dataframe_categories(
    uniontest_df1, uniontest_df2, uniontest_df3
):
    udf1, udf2, udf3 = janitor.unionize_dataframe_categories(
        uniontest_df1, uniontest_df2, uniontest_df3
    )

    # test categories were unioned properly

    assert set(udf1["jerbs"].dtype.categories) == set(
        udf2["jerbs"].dtype.categories
    )

    assert set(udf1["jerbs"].dtype.categories) == set(
        udf3["jerbs"].dtype.categories
    )

    assert set(udf1["fruits"].dtype.categories) == set(
        udf3["fruits"].dtype.categories
    )

    assert set(udf1["fruits"].dtype.categories) == set(
        udf3["fruits"].dtype.categories
    )

    assert set(udf2["animals"].dtype.categories) == set(
        udf3["animals"].dtype.categories
    )

    # test columns did not bleed in

    assert "df2_exclusive" not in udf1.columns
    assert "df2_exclusive" not in udf3.columns
    assert "df1_exclusive" not in udf2.columns
    assert "df1_exclusive" not in udf3.columns
    assert "df3_exclusive" not in udf1.columns
    assert "df3_exclusive" not in udf2.columns

    assert "animals" not in udf1.columns

    # test that pd.concat now does not destroy categoricals

    # NOTE: 'animals' column will not be preserved as categorical after concat
    # because it is not present in df1. Instead, this column will be set to
    # object with NaN filled in for df1 missing values.

    udf = pd.concat([udf1, udf2, udf3], ignore_index=True)

    assert isinstance(udf["jerbs"].dtype, pd.CategoricalDtype)
    assert isinstance(udf["fruits"].dtype, pd.CategoricalDtype)

    # test that the data is the same

    assert_frame_equal(udf1, uniontest_df1, check_categorical=False)
    assert_frame_equal(udf2, uniontest_df2, check_categorical=False)
    assert_frame_equal(udf3, uniontest_df3, check_categorical=False)


def test_unionize_dataframe_categories_original_preservation(
    uniontest_df1, uniontest_df2
):
    udf1, udf2 = janitor.unionize_dataframe_categories(
        uniontest_df1, uniontest_df2
    )

    assert not (
        set(uniontest_df1["fruits"].dtype.categories)
        == set(udf1["fruits"].dtype.categories)
    )


def test_unionize_dataframe_categories_single(
    uniontest_df1, uniontest_df2, uniontest_df3
):
    udf1, udf2, udf3 = janitor.unionize_dataframe_categories(
        uniontest_df1, uniontest_df2, uniontest_df3, column_names="fruits"
    )

    # check that fruits did get unionized

    assert set(udf1["fruits"].dtype.categories) == set(
        udf2["fruits"].dtype.categories
    )

    assert set(udf1["fruits"].dtype.categories) == set(
        udf3["fruits"].dtype.categories
    )

    # check that jerbs did not when we didn't want it to

    assert not (
        set(udf1["jerbs"].dtype.categories)
        == set(udf2["jerbs"].dtype.categories)
    )
