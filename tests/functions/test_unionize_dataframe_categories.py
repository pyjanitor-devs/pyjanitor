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
        janitor.unionize_dataframe_categories(uniontest_df1, uniontest_df1["jerbs"])


@pytest.mark.parametrize("container", [list, tuple])
@pytest.mark.parametrize("column_names", [None, "fruits"])
def test_unionize_dataframe_categories_collection(
    uniontest_df1, uniontest_df2, uniontest_df3, container, column_names
):
    """Collections match positional calls without changing their inputs."""
    frames = container([uniontest_df1, uniontest_df2, uniontest_df3])
    originals = [df.copy(deep=True) for df in frames]
    expected = janitor.unionize_dataframe_categories(*frames, column_names=column_names)

    result = janitor.unionize_dataframe_categories(frames, column_names=column_names)

    assert isinstance(result, list)
    assert len(result) == len(frames)
    for actual, wanted, original, snapshot in zip(result, expected, frames, originals):
        assert_frame_equal(actual, wanted)
        assert_frame_equal(original, snapshot)
        assert actual is not original
    assert set(result[0]["fruits"].cat.categories) == {
        "apple",
        "banana",
        "orange",
        "grape",
        "strawberry",
        "cherry",
    }


@pytest.mark.parametrize("container", [list, tuple])
def test_unionize_dataframe_categories_invalid_collection(uniontest_df1, container):
    """Validate collection members, including nested collections."""
    for invalid in [None, uniontest_df1["jerbs"], [uniontest_df1]]:
        with pytest.raises(TypeError, match=r"^Inputs must all be dataframes\.$"):
            janitor.unionize_dataframe_categories(container([uniontest_df1, invalid]))


@pytest.mark.parametrize("args", [(), ([],), ((),)])
def test_unionize_dataframe_categories_empty(args):
    """Empty collections return the same empty list as a call with no arguments."""
    assert janitor.unionize_dataframe_categories(*args) == []


@pytest.mark.parametrize("container", [None, list, tuple])
def test_unionize_dataframe_categories_one_dataframe(uniontest_df1, container):
    """A single frame is copied with or without a surrounding collection."""
    argument = uniontest_df1 if container is None else container([uniontest_df1])
    result = janitor.unionize_dataframe_categories(argument)
    assert len(result) == 1
    assert_frame_equal(result[0], uniontest_df1)
    assert result[0] is not uniontest_df1


@pytest.mark.parametrize("container", [list, tuple])
def test_unionize_dataframe_categories_mixed_arguments(uniontest_df1, container):
    """A collection is accepted only as the sole positional argument."""
    with pytest.raises(TypeError, match=r"^Inputs must all be dataframes\.$"):
        janitor.unionize_dataframe_categories(uniontest_df1, container([uniontest_df1]))


def test_unionize_dataframe_categories(uniontest_df1, uniontest_df2, uniontest_df3):
    udf1, udf2, udf3 = janitor.unionize_dataframe_categories(
        uniontest_df1, uniontest_df2, uniontest_df3
    )

    # test categories were unioned properly

    assert set(udf1["jerbs"].dtype.categories) == set(udf2["jerbs"].dtype.categories)

    assert set(udf1["jerbs"].dtype.categories) == set(udf3["jerbs"].dtype.categories)

    assert set(udf1["fruits"].dtype.categories) == set(udf3["fruits"].dtype.categories)

    assert set(udf1["fruits"].dtype.categories) == set(udf3["fruits"].dtype.categories)

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
    udf1, udf2 = janitor.unionize_dataframe_categories(uniontest_df1, uniontest_df2)

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

    assert set(udf1["fruits"].dtype.categories) == set(udf2["fruits"].dtype.categories)

    assert set(udf1["fruits"].dtype.categories) == set(udf3["fruits"].dtype.categories)

    # check that jerbs did not when we didn't want it to

    assert not (
        set(udf1["jerbs"].dtype.categories) == set(udf2["jerbs"].dtype.categories)
    )
