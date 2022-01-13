import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def process_test_df():
    "Base DataFrame"
    return pd.DataFrame(
        {"text": ["a_b_c", "c_d_e", np.nan, "f_g_h"], "numbers": range(1, 5)}
    )


@pytest.fixture
def test_returns_dataframe():
    "Base DataFrame"
    return pd.DataFrame(
        {"text": ["a1a2", "b1", "c1"], "numbers": [1, 2, 3]},
        index=["A", "B", "C"],
    )


def test_column_name_type(process_test_df):
    """Raise TypeError if `column_name` type is not `str`."""
    with pytest.raises(TypeError):
        process_test_df.process_text(["text"])


@pytest.mark.xfail(reason="new_column_names is deprecated.")
def test_new_column_names_type(process_test_df):
    """Raise TypeError if `new_column_names` type is not string or list."""
    with pytest.raises(TypeError):
        process_test_df.process_text(
            column_name="text", new_column_names={"nutext": "rar"}
        )


def test_column_name_presence(process_test_df):
    """Raise ValueError if `column_name` is not in dataframe."""
    with pytest.raises(ValueError):
        process_test_df.process_text(
            column_name="Test", string_function="lower"
        )


@pytest.mark.xfail(reason="new_column_names is deprecated.")
def test_new_column_names_presence_str(test_returns_dataframe):
    """
    Raise ValueError if `new_column_names` is a str
    and is in the dataframe.
    """
    with pytest.raises(ValueError):
        test_returns_dataframe.process_text(
            column_name="text",
            new_column_names="text",
            string_function="extractall",
            pat=r"([ab])?(\d)",
        )


@pytest.mark.xfail(reason="new_column_names is deprecated.")
def test_new_column_names_presence_list(test_returns_dataframe):
    """
    Raise ValueError if `new_column_names` is a list and at least
    one of the new names is in the dataframe.
    """
    with pytest.raises(ValueError):
        test_returns_dataframe.process_text(
            column_name="text",
            new_column_names=["numbers", "newtext"],
            string_function="extractall",
            pat=r"([ab])?(\d)",
        )


@pytest.mark.xfail(reason="merge_frame is deprecated.")
def test_merge_frame_type(test_returns_dataframe):
    """
    Raise TypeError if `merge_frame` type is not bool."""
    with pytest.raises(TypeError):
        test_returns_dataframe.process_text(
            column_name="text",
            new_column_names=["number", "newtext"],
            string_function="extractall",
            pat=r"([ab])?(\d)",
            merge_frame="True",
        )


@pytest.mark.xfail(reason="string_function must be present.")
def test_string_function_is_None(process_test_df):
    """Test that dataframe is returned if string_function is None."""
    result = process_test_df.process_text(column_name="text")
    assert_frame_equal(result, process_test_df)


def test_str_split(process_test_df):
    """Test wrapper for Pandas `str.split()` method."""

    expected = process_test_df.assign(
        text=process_test_df["text"].str.split("_")
    )

    result = process_test_df.process_text(
        column_name="text", string_function="split", pat="_"
    )

    assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="new_column_names is deprecated.")
def test_new_column_names(process_test_df):
    """
    Test that a new column name is created when
    `new_column_name` is not None.
    """
    result = process_test_df.process_text(
        column_name="text",
        new_column_names="new_text",
        string_function="slice",
        start=2,
    )
    expected = process_test_df.assign(
        new_text=process_test_df["text"].str.slice(start=2)
    )
    assert_frame_equal(result, expected)


@pytest.fixture
def no_nulls_df():
    return pd.DataFrame({"text": ["a", "b", "c", "d"], "numbers": range(1, 5)})


def test_str_cat(no_nulls_df):
    """Test outcome for Pandas `.str.cat()` method."""

    result = no_nulls_df.process_text(
        column_name="text",
        string_function="cat",
        others=["A", "B", "C", "D"],
    )

    expected = no_nulls_df.assign(
        text=no_nulls_df["text"].str.cat(others=["A", "B", "C", "D"])
    )

    assert_frame_equal(result, expected)


def test_str_cat_result_is_a_string(no_nulls_df):
    """
    Test wrapper for Pandas `.str.cat()` method
    when the outcome is a string.
    """

    result = no_nulls_df.process_text(
        column_name="text",
        string_function="cat",
    )

    expected = no_nulls_df.assign(text=no_nulls_df["text"].str.cat())

    assert_frame_equal(result, expected)


@pytest.mark.xfail(reason="new_column_names is deprecated.")
def test_str_cat_result_is_a_string_and_new_column_names(no_nulls_df):
    """
    Test wrapper for Pandas `.str.cat()` method when the outcome is a string,
    and `new_column_names` is not None.
    """

    result = no_nulls_df.process_text(
        column_name="text", string_function="cat", new_column_names="combined"
    )

    expected = no_nulls_df.assign(combined=no_nulls_df["text"].str.cat())

    assert_frame_equal(result, expected)


def test_str_get():
    """Test outcome for Pandas `.str.get()` method."""

    df = pd.DataFrame(
        {"text": ["aA", "bB", "cC", "dD"], "numbers": range(1, 5)}
    )

    expected = df.assign(text=df["text"].str.get(1))

    result = df.process_text(column_name="text", string_function="get", i=-1)

    assert_frame_equal(result, expected)


def test_str_lower():
    """Test string conversion to lowercase using `.str.lower()`."""

    df = pd.DataFrame(
        {
            "codes": range(1, 7),
            "names": [
                "Graham Chapman",
                "John Cleese",
                "Terry Gilliam",
                "Eric Idle",
                "Terry Jones",
                "Michael Palin",
            ],
        }
    )

    expected = df.assign(names=df["names"].str.lower())

    result = df.process_text(column_name="names", string_function="lower")

    assert_frame_equal(result, expected)


def test_str_wrong(process_test_df):
    """Test that an invalid Pandas string method raises an exception."""
    with pytest.raises(KeyError):
        process_test_df.process_text(
            column_name="text", string_function="invalid_function"
        )


def test_str_wrong_parameters(process_test_df):
    """Test that invalid argument for Pandas string method raises an error."""
    with pytest.raises(TypeError):
        process_test_df.process_text(
            column_name="text", string_function="split", pattern="_"
        )


@pytest.fixture
def returns_frame_1():
    return pd.DataFrame(
        {
            "ticker": [
                "spx 5/25/2001 p500",
                "spx 5/25/2001 p600",
                "spx 5/25/2001 p700",
            ]
        }
    )


@pytest.mark.xfail(reason="merge_frame is deprecated.")
def test_return_dataframe_merge_is_None(returns_frame_1):
    """
    Test that the dataframe returned when `merge_frame` is None
    is the result of the text processing, and is not merged to
    the original dataframe.
    """

    expected_output = returns_frame_1["ticker"].str.split(" ", expand=True)
    result = returns_frame_1.process_text(
        column_name="ticker", string_function="split", expand=True, pat=" "
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.xfail(reason="merge_frame is deprecated.")
def test_return_dataframe_merge_is_not_None(returns_frame_1):
    """
    Test that the dataframe returned when `merge_frame` is not None
    is a merger of the original dataframe, and the dataframe
    generated from the text processing.
    """
    expected_output = pd.concat(
        [
            returns_frame_1,
            returns_frame_1["ticker"]
            .str.split(" ", expand=True)
            .add_prefix("new_"),
        ],
        axis="columns",
    )
    result = returns_frame_1.process_text(
        column_name="ticker",
        new_column_names="new_",
        merge_frame=True,
        string_function="split",
        expand=True,
        pat=" ",
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.xfail(reason="merge_frame is deprecated.")
def test_return_dataframe_merge_is_not_None_new_column_names_is_a_list(
    returns_frame_1,
):
    """
    Test that the dataframe returned when `merge_frame` is not None
    is a merger of the original dataframe, and the dataframe
    generated from the text processing. Also, the `new_column_names`
    is a list.
    """

    expected_output = pd.concat(
        [
            returns_frame_1,
            returns_frame_1["ticker"]
            .str.split(" ", expand=True)
            .set_axis(["header1", "header2", "header3"], axis="columns"),
        ],
        axis="columns",
    )
    result = returns_frame_1.process_text(
        column_name="ticker",
        new_column_names=["header1", "header2", "header3"],
        merge_frame=True,
        string_function="split",
        expand=True,
        pat=" ",
    )
    assert_frame_equal(result, expected_output)


@pytest.mark.xfail(reason="new_column_names is deprecated.")
def test_return_dataframe_new_column_names_is_a_list_len_unequal(
    returns_frame_1,
):
    """
    Raise error if text processing returns a dataframe,
    `new_column_names` is not None, and the length of
    `new_column_names` is not equal to the length of the
    new dataframe's columns.
    """

    with pytest.raises(ValueError):
        returns_frame_1.process_text(
            column_name="ticker",
            new_column_names=["header1", "header2"],
            merge_frame=True,
            string_function="split",
            expand=True,
            pat=" ",
        )


def test_output_extractall(test_returns_dataframe):
    """
    Raise ValueError if the output is a dataframe.
    """
    with pytest.raises(ValueError):
        test_returns_dataframe.process_text(
            column_name="text",
            string_function="extractall",
            pat=r"(?P<letter>[ab])?(?P<digit>\d)",
        )


@pytest.mark.xfail(reason="merge_frame is deprecated.")
def test_output_extractall_merge_frame_is_not_None(test_returns_dataframe):
    """
    Test output when `string_function` is "extractall"
    and `merge_frame` is not None.
    """
    expected_output = test_returns_dataframe["text"].str.extractall(
        r"(?P<letter>[ab])?(?P<digit>\d)"
    )
    expected_output = test_returns_dataframe.join(
        expected_output.reset_index("match"), how="outer"
    ).set_index("match", append=True)
    result = test_returns_dataframe.process_text(
        column_name="text",
        merge_frame=True,
        string_function="extractall",
        pat=r"(?P<letter>[ab])?(?P<digit>\d)",
    )
    assert_frame_equal(result, expected_output)
