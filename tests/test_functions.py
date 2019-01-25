"""Tests for pyjanitor."""

import numpy as np
import pandas as pd
import pytest
import requests

import janitor
from janitor import (
    # change_type,
    # clean_names,
    coalesce,
    concatenate_columns,
    convert_excel_date,
    convert_matlab_date,
    deconcatenate_column,
    encode_categorical,
    expand_column,
    filter_on,
    filter_string,
    get_dupes,
    remove_columns,
    remove_empty,
)
from janitor.errors import JanitorError
import janitor.finance


@pytest.fixture
def dataframe():
    data = {
        "a": [1, 2, 3] * 3,
        "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
        "decorated-elephant": [1, 2, 3] * 3,
        "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
        "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def null_df():
    np.random.seed([3, 1415])
    df = pd.DataFrame(np.random.choice((1, np.nan), (10, 2)))
    df["2"] = np.nan * 10
    df["3"] = np.nan * 10
    return df


@pytest.fixture
def multiindex_dataframe():
    data = {
        ("a", "b"): [1, 2, 3],
        ("Bell__Chart", "Normal  Distribution"): [1, 2, 3],
        ("decorated-elephant", "r.i.p-rhino"): [1, 2, 3],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def multiindex_with_missing_dataframe():
    data = {
        ("a", ""): [1, 2, 3],
        ("", "Normal  Distribution"): [1, 2, 3],
        ("decorated-elephant", "r.i.p-rhino :'("): [1, 2, 3],
    }
    df = pd.DataFrame(data)
    return df


@pytest.fixture
def multiindex_with_missing_3level_dataframe():
    data = {
        ("a", "", ""): [1, 2, 3],
        ("", "Normal  Distribution", "Hypercuboid (???)"): [1, 2, 3],
        ("decorated-elephant", "r.i.p-rhino :'(", "deadly__flamingo"): [
            1,
            2,
            3,
        ],
    }
    df = pd.DataFrame(data)
    return df


def test_clean_names_functional(dataframe):
    df = dataframe.clean_names()
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals@#$%^",
        "cities",
    ]
    assert set(df.columns) == set(expected_columns)


def test_clean_names_method_chain(dataframe):
    df = dataframe.clean_names()
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals@#$%^",
        "cities",
    ]
    assert set(df.columns) == set(expected_columns)


# def test_clean_names_pipe(dataframe):
#     df = dataframe.pipe(clean_names)
#     expected_columns = [
#         "a",
#         "bell_chart",
#         "decorated_elephant",
#         "animals@#$%^",
#         "cities",
#     ]
#     assert set(df.columns) == set(expected_columns)
#


def test_clean_names_special_characters(dataframe):
    df = dataframe.clean_names(remove_special=True)
    expected_columns = [
        "a",
        "bell_chart",
        "decorated_elephant",
        "animals",
        "cities",
    ]
    assert set(df.columns) == set(expected_columns)


def test_clean_names_uppercase(dataframe):
    df = dataframe.clean_names(case_type="upper", remove_special=True)
    expected_columns = [
        "A",
        "BELL_CHART",
        "DECORATED_ELEPHANT",
        "ANIMALS",
        "CITIES",
    ]
    assert set(df.columns) == set(expected_columns)


def test_clean_names_original_columns(dataframe):
    df = dataframe.clean_names(preserve_original_columns=True)
    expected_columns = [
        "a",
        "Bell__Chart",
        "decorated-elephant",
        "animals@#$%^",
        "cities",
    ]
    assert set(df.original_columns) == set(expected_columns)


def test_remove_empty(null_df):
    df = remove_empty(null_df)
    assert df.shape == (8, 2)


def test_get_dupes():
    df = pd.DataFrame()
    df["a"] = [1, 2, 1]
    df["b"] = [1, 2, 1]
    df_dupes = get_dupes(df)
    assert df_dupes.shape == (2, 2)

    df2 = pd.DataFrame()
    df2["a"] = [1, 2, 3]
    df2["b"] = [1, 2, 3]
    df2_dupes = get_dupes(df2)
    assert df2_dupes.shape == (0, 2)


def test_encode_categorical():
    df = pd.DataFrame()
    df["class_label"] = ["test1", "test2", "test1", "test2"]
    df["numbers"] = [1, 2, 3, 2]
    df = encode_categorical(df, "class_label")
    assert df["class_label"].dtypes == "category"


def test_encode_categorical_missing_column(dataframe):
    with pytest.raises(AssertionError):
        dataframe.encode_categorical("aloha")


def test_encode_categorical_missing_columns(dataframe):
    with pytest.raises(AssertionError):
        dataframe.encode_categorical(["animals@#$%^", "cities", "aloha"])


def test_encode_categorical_invalid_input(dataframe):
    with pytest.raises(JanitorError):
        dataframe.encode_categorical(1)


def test_get_features_targets(dataframe):
    dataframe = dataframe.clean_names()
    X, y = dataframe.get_features_targets(target_columns="bell_chart")
    assert X.shape == (9, 4)
    assert y.shape == (9,)


def test_get_features_targets_multi_features(dataframe):
    dataframe = dataframe.clean_names()
    X, y = dataframe.get_features_targets(
        feature_columns=["animals@#$%^", "cities"], target_columns="bell_chart"
    )
    assert X.shape == (9, 2)
    assert y.shape == (9,)


def test_get_features_target_multi_columns(dataframe):
    dataframe = dataframe.clean_names()
    X, y = dataframe.get_features_targets(target_columns=["a", "bell_chart"])
    assert X.shape == (9, 3)
    assert y.shape == (9, 2)


def test_rename_column(dataframe):
    df = dataframe.clean_names().rename_column("a", "index")
    assert set(df.columns) == set(
        ["index", "bell_chart", "decorated_elephant", "animals@#$%^", "cities"]
    )  # noqa: E501


def test_reorder_columns(dataframe):
    # sanity checking of inputs

    # input is not a list or pd.Index
    with pytest.raises(TypeError):
        dataframe.reorder_columns("a")

    # one of the columns is not present in the DataFrame
    with pytest.raises(IndexError):
        dataframe.reorder_columns(["notpresent"])

    # reordering functionality

    # sanity check when desired order matches current order
    # this also tests whether the function can take Pandas Index objects
    assert all(
        dataframe.reorder_columns(dataframe.columns).columns
        == dataframe.columns
    )

    # when columns are list & not all columns of DataFrame are included
    assert all(
        dataframe.reorder_columns(["animals@#$%^", "Bell__Chart"]).columns
        == ["animals@#$%^", "Bell__Chart", "a", "decorated-elephant", "cities"]
    )


def test_coalesce():
    df = pd.DataFrame(
        {"a": [1, np.nan, 3], "b": [2, 3, 1], "c": [2, np.nan, 9]}
    )

    df = coalesce(df, ["a", "b", "c"], "a")
    assert df.shape == (3, 1)
    assert pd.isnull(df).sum().sum() == 0


def test_convert_excel_date():
    df = pd.read_excel("examples/dirty_data.xlsx").clean_names()
    df = convert_excel_date(df, "hire_date")

    assert df["hire_date"].dtype == "M8[ns]"


def test_convert_matlab_date():
    mlab = [
        733_301.0,
        729_159.0,
        734_471.0,
        737_299.563_296_356_5,
        737_300.000_000_000_0,
    ]
    df = pd.DataFrame(mlab, columns=["dates"])
    df = convert_matlab_date(df, "dates")

    assert df["dates"].dtype == "M8[ns]"


def test_fill_empty(null_df):
    df = null_df.fill_empty(columns=["2"], value=3)
    assert set(df.loc[:, "2"]) == set([3])


def test_fill_empty_column_string(null_df):
    df = null_df.fill_empty(columns="2", value=3)
    assert set(df.loc[:, "2"]) == set([3])


def test_single_column_label_encode():
    df = pd.DataFrame(
        {"a": ["hello", "hello", "sup"], "b": [1, 2, 3]}
    ).label_encode(columns="a")
    assert "a_enc" in df.columns


def test_single_column_fail_label_encode():
    with pytest.raises(AssertionError):
        df = pd.DataFrame(
            {"a": ["hello", "hello", "sup"], "b": [1, 2, 3]}
        ).label_encode(
            columns="c"
        )  # noqa: 841


def test_multicolumn_label_encode():
    df = pd.DataFrame(
        {
            "a": ["hello", "hello", "sup"],
            "b": [1, 2, 3],
            "c": ["aloha", "nihao", "nihao"],
        }
    ).label_encode(columns=["a", "c"])
    assert "a_enc" in df.columns
    assert "c_enc" in df.columns


def test_label_encode_invalid_input(dataframe):
    with pytest.raises(JanitorError):
        dataframe.label_encode(1)


# def test_multiindex_clean_names_functional(multiindex_dataframe):
#     df = multiindex_dataframe.clean_names()

#     levels = [
#         ["a", "bell_chart", "decorated_elephant"],
#         ["b", "normal_distribution", "r_i_p_rhino_"],
#     ]

#     labels = [[1, 0, 2], [1, 0, 2]]

#     expected_columns = pd.MultiIndex(levels=levels, labels=labels)
#     assert set(df.columns) == set(expected_columns)


def test_multiindex_clean_names(multiindex_dataframe):
    df = multiindex_dataframe.clean_names()

    levels = [
        ["a", "bell_chart", "decorated_elephant"],
        ["b", "normal_distribution", "r_i_p_rhino"],
    ]

    labels = [[0, 1, 2], [0, 1, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


# def test_multiindex_clean_names_pipe(multiindex_dataframe):
#     df = multiindex_dataframe.pipe(clean_names)
#
#     levels = [
#         ["a", "bell_chart", "decorated_elephant"],
#         ["b", "normal_distribution", "r_i_p_rhino_"],
#     ]
#
#     labels = [[0, 1, 2], [0, 1, 2]]
#
#     expected_columns = pd.MultiIndex(levels=levels, labels=labels)
#     assert set(df.columns) == set(expected_columns)


@pytest.mark.test
@pytest.mark.parametrize(
    "strip_underscores", ["both", True, "right", "r", "left", "l"]
)
def test_clean_names_strip_underscores(
    multiindex_dataframe, strip_underscores
):
    if strip_underscores in ["right", "r"]:
        df = multiindex_dataframe.rename(columns=lambda x: x + "_")
    elif strip_underscores in ["left", "l"]:
        df = multiindex_dataframe.rename(columns=lambda x: "_" + x)
    else:
        df = multiindex_dataframe
    df = df.clean_names(strip_underscores=strip_underscores)

    levels = [
        ["a", "bell_chart", "decorated_elephant"],
        ["b", "normal_distribution", "r_i_p_rhino"],
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_incorrect_strip_underscores(multiindex_dataframe):
    with pytest.raises(JanitorError):
        df = multiindex_dataframe.clean_names(
            strip_underscores="hello"
        )  # noqa: E501, F841


def test_clean_names_preserve_case_true(multiindex_dataframe):
    # df = multiindex_dataframe.rename(columns=lambda x: "_" + x)
    df = multiindex_dataframe.clean_names(case_type="preserve")

    levels = [
        ["a", "Bell_Chart", "decorated_elephant"],
        ["b", "Normal_Distribution", "r_i_p_rhino"],
    ]

    labels = [[1, 0, 2], [1, 0, 2]]

    expected_columns = pd.MultiIndex(levels=levels, labels=labels)
    assert set(df.columns) == set(expected_columns)


def test_expand_column():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data)
    expanded = expand_column(df, "col1", sep=", ", concat=False)
    assert expanded.shape[1] == 6


def test_expand_and_concat():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data).expand_column("col1", sep=", ", concat=True)
    assert df.shape[1] == 8


def test_concatenate_columns(dataframe):
    df = concatenate_columns(
        dataframe,
        columns=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    )
    assert "index" in df.columns


def test_deconcatenate_column(dataframe):
    df = concatenate_columns(
        dataframe,
        columns=["a", "decorated-elephant"],
        sep="-",
        new_column_name="index",
    )
    df = deconcatenate_column(
        df, column="index", new_column_names=["A", "B"], sep="-"
    )
    assert "A" in df.columns
    assert "B" in df.columns


def test_filter_string(dataframe):
    df = filter_string(dataframe, column="animals@#$%^", search_string="bbit")
    assert len(df) == 3


def test_filter_string_complement(dataframe):
    df = filter_string(
        dataframe, column="cities", search_string="hang", complement=True
    )
    assert len(df) == 6


def test_filter_on(dataframe):
    df = filter_on(dataframe, "a == 3")
    assert len(df) == 3


def test_filter_on_complement(dataframe):
    df = filter_on(dataframe, "a == 3", complement=True)
    assert len(df) == 6


def test_remove_columns(dataframe):
    df = remove_columns(dataframe, columns=["a"])
    assert len(df.columns) == 4


def test_change_type(dataframe):
    df = dataframe.change_type(column="a", dtype=float)
    assert df["a"].dtype == float


def test_add_column(dataframe):

    # sanity checking of inputs

    # col_name wasn't a string
    with pytest.raises(TypeError):
        dataframe.add_column(col_name=42, value=42)

    # column already exists
    with pytest.raises(ValueError):
        dataframe.add_column("a", 42)

    # too many values for dataframe num rows:
    with pytest.raises(ValueError):
        dataframe.add_column("toomany", np.ones(100))

    # functionality testing

    # column appears in DataFrame
    df = dataframe.add_column("fortytwo", 42)
    assert "fortytwo" in df.columns

    # values are correct in dataframe for scalar
    series = pd.Series([42] * len(dataframe))
    series.name = "fortytwo"
    pd.testing.assert_series_equal(df["fortytwo"], series)

    # scalar values are correct for strings
    # also, verify sanity check excludes strings, which have a length:

    df = dataframe.add_column("fortythousand", "test string")
    series = pd.Series(["test string"] * len(dataframe))
    series.name = "fortythousand"
    pd.testing.assert_series_equal(df["fortythousand"], series)

    # values are correct in dataframe for iterable
    vals = np.linspace(0, 43, len(dataframe))
    df = dataframe.add_column("fortythree", vals)
    series = pd.Series(vals)
    series.name = "fortythree"
    pd.testing.assert_series_equal(df["fortythree"], series)

    # fill_remaining works - iterable shorter than DataFrame
    vals = [0, 42]
    target = [0, 42] * 4 + [0]
    df = dataframe.add_column("fill_in_iterable", vals, fill_remaining=True)
    series = pd.Series(target)
    series.name = "fill_in_iterable"
    pd.testing.assert_series_equal(df["fill_in_iterable"], series)

    # fill_remaining works - value is scalar
    vals = 42
    df = dataframe.add_column("fill_in_scalar", vals, fill_remaining=True)
    series = pd.Series([42] * len(df))
    series.name = "fill_in_scalar"
    pd.testing.assert_series_equal(df["fill_in_scalar"], series)


def test_add_columns(dataframe):
    # sanity checking is pretty much handled in test_add_column

    # multiple column addition with scalar and iterable

    x_vals = 42
    y_vals = np.linspace(0, 42, len(dataframe))

    df = dataframe.add_columns(x=x_vals, y=y_vals)

    series = pd.Series([x_vals] * len(dataframe))
    series.name = "x"
    pd.testing.assert_series_equal(df["x"], series)

    series = pd.Series(y_vals)
    series.name = "y"
    pd.testing.assert_series_equal(df["y"], series)


def test_limit_column_characters(dataframe):
    df = dataframe.limit_column_characters(1)
    assert df.columns[0] == "a"
    assert df.columns[1] == "B"
    assert df.columns[2] == "d"
    assert df.columns[3] == "a_1"
    assert df.columns[4] == "c"


def test_limit_column_characters_different_positions(dataframe):
    df = dataframe
    df.columns = ["first", "first", "second", "second", "first"]
    df.limit_column_characters(3)

    assert df.columns[0] == "fir"
    assert df.columns[1] == "fir_1"
    assert df.columns[2] == "sec"
    assert df.columns[3] == "sec_1"
    assert df.columns[4] == "fir_2"


def test_limit_column_characters_different_positions_different_separator(
    dataframe
):
    df = dataframe
    df.columns = ["first", "first", "second", "second", "first"]
    df.limit_column_characters(3, ".")

    assert df.columns[0] == "fir"
    assert df.columns[1] == "fir.1"
    assert df.columns[2] == "sec"
    assert df.columns[3] == "sec.1"
    assert df.columns[4] == "fir.2"


def test_limit_column_characters_all_unique(dataframe):
    df = dataframe.limit_column_characters(2)
    assert df.columns[0] == "a"
    assert df.columns[1] == "Be"
    assert df.columns[2] == "de"
    assert df.columns[3] == "an"
    assert df.columns[4] == "ci"


def test_add_column_single_value(dataframe):
    df = dataframe.add_column("city_pop", 100)
    assert df.city_pop.mean() == 100


def test_add_column_iterator_repeat(dataframe):
    df = dataframe.add_column("city_pop", range(3), fill_remaining=True)
    assert df.city_pop.iloc[0] == 0
    assert df.city_pop.iloc[1] == 1
    assert df.city_pop.iloc[2] == 2
    assert df.city_pop.iloc[3] == 0
    assert df.city_pop.iloc[4] == 1
    assert df.city_pop.iloc[5] == 2


def test_add_column_raise_error(dataframe):
    with pytest.raises(Exception):
        dataframe.add_column("cities", 1)


def test_add_column_iterator_repeat_subtraction(dataframe):
    df = dataframe.add_column("city_pop", dataframe.a - dataframe.a)
    assert df.city_pop.sum() == 0
    assert df.city_pop.iloc[0] == 0


def test_row_to_names(dataframe):
    df = dataframe.row_to_names(2)
    assert df.columns[0] == 3
    assert df.columns[1] == 3.234_612_5
    assert df.columns[2] == 3
    assert df.columns[3] == "lion"
    assert df.columns[4] == "Basel"


def test_row_to_names_delete_this_row(dataframe):
    df = dataframe.row_to_names(2, remove_row=True)
    assert df.iloc[2, 0] == 1
    assert df.iloc[2, 1] == 1.234_523_45
    assert df.iloc[2, 2] == 1
    assert df.iloc[2, 3] == "rabbit"
    assert df.iloc[2, 4] == "Cambridge"


def test_row_to_names_delete_above(dataframe):
    df = dataframe.row_to_names(2, remove_rows_above=True)
    assert df.iloc[0, 0] == 3
    assert df.iloc[0, 1] == 3.234_612_5
    assert df.iloc[0, 2] == 3
    assert df.iloc[0, 3] == "lion"
    assert df.iloc[0, 4] == "Basel"


def test_round_to_nearest_half(dataframe):
    df = dataframe.round_to_fraction("Bell__Chart", 2)
    assert df.iloc[0, 1] == 1.0
    assert df.iloc[1, 1] == 2.5
    assert df.iloc[2, 1] == 3.0
    assert df.iloc[3, 1] == 1.0
    assert df.iloc[4, 1] == 2.5
    assert df.iloc[5, 1] == 3.0
    assert df.iloc[6, 1] == 1.0
    assert df.iloc[7, 1] == 2.5
    assert df.iloc[8, 1] == 3.0


def test_make_currency_api_request():
    r = requests.get("https://api.exchangeratesapi.io")
    assert r.status_code == 200


def test_make_new_currency_col(dataframe):
    df = dataframe.convert_currency("a", "USD", "USD", make_new_column=True)
    assert all(df["a"] == df["a_USD"])


def test_transform_column(dataframe):
    # replacing the data of the original column

    df = dataframe.transform_column("a", np.log10)
    expected = pd.Series(np.log10([1, 2, 3] * 3))
    expected.name = "a"
    pd.testing.assert_series_equal(df["a"], expected)


def test_transform_column_with_dest(dataframe):
    # creating a new destination column

    expected_df = dataframe.assign(a_log10=np.log10(dataframe["a"]))

    df = dataframe.copy().transform_column(
        "a", np.log10, dest_col_name="a_log10"
    )

    pd.testing.assert_frame_equal(df, expected_df)


def test_min_max_scale(dataframe):
    df = dataframe.min_max_scale(col_name="a")
    assert df["a"].min() == 0
    assert df["a"].max() == 1


def test_min_max_scale_custom_new_min_max(dataframe):
    df = dataframe.min_max_scale(col_name="a", new_min=1, new_max=2)
    assert df["a"].min() == 1
    assert df["a"].max() == 2


def test_min_max_old_min_max_errors(dataframe):
    with pytest.raises(ValueError):
        df = dataframe.min_max_scale(col_name="a", old_min=10, old_max=0)


def test_min_max_new_min_max_errors(dataframe):
    with pytest.raises(ValueError):
        df = dataframe.min_max_scale(col_name="a", new_min=10, new_max=0)


def test_collapse_levels_sanity(multiindex_with_missing_dataframe):
    with pytest.raises(TypeError):
        multiindex_with_missing_dataframe.collapse_levels(sep=3)


def test_collapse_levels_non_multilevel(multiindex_with_missing_dataframe):
    # an already single-level DataFrame is not distorted
    pd.testing.assert_frame_equal(
        multiindex_with_missing_dataframe.copy().collapse_levels(),
        multiindex_with_missing_dataframe.collapse_levels().collapse_levels(),
    )


def test_collapse_levels_functionality_2level(
    multiindex_with_missing_dataframe
):

    assert all(
        multiindex_with_missing_dataframe.copy()
        .collapse_levels()
        .columns.values
        == ["a", "Normal  Distribution", "decorated-elephant_r.i.p-rhino :'("]
    )
    assert all(
        multiindex_with_missing_dataframe.copy()
        .collapse_levels(sep="AsDf")
        .columns.values
        == [
            "a",
            "Normal  Distribution",
            "decorated-elephantAsDfr.i.p-rhino :'(",
        ]
    )


def test_collapse_levels_functionality_3level(
    multiindex_with_missing_3level_dataframe
):
    assert all(
        multiindex_with_missing_3level_dataframe.copy()
        .collapse_levels()
        .columns.values
        == [
            "a",
            "Normal  Distribution_Hypercuboid (???)",
            "decorated-elephant_r.i.p-rhino :'(_deadly__flamingo",
        ]
    )
    assert all(
        multiindex_with_missing_3level_dataframe.copy()
        .collapse_levels(sep="AsDf")
        .columns.values
        == [
            "a",
            "Normal  DistributionAsDfHypercuboid (???)",
            "decorated-elephantAsDfr.i.p-rhino :'(AsDfdeadly__flamingo",
        ]
    )


def test_reset_index_inplace_obj_equivalence(dataframe):
    """ Make sure operation is indeed in place. """

    df_riip = dataframe.reset_index_inplace()

    assert df_riip is dataframe


def test_reset_index_inplace_after_group(dataframe):
    """ Make sure equivalent output to non-in place. """

    df_sum = dataframe.groupby(["animals@#$%^", "cities"]).sum()

    df_sum_ri = df_sum.reset_index()
    df_sum.reset_index_inplace()

    pd.testing.assert_frame_equal(df_sum_ri, df_sum)


def test_reset_index_inplace_drop(dataframe):
    """ Test that correctly accepts `reset_index()` parameters. """

    pd.testing.assert_frame_equal(
        dataframe.reset_index(drop=True),
        dataframe.reset_index_inplace(drop=True),
    )


def test_select_columns(dataframe):
    columns = ["a", "Bell__Chart", "cities"]
    df = dataframe.select_columns(columns=columns)

    pd.testing.assert_frame_equal(df, dataframe[columns])


def test_select_columns_invert(dataframe):
    columns = ["a", "Bell__Chart", "cities"]
    df = dataframe.select_columns(columns=columns, invert=True)
    columns2 = ["decorated-elephant", "animals@#$%^"]
    pd.testing.assert_frame_equal(df, dataframe[columns2])


@pytest.fixture
def missingdata_df():
    np.random.seed(9)
    data = {
        "a": [1, 2, np.nan] * 3,
        "Bell__Chart": [1.234_523_45, np.nan, 3.234_612_5] * 3,
        "decorated-elephant": [1, 2, 3] * 3,
        "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
        "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    }
    df = pd.DataFrame(data)
    return df


def test_imputation_single_value(missingdata_df):
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
def test_imputation_statistical(missingdata_df, statistic, expected):
    df = missingdata_df.impute("a", statistic=statistic)
    assert set(df["a"]) == expected
