from itertools import product

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor import patterns

df_checks = pd.DataFrame(
    [
        {"region": "Pacific", "2007": 1039, "2009": 2587},
        {"region": "Southwest", "2007": 51, "2009": 176},
        {"region": "Rocky Mountains and Plains", "2007": 200, "2009": 338},
    ]
)


@pytest.fixture
def df_checks_output():
    return pd.DataFrame(
        {
            "region": [
                "Pacific",
                "Pacific",
                "Southwest",
                "Southwest",
                "Rocky Mountains and Plains",
                "Rocky Mountains and Plains",
            ],
            "year": ["2007", "2009", "2007", "2009", "2007", "2009"],
            "num_nests": [1039, 2587, 51, 176, 200, 338],
        }
    )


@pytest.fixture
def test_df():
    return pd.DataFrame(
        {
            "off_loc": ["A", "B", "C", "D", "E", "F"],
            "pt_loc": ["G", "H", "I", "J", "K", "L"],
            "pt_lat": [
                100.07548220000001,
                75.191326,
                122.65134479999999,
                124.13553329999999,
                124.13553329999999,
                124.01028909999998,
            ],
            "off_lat": [
                121.271083,
                75.93845266,
                135.043791,
                134.51128400000002,
                134.484374,
                137.962195,
            ],
            "pt_long": [
                4.472089953,
                -144.387785,
                -40.45611048,
                -46.07156181,
                -46.07156181,
                -46.01594293,
            ],
            "off_long": [
                -7.188632000000001,
                -143.2288569,
                21.242563,
                40.937416999999996,
                40.78472,
                22.905889000000002,
            ],
        }
    )


@pytest.fixture
def df_multi():
    """MultiIndex dataframe fixture."""
    return pd.DataFrame(
        {
            ("name", "a"): {0: "Wilbur", 1: "Petunia", 2: "Gregory"},
            ("names", "aa"): {0: 67, 1: 80, 2: 64},
            ("more_names", "aaa"): {0: 56, 1: 90, 2: 50},
        }
    )


index_labels = [2007, 20.09]
column_labels = [2007, 20.99]
names_to_labels = [1, {12, "newnames"}]

index_does_not_exist = ["Region", ["207", "region"]]
column_does_not_exist = ["two thousand and seven", ("2007", 2009)]

index_type_checks = (
    (frame, index) for frame, index in product([df_checks], index_labels)
)
column_type_checks = (
    (frame, column_name)
    for frame, column_name in product([df_checks], column_labels)
)
names_to_type_checks = (
    (frame, names_to)
    for frame, names_to in product([df_checks], names_to_labels)
)

names_to_sub_type_checks = (
    (df_checks, (1, "rar")),
    (df_checks, [{"set"}, 20]),
)

index_presence_checks = (
    (frame, index)
    for frame, index in product([df_checks], index_does_not_exist)
)
column_presence_checks = (
    (frame, column_name)
    for frame, column_name in product([df_checks], column_does_not_exist)
)

names_sep_not_required = [
    (df_checks, "rar", "_"),
    (df_checks, ["blessed"], ","),
]

names_sep_type_check = [
    (df_checks, ["rar", "bar"], 1),
    (df_checks, ("rar", "ragnar"), ["\\d+"]),
]
names_pattern_type_check = [
    (df_checks, "rar", 1),
    (df_checks, {1: "rar"}, {"\\d+"}),
]


@pytest.mark.parametrize("df,index", index_type_checks)
def test_type_index(df, index):
    """Raise TypeError if wrong type is provided for the `index`."""
    with pytest.raises(TypeError):
        df.pivot_longer(index=index)


@pytest.mark.parametrize("df,column", column_type_checks)
def test_type_column_names(df, column):
    """Raise TypeError if wrong type is provided for `column_names`."""
    with pytest.raises(TypeError):
        df.pivot_longer(column_names=column)


@pytest.mark.parametrize("df,names_to", names_to_type_checks)
def test_type_names_to(df, names_to):
    """Raise TypeError if wrong type is provided for `names_to`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to)


@pytest.mark.parametrize("df,names_to", names_to_sub_type_checks)
def test_subtype_names_to(df, names_to):
    """
    Raise TypeError if `names_to` is a list/tuple and the wrong type is
    provided for entries in `names_to`.
    """
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to)


@pytest.mark.parametrize("df,index", index_presence_checks)
def test_presence_index(df, index):
    """Raise KeyError if labels in `index` do not exist."""
    with pytest.raises(KeyError):
        df.pivot_longer(index=index)


@pytest.mark.parametrize("df,column", column_presence_checks)
def test_presence_columns(df, column):
    """Raise KeyError if labels in `column_names` do not exist."""
    with pytest.raises(KeyError):
        df.pivot_longer(column_names=column)


@pytest.mark.parametrize("df,names_to, names_sep", names_sep_not_required)
def test_name_sep_names_to_len(df, names_to, names_sep):
    """
    Raise ValueError if `names_to` is a string,
    or `names_to` is a list/tuple,
    its length is one,
    and `names_sep` is provided.
    """
    with pytest.raises(ValueError):
        df.pivot_longer(names_to=names_to, names_sep=names_sep)


@pytest.mark.parametrize("df,names_to, names_sep", names_sep_type_check)
def test_name_sep_wrong_type(df, names_to, names_sep):
    """Raise TypeError if the wrong type is provided for `names_sep`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to, names_sep=names_sep)


@pytest.mark.parametrize(
    "df,names_to, names_pattern", names_pattern_type_check
)
def test_name_pattern_wrong_type(df, names_to, names_pattern):
    """Raise TypeError if the wrong type provided for `names_pattern`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to, names_pattern=names_pattern)


def test_names_pattern_wrong_subtype():
    """
    Raise TypeError if `names_pattern` is a list/tuple
    and wrong subtype is supplied.
    """
    with pytest.raises(TypeError):
        df_checks.pivot_longer(
            names_to=["variable", "value"], names_pattern=[1, "rar"]
        )


def test_names_pattern_names_to_wrong_type():
    """
    Raise TypeError if `names_pattern` is a list/tuple
    and wrong type is supplied for `names_to`.
    """
    with pytest.raises(TypeError):
        df_checks.pivot_longer(
            names_to={"variable, value"}, names_pattern=["1", "rar"]
        )


def test_column_level_wrong_type(df_multi):
    """Raise TypeError if wrong type is provided for `column_level`."""
    with pytest.raises(TypeError):
        df_multi.pivot_longer(index="name", column_level={0})


def test_sort_by_appearance(test_df):
    """Raise error if `sort_by_appearance` is not boolean."""
    with pytest.raises(TypeError):
        test_df.pivot_longer(
            names_to=[".value", "value"],
            names_sep="_",
            sort_by_appearance="TRUE",
        )


def test_ignore_index(test_df):
    """Raise error if `ignore_index` is not boolean."""
    with pytest.raises(TypeError):
        test_df.pivot_longer(
            names_to=[".value", "value"], names_sep="_", ignore_index="TRUE"
        )


def test_duplicate_dot_value(test_df):
    """Raise error if `names_to` contains more than one `.value."""
    with pytest.raises(ValueError):
        test_df.pivot_longer(
            names_to=[".value", ".value"], names_pattern="(.+)_(.+)"
        )


def test_names_pattern_names_to_unequal_length():
    """
    Raise ValueError if `names_pattern` is a list/tuple
    and wrong number of items in `names_to`.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["variable"], names_pattern=["1", "rar"]
        )


def test_names_pattern_names_to_dot_value():
    """
    Raise Error if `names_pattern` is a list/tuple and
    `.value` in `names_to`.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["variable", ".value"], names_pattern=["1", "rar"]
        )


def test_both_names_sep_and_pattern():
    """
    Raise ValueError if both `names_sep`
    and `names_pattern` is provided.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["rar", "bar"], names_sep="-", names_pattern=r"\\d+"
        )


def test_names_pattern_column_MultiIndex(df_multi):
    """Raise ValueError if `names_pattern` and MultiIndex column"""
    with pytest.raises(ValueError):
        df_multi.pivot_longer(index="name", names_pattern=r"(.+)(.)")


def test_index_tuple_MultiIndex(df_multi):
    """
    Raise ValueError if `index` is a tuple,
    instead of a list of tuples,
    and the dataframe's column is a MultiIndex.
    """
    with pytest.raises(ValueError):
        df_multi.pivot_longer(index=("name", "a"))


def test_column_names_tuple_MultiIndex(df_multi):
    """
    Raise ValueError if `column_names` is a tuple,
    instead of a list of tuples,
    and the dataframe's column is a MultiIndex.
    """
    with pytest.raises(ValueError):
        df_multi.pivot_longer(column_names=("names", "aa"))


def test_values_to_wrong_type():
    """Raise TypeError if the wrong type is provided for `values_to`."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(values_to=["salvo"])


def test_values_to_exists_in_columns():
    """
    Raise ValueError if `values_to` already
    exists in the dataframe's columns.
    """
    with pytest.raises(ValueError):
        df_checks.pivot_longer(values_to="region")


def test_MultiIndex_column_level(df_multi):
    """Test output from MultiIndex column"""
    result = df_multi.pivot_longer(
        index="name", column_names="names", column_level=0
    )
    expected_output = df_multi.melt(
        id_vars="name", value_vars="names", col_level=0
    )
    assert_frame_equal(result, expected_output)


def test_pivot_no_args_passed():
    """Test output if no arguments are passed."""
    df_no_args = pd.DataFrame({"name": ["Wilbur", "Petunia", "Gregory"]})
    df_no_args_output = pd.DataFrame(
        {
            "variable": ["name", "name", "name"],
            "value": ["Wilbur", "Petunia", "Gregory"],
        }
    )
    result = df_no_args.pivot_longer()

    assert_frame_equal(result, df_no_args_output)


def test_pivot_index_only_and_sort_by_appearance(df_checks_output):
    """
    Test output if only `index` is passed and
    `sort_by_apperance is ``True``.
    """
    result = df_checks.pivot_longer(
        index="region",
        names_to="year",
        values_to="num_nests",
        sort_by_appearance=True,
    )
    assert_frame_equal(result, df_checks_output)


def test_pivot_index_only():
    """Test output if only `index` is passed."""
    result = df_checks.pivot_longer(
        index="region", names_to="year", values_to="num_nests",
    )

    df_out = pd.DataFrame(
        {
            "region": [
                "Pacific",
                "Southwest",
                "Rocky Mountains and Plains",
                "Pacific",
                "Southwest",
                "Rocky Mountains and Plains",
            ],
            "year": ["2007", "2007", "2007", "2009", "2009", "2009"],
            "num_nests": [1039, 51, 200, 2587, 176, 338],
        }
    )

    assert_frame_equal(result, df_out)


def test_ignore_index_False():
    """
    Test dataframe output if `ignore_index` is False,
    and `sort_by_appearance` is False.
    """
    df_in = pd.DataFrame(
        {
            "A": {0: "a", 1: "b", 2: "c"},
            "B": {0: 1, 1: 3, 2: 5},
            "C": {0: 2, 1: 4, 2: 6},
        }
    )

    result = df_in.pivot_longer(
        index="A",
        column_names=["B", "C"],
        ignore_index=False,
        sort_by_appearance=False,
    )

    df_out = pd.DataFrame(
        {
            "A": ["a", "b", "c", "a", "b", "c"],
            "variable": ["B", "B", "B", "C", "C", "C"],
            "value": [1, 3, 5, 2, 4, 6],
        },
        index=pd.Int64Index([0, 1, 2, 0, 1, 2], dtype="int64"),
    )

    assert_frame_equal(result, df_out)


def test_ignore_index_false_same_length():
    """
    Test output when `ignore_index` is False
    and the length of the new dataframe is the same
    as the length of the original dataframe.
    """
    df_in = pd.DataFrame(
        {
            "name": {
                (67, 56): "Wilbur",
                (80, 90): "Petunia",
                (64, 50): "Gregory",
            }
        }
    )
    df_out = pd.DataFrame(
        {
            "variable": {(64, 50): "name", (67, 56): "name", (80, 90): "name"},
            "value": {
                (64, 50): "Gregory",
                (67, 56): "Wilbur",
                (80, 90): "Petunia",
            },
        }
    )

    result = df_in.pivot_longer(ignore_index=False)
    assert_frame_equal(result, df_out)


def test_pivot_column_only(df_checks_output):
    """Test output if only `column_names` is passed."""
    result = df_checks.pivot_longer(
        column_names=["2007", "2009"],
        names_to="year",
        values_to="num_nests",
        sort_by_appearance=True,
    )
    assert_frame_equal(result, df_checks_output)


def test_pivot_index_patterns_only_sort_by_appearance(df_checks_output):
    """
    Test output if the `patterns` function is passed to `index`, and
    `sort_by_appearance` is ``True``.
    """
    result = df_checks.pivot_longer(
        index=patterns(r"[^\d+]"),
        names_to="year",
        values_to="num_nests",
        sort_by_appearance=True,
    )
    assert_frame_equal(result, df_checks_output)


def test_pivot_no_index_no_columns():
    """Test output if neither `index`/`columns_names` is passed."""
    test = pd.DataFrame(
        {"1": ["fur", "lace"], "2": ["car", "plane"], "3": ["nsw", "vic"]}
    )
    result = test.pivot_longer()
    expected_output = pd.DataFrame(
        {
            "variable": ["1", "1", "2", "2", "3", "3"],
            "value": ["fur", "lace", "car", "plane", "nsw", "vic"],
        }
    )
    assert_frame_equal(result, expected_output)


def test_pivot_columns_patterns_only(df_checks_output):
    """Test output if the `patterns` function is passed to `column_names`."""
    result = df_checks.pivot_longer(
        column_names=patterns(r"\d+"),
        names_to="year",
        values_to="num_nests",
        sort_by_appearance=True,
    )
    assert_frame_equal(result, df_checks_output)


def test_length_mismatch():
    """
    Raise error if `names_to` is a list/tuple and its length
    does not match the number of extracted columns.
    """
    data = pd.DataFrame(
        {
            "country": ["United States", "Russia", "China"],
            "vault_2012_f": [48.132, 46.366, 44.266],
            "vault_2012_m": [46.632, 46.866, 48.316],
            "vault_2016_f": [46.866, 45.733, 44.332],
            "vault_2016_m": [45.865, 46.033, 45.0],
            "floor_2012_f": [45.366, 41.599, 40.833],
            "floor_2012_m": [45.266, 45.308, 45.133],
            "floor_2016_f": [45.999, 42.032, 42.066],
            "floor_2016_m": [43.757, 44.766, 43.799],
        }
    )
    with pytest.raises(ValueError):
        data.pivot_longer(names_to=["event", "year"], names_sep="-")


def test_empty_mapping(test_df):
    """Raise error if `names_pattern` is a regex and returns no matches."""
    with pytest.raises(ValueError):
        test_df.pivot_longer(
            names_to=[".value", "value"], names_pattern=r"(\d+)([A-Z])"
        )


def test_len_mapping_gt_len_names_to(test_df):
    """
    Raise error if `names_pattern` is a regex and returns number of
    matches more than length of `names_to`.
    """
    with pytest.raises(ValueError):
        test_df.pivot_longer(
            names_to=[".value", "value"], names_pattern="(.+)(_)(.+)"
        )


@pytest.fixture
def names_pattern_list_df():
    return pd.DataFrame(
        [
            {
                "ID": 1,
                "DateRange1Start": "1/1/90",
                "DateRange1End": "3/1/90",
                "Value1": 4.4,
                "DateRange2Start": "4/5/91",
                "DateRange2End": "6/7/91",
                "Value2": 6.2,
                "DateRange3Start": "5/5/95",
                "DateRange3End": "6/6/96",
                "Value3": 3.3,
            }
        ]
    )


def test_names_pattern_list_empty(names_pattern_list_df):
    """
    Raise ValueError if `names_pattern` is a list,
    and nothing is returned.
    """
    with pytest.raises(ValueError):
        names_pattern_list_df.pivot_longer(
            index="ID",
            names_to=("DateRangeStart", "DateRangeEnd", "Value"),
            names_pattern=("^Start", "^End", "Value$"),
        )


def test_names_pattern_list(names_pattern_list_df):
    """Test output if `names_pattern` is a list."""
    result = names_pattern_list_df.pivot_longer(
        index="ID",
        names_to=("DateRangeStart", "DateRangeEnd", "Value"),
        names_pattern=("Start$", "End$", "^Value"),
    )

    expected_output = pd.DataFrame(
        {
            "ID": [1, 1, 1],
            "DateRangeStart": ["1/1/90", "4/5/91", "5/5/95"],
            "DateRangeEnd": ["3/1/90", "6/7/91", "6/6/96"],
            "Value": [4.4, 6.2, 3.3],
        }
    )

    assert_frame_equal(result, expected_output)


multiple_values_pattern = [
    (
        pd.DataFrame(
            {
                "country": ["United States", "Russia", "China"],
                "vault2012": [48.1, 46.4, 44.3],
                "floor2012": [45.4, 41.6, 40.8],
                "vault2016": [46.9, 45.7, 44.3],
                "floor2016": [46.0, 42.0, 42.1],
            }
        ),
        pd.DataFrame(
            [
                {
                    "country": "United States",
                    "event": "vault",
                    "year": "2012",
                    "score": 48.1,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": "2012",
                    "score": 45.4,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": "2016",
                    "score": 46.9,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": "2016",
                    "score": 46.0,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": "2012",
                    "score": 46.4,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": "2012",
                    "score": 41.6,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": "2016",
                    "score": 45.7,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": "2016",
                    "score": 42.0,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": "2012",
                    "score": 44.3,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": "2012",
                    "score": 40.8,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": "2016",
                    "score": 44.3,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": "2016",
                    "score": 42.1,
                },
            ]
        ),
        "country",
        ("event", "year"),
        r"([A-Za-z]+)(\d+)",
        True,
    )
]


# probably unneccesary as it is just one item
# the idea is that more tests will be added for this
@pytest.mark.parametrize(
    "df_in,df_out,index,names_to,names_pattern, sort_by_appearance",
    multiple_values_pattern,
)
def test_multiple_values_pattern(
    df_in, df_out, index, names_to, names_pattern, sort_by_appearance
):
    """
    Test function to extract multiple columns, using the `names_to` and
    names_pattern arguments.
    """
    result = df_in.pivot_longer(
        index=index,
        names_to=names_to,
        names_pattern=names_pattern,
        values_to="score",
        sort_by_appearance=sort_by_appearance,
    )
    assert_frame_equal(result, df_out)


multiple_values_sep = [
    (
        pd.DataFrame(
            {
                "country": ["United States", "Russia", "China"],
                "vault_2012": [48.1, 46.4, 44.3],
                "floor_2012": [45.4, 41.6, 40.8],
                "vault_2016": [46.9, 45.7, 44.3],
                "floor_2016": [46.0, 42.0, 42.1],
            }
        ),
        pd.DataFrame(
            [
                {
                    "country": "United States",
                    "event": "vault",
                    "year": "2012",
                    "score": 48.1,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": "2012",
                    "score": 45.4,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": "2016",
                    "score": 46.9,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": "2016",
                    "score": 46.0,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": "2012",
                    "score": 46.4,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": "2012",
                    "score": 41.6,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": "2016",
                    "score": 45.7,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": "2016",
                    "score": 42.0,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": "2012",
                    "score": 44.3,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": "2012",
                    "score": 40.8,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": "2016",
                    "score": 44.3,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": "2016",
                    "score": 42.1,
                },
            ]
        ),
        "country",
        ("event", "year"),
        "_",
        True,
    ),
    (
        pd.DataFrame(
            {
                "country": ["United States", "Russia", "China"],
                "vault_2012_f": [
                    48.132,
                    46.36600000000001,
                    44.266000000000005,
                ],
                "vault_2012_m": [46.632, 46.86600000000001, 48.316],
                "vault_2016_f": [
                    46.86600000000001,
                    45.733000000000004,
                    44.332,
                ],
                "vault_2016_m": [45.865, 46.033, 45.0],
                "floor_2012_f": [45.36600000000001, 41.599, 40.833],
                "floor_2012_m": [45.266000000000005, 45.308, 45.133],
                "floor_2016_f": [45.998999999999995, 42.032, 42.066],
                "floor_2016_m": [43.757, 44.766000000000005, 43.799],
            }
        ),
        pd.DataFrame(
            [
                {
                    "country": "United States",
                    "event": "vault",
                    "year": "2012",
                    "gender": "f",
                    "score": 48.132,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": "2012",
                    "gender": "m",
                    "score": 46.632,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": "2016",
                    "gender": "f",
                    "score": 46.86600000000001,
                },
                {
                    "country": "United States",
                    "event": "vault",
                    "year": "2016",
                    "gender": "m",
                    "score": 45.865,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": "2012",
                    "gender": "f",
                    "score": 45.36600000000001,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": "2012",
                    "gender": "m",
                    "score": 45.266000000000005,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": "2016",
                    "gender": "f",
                    "score": 45.998999999999995,
                },
                {
                    "country": "United States",
                    "event": "floor",
                    "year": "2016",
                    "gender": "m",
                    "score": 43.757,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": "2012",
                    "gender": "f",
                    "score": 46.36600000000001,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": "2012",
                    "gender": "m",
                    "score": 46.86600000000001,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": "2016",
                    "gender": "f",
                    "score": 45.733000000000004,
                },
                {
                    "country": "Russia",
                    "event": "vault",
                    "year": "2016",
                    "gender": "m",
                    "score": 46.033,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": "2012",
                    "gender": "f",
                    "score": 41.599,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": "2012",
                    "gender": "m",
                    "score": 45.308,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": "2016",
                    "gender": "f",
                    "score": 42.032,
                },
                {
                    "country": "Russia",
                    "event": "floor",
                    "year": "2016",
                    "gender": "m",
                    "score": 44.766000000000005,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": "2012",
                    "gender": "f",
                    "score": 44.266000000000005,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": "2012",
                    "gender": "m",
                    "score": 48.316,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": "2016",
                    "gender": "f",
                    "score": 44.332,
                },
                {
                    "country": "China",
                    "event": "vault",
                    "year": "2016",
                    "gender": "m",
                    "score": 45.0,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": "2012",
                    "gender": "f",
                    "score": 40.833,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": "2012",
                    "gender": "m",
                    "score": 45.133,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": "2016",
                    "gender": "f",
                    "score": 42.066,
                },
                {
                    "country": "China",
                    "event": "floor",
                    "year": "2016",
                    "gender": "m",
                    "score": 43.799,
                },
            ]
        ),
        "country",
        ("event", "year", "gender"),
        "_",
        True,
    ),
]


@pytest.mark.parametrize(
    "df_in,df_out,index,names_to,names_sep,sort_by_appearance",
    multiple_values_sep,
)
def test_multiple_values_sep(
    df_in, df_out, index, names_to, names_sep, sort_by_appearance
):
    """
    Test function to extract multiple columns, using the `names_to` and
    names_sep arguments.
    """
    result = df_in.pivot_longer(
        index=index,
        names_to=names_to,
        names_sep=names_sep,
        values_to="score",
        sort_by_appearance=sort_by_appearance,
    )
    assert_frame_equal(result, df_out)


paired_columns_pattern = [
    (
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "M_start_date_1": [201709, 201709, 201709],
                "M_end_date_1": [201905, 201905, 201905],
                "M_start_date_2": [202004, 202004, 202004],
                "M_end_date_2": [202005, 202005, 202005],
                "F_start_date_1": [201803, 201803, 201803],
                "F_end_date_1": [201904, 201904, 201904],
                "F_start_date_2": [201912, 201912, 201912],
                "F_end_date_2": [202007, 202007, 202007],
            }
        ),
        pd.DataFrame(
            [
                {"id": 1, "cod": "M", "start": 201709, "end": 201905},
                {"id": 1, "cod": "M", "start": 202004, "end": 202005},
                {"id": 1, "cod": "F", "start": 201803, "end": 201904},
                {"id": 1, "cod": "F", "start": 201912, "end": 202007},
                {"id": 2, "cod": "M", "start": 201709, "end": 201905},
                {"id": 2, "cod": "M", "start": 202004, "end": 202005},
                {"id": 2, "cod": "F", "start": 201803, "end": 201904},
                {"id": 2, "cod": "F", "start": 201912, "end": 202007},
                {"id": 3, "cod": "M", "start": 201709, "end": 201905},
                {"id": 3, "cod": "M", "start": 202004, "end": 202005},
                {"id": 3, "cod": "F", "start": 201803, "end": 201904},
                {"id": 3, "cod": "F", "start": 201912, "end": 202007},
            ]
        ),
        "id",
        ("cod", ".value"),
        "(M|F)_(start|end)_.+",
        True,
    ),
    (
        pd.DataFrame(
            {
                "person_id": [1, 2, 3],
                "date1": ["12/31/2007", "11/25/2009", "10/06/2005"],
                "val1": [2, 4, 6],
                "date2": ["12/31/2017", "11/25/2019", "10/06/2015"],
                "val2": [1, 3, 5],
                "date3": ["12/31/2027", "11/25/2029", "10/06/2025"],
                "val3": [7, 9, 11],
            }
        ),
        pd.DataFrame(
            [
                {"person_id": 1, "value": "1", "date": "12/31/2007", "val": 2},
                {"person_id": 1, "value": "2", "date": "12/31/2017", "val": 1},
                {"person_id": 1, "value": "3", "date": "12/31/2027", "val": 7},
                {"person_id": 2, "value": "1", "date": "11/25/2009", "val": 4},
                {"person_id": 2, "value": "2", "date": "11/25/2019", "val": 3},
                {"person_id": 2, "value": "3", "date": "11/25/2029", "val": 9},
                {"person_id": 3, "value": "1", "date": "10/06/2005", "val": 6},
                {"person_id": 3, "value": "2", "date": "10/06/2015", "val": 5},
                {
                    "person_id": 3,
                    "value": "3",
                    "date": "10/06/2025",
                    "val": 11,
                },
            ]
        ),
        patterns("^(?!(date|val))"),
        (".value", "value"),
        r"([a-z]+)(\d)",
        True,
    ),
    (
        pd.DataFrame(
            [
                {
                    "id": 1,
                    "a1": "a",
                    "a2": "b",
                    "a3": "c",
                    "A1": "A",
                    "A2": "B",
                    "A3": "C",
                }
            ]
        ),
        pd.DataFrame(
            {
                "id": [1, 1, 1],
                "instance": ["1", "2", "3"],
                "a": ["a", "b", "c"],
                "A": ["A", "B", "C"],
            },
            index=[0, 0, 0],
        ),
        "id",
        (".value", "instance"),
        r"(\w)(\d)",
        True,
    ),
    (
        pd.DataFrame(
            {
                "A1970": ["a", "b", "c"],
                "A1980": ["d", "e", "f"],
                "B1970": [2.5, 1.2, 0.7],
                "B1980": [3.2, 1.3, 0.1],
                "X": [-1.085631, 0.997345, 0.282978],
            }
        ),
        pd.DataFrame(
            {
                "X": [
                    -1.085631,
                    -1.085631,
                    0.997345,
                    0.997345,
                    0.282978,
                    0.282978,
                ],
                "year": ["1970", "1980", "1970", "1980", "1970", "1980"],
                "A": ["a", "d", "b", "e", "c", "f"],
                "B": [2.5, 3.2, 1.2, 1.3, 0.7, 0.1],
            }
        ),
        "X",
        (".value", "year"),
        "([A-Z])(.+)",
        True,
    ),
    (
        pd.DataFrame(
            {
                "id": ["A", "B", "C", "D", "E", "F"],
                "f_start": ["p", "i", "i", "p", "p", "i"],
                "d_start": [
                    "2018-01-01",
                    "2019-04-01",
                    "2018-06-01",
                    "2019-12-01",
                    "2019-02-01",
                    "2018-04-01",
                ],
                "f_end": ["p", "p", "i", "p", "p", "i"],
                "d_end": [
                    "2018-02-01",
                    "2020-01-01",
                    "2019-03-01",
                    "2020-05-01",
                    "2019-05-01",
                    "2018-07-01",
                ],
            }
        ),
        pd.DataFrame(
            [
                {"id": "A", "status": "start", "f": "p", "d": "2018-01-01"},
                {"id": "A", "status": "end", "f": "p", "d": "2018-02-01"},
                {"id": "B", "status": "start", "f": "i", "d": "2019-04-01"},
                {"id": "B", "status": "end", "f": "p", "d": "2020-01-01"},
                {"id": "C", "status": "start", "f": "i", "d": "2018-06-01"},
                {"id": "C", "status": "end", "f": "i", "d": "2019-03-01"},
                {"id": "D", "status": "start", "f": "p", "d": "2019-12-01"},
                {"id": "D", "status": "end", "f": "p", "d": "2020-05-01"},
                {"id": "E", "status": "start", "f": "p", "d": "2019-02-01"},
                {"id": "E", "status": "end", "f": "p", "d": "2019-05-01"},
                {"id": "F", "status": "start", "f": "i", "d": "2018-04-01"},
                {"id": "F", "status": "end", "f": "i", "d": "2018-07-01"},
            ]
        ),
        "id",
        (".value", "status"),
        "(.*)_(.*)",
        True,
    ),
]


@pytest.mark.parametrize(
    """
    df_in,df_out,index,names_to,names_pattern,
    sort_by_appearance
    """,
    paired_columns_pattern,
)
def test_extract_column_names_pattern(
    df_in, df_out, index, names_to, names_pattern, sort_by_appearance,
):
    """
    Test output if `.value` is in the `names_to`
    argument and names_pattern is used.
    """
    result = df_in.pivot_longer(
        index=index,
        names_to=names_to,
        names_pattern=names_pattern,
        sort_by_appearance=sort_by_appearance,
    )
    assert_frame_equal(result, df_out)


paired_columns_sep = [
    # https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.wide_to_long.html
    (
        pd.DataFrame(
            {
                "A(weekly)-2010": [0.548814, 0.7151890000000001, 0.602763],
                "A(weekly)-2011": [0.544883, 0.423655, 0.645894],
                "B(weekly)-2010": [
                    0.437587,
                    0.8917729999999999,
                    0.9636629999999999,
                ],
                "B(weekly)-2011": [0.383442, 0.791725, 0.528895],
                "X": [0, 1, 1],
            }
        ),
        pd.DataFrame(
            {
                "X": [0, 0, 1, 1, 1, 1],
                "year": ["2010", "2011", "2010", "2011", "2010", "2011"],
                "A(weekly)": [
                    0.548814,
                    0.544883,
                    0.7151890000000001,
                    0.423655,
                    0.602763,
                    0.645894,
                ],
                "B(weekly)": [
                    0.437587,
                    0.383442,
                    0.8917729999999999,
                    0.791725,
                    0.9636629999999999,
                    0.528895,
                ],
            }
        ),
        "X",
        (".value", "year"),
        "-",
        True,
    ),
    (
        pd.DataFrame(
            {
                "indexer": [0, 1],
                "S_1": [1, 1],
                "S_2": [0, 1],
                "S_3": ["0", np.nan],
                "S_4": ["1", np.nan],
            }
        ),
        pd.DataFrame(
            {
                "indexer": [0, 0, 0, 0, 1, 1, 1, 1],
                "num": ["1", "2", "3", "4", "1", "2", "3", "4"],
                "S": [1, 0, "0", "1", 1, 1, np.nan, np.nan],
            }
        ),
        "indexer",
        (".value", "num"),
        "_",
        True,
    ),
    (
        pd.DataFrame(
            {
                "county": [1001, 1003, 1005],
                "area": [275, 394, 312],
                "pop_2006": [1037, 2399, 1638],
                "pop_2007": [1052, 2424, 1647],
                "pop_2008": [1102, 2438, 1660],
            }
        ),
        pd.DataFrame(
            {
                "county": [
                    1001,
                    1001,
                    1001,
                    1003,
                    1003,
                    1003,
                    1005,
                    1005,
                    1005,
                ],
                "area": [275, 275, 275, 394, 394, 394, 312, 312, 312],
                "year": [
                    "2006",
                    "2007",
                    "2008",
                    "2006",
                    "2007",
                    "2008",
                    "2006",
                    "2007",
                    "2008",
                ],
                "pop": [1037, 1052, 1102, 2399, 2424, 2438, 1638, 1647, 1660],
            }
        ),
        ["county", "area"],
        (".value", "year"),
        "_",
        True,
    ),
    (
        pd.DataFrame(
            {
                "family": [1, 2, 3, 4, 5],
                "dob_child1": [
                    "1998-11-26",
                    "1996-06-22",
                    "2002-07-11",
                    "2004-10-10",
                    "2000-12-05",
                ],
                "dob_child2": [
                    "2000-01-29",
                    np.nan,
                    "2004-04-05",
                    "2009-08-27",
                    "2005-02-28",
                ],
                "gender_child1": [1, 2, 2, 1, 2],
                "gender_child2": [2.0, np.nan, 2.0, 1.0, 1.0],
            }
        ),
        pd.DataFrame(
            [
                {
                    "family": 1,
                    "child": "child1",
                    "dob": "1998-11-26",
                    "gender": 1.0,
                },
                {
                    "family": 1,
                    "child": "child2",
                    "dob": "2000-01-29",
                    "gender": 2.0,
                },
                {
                    "family": 2,
                    "child": "child1",
                    "dob": "1996-06-22",
                    "gender": 2.0,
                },
                {
                    "family": 2,
                    "child": "child2",
                    "dob": np.nan,
                    "gender": np.nan,
                },
                {
                    "family": 3,
                    "child": "child1",
                    "dob": "2002-07-11",
                    "gender": 2.0,
                },
                {
                    "family": 3,
                    "child": "child2",
                    "dob": "2004-04-05",
                    "gender": 2.0,
                },
                {
                    "family": 4,
                    "child": "child1",
                    "dob": "2004-10-10",
                    "gender": 1.0,
                },
                {
                    "family": 4,
                    "child": "child2",
                    "dob": "2009-08-27",
                    "gender": 1.0,
                },
                {
                    "family": 5,
                    "child": "child1",
                    "dob": "2000-12-05",
                    "gender": 2.0,
                },
                {
                    "family": 5,
                    "child": "child2",
                    "dob": "2005-02-28",
                    "gender": 1.0,
                },
            ]
        ),
        "family",
        (".value", "child"),
        "_",
        True,
    ),
    (
        pd.DataFrame(
            {
                "dob_child2": [
                    "2000-01-29",
                    np.nan,
                    "2004-04-05",
                    "2009-08-27",
                    "2005-02-28",
                ],
                "gender_child1": [1, 2, 2, 1, 2],
                "gender_child2": [2.0, np.nan, 2.0, 1.0, 1.0],
                "dob_child1": [
                    "1998-11-26",
                    "1996-06-22",
                    "2002-07-11",
                    "2004-10-10",
                    "2000-12-05",
                ],
            }
        ),
        pd.DataFrame(
            {
                "child": [
                    "child2",
                    "child1",
                    "child2",
                    "child1",
                    "child2",
                    "child1",
                    "child2",
                    "child1",
                    "child2",
                    "child1",
                ],
                "dob": [
                    "2000-01-29",
                    "1998-11-26",
                    np.nan,
                    "1996-06-22",
                    "2004-04-05",
                    "2002-07-11",
                    "2009-08-27",
                    "2004-10-10",
                    "2005-02-28",
                    "2000-12-05",
                ],
                "gender": [2.0, 1, np.nan, 2, 2.0, 2, 1.0, 1, 1.0, 2],
            }
        ),
        None,
        (".value", "child"),
        "_",
        True,
    ),
    (
        pd.DataFrame(
            [
                {
                    "id": "A",
                    "Q1r1_pepsi": 1,
                    "Q1r1_cola": 0,
                    "Q1r2_pepsi": 1,
                    "Q1r2_cola": 0,
                },
                {
                    "id": "B",
                    "Q1r1_pepsi": 0,
                    "Q1r1_cola": 0,
                    "Q1r2_pepsi": 1,
                    "Q1r2_cola": 1,
                },
                {
                    "id": "C",
                    "Q1r1_pepsi": 1,
                    "Q1r1_cola": 1,
                    "Q1r2_pepsi": 1,
                    "Q1r2_cola": 1,
                },
            ]
        ),
        pd.DataFrame(
            [
                {"id": "A", "brand": "pepsi", "Q1r1": 1, "Q1r2": 1},
                {"id": "A", "brand": "cola", "Q1r1": 0, "Q1r2": 0},
                {"id": "B", "brand": "pepsi", "Q1r1": 0, "Q1r2": 1},
                {"id": "B", "brand": "cola", "Q1r1": 0, "Q1r2": 1},
                {"id": "C", "brand": "pepsi", "Q1r1": 1, "Q1r2": 1},
                {"id": "C", "brand": "cola", "Q1r1": 1, "Q1r2": 1},
            ]
        ),
        "id",
        (".value", "brand"),
        "_",
        True,
    ),
    (
        pd.DataFrame(
            {
                "event": [1, 2],
                "url_1": ["g1", "g3"],
                "name_1": ["dc", "nyc"],
                "url_2": ["g2", "g4"],
                "name_2": ["sf", "la"],
            }
        ),
        pd.DataFrame(
            {
                "event": [1, 1, 2, 2],
                "item": ["1", "2", "1", "2"],
                "url": ["g1", "g2", "g3", "g4"],
                "name": ["dc", "sf", "nyc", "la"],
            }
        ),
        "event",
        (".value", "item"),
        "_",
        True,
    ),
]


@pytest.mark.parametrize(
    """
    df_in,df_out,index,names_to,names_sep,
    sort_by_appearance
    """,
    paired_columns_sep,
)
def test_extract_column_names_sep(
    df_in, df_out, index, names_to, names_sep, sort_by_appearance,
):
    """
    Test output if `.value` is in the `names_to` argument
    and names_sep is used.
    """
    result = df_in.pivot_longer(
        index=index,
        names_to=names_to,
        names_sep=names_sep,
        sort_by_appearance=sort_by_appearance,
    )
    assert_frame_equal(result, df_out)


# https://community.rstudio.com/t/pivot-longer-on-multiple-column-sets-pairs/43958/10
paired_columns_no_index_pattern = [
    (
        pd.DataFrame(
            {
                "off_loc": ["A", "B", "C", "D", "E", "F"],
                "pt_loc": ["G", "H", "I", "J", "K", "L"],
                "pt_lat": [
                    100.07548220000001,
                    75.191326,
                    122.65134479999999,
                    124.13553329999999,
                    124.13553329999999,
                    124.01028909999998,
                ],
                "off_lat": [
                    121.271083,
                    75.93845266,
                    135.043791,
                    134.51128400000002,
                    134.484374,
                    137.962195,
                ],
                "pt_long": [
                    4.472089953,
                    -144.387785,
                    -40.45611048,
                    -46.07156181,
                    -46.07156181,
                    -46.01594293,
                ],
                "off_long": [
                    -7.188632000000001,
                    -143.2288569,
                    21.242563,
                    40.937416999999996,
                    40.78472,
                    22.905889000000002,
                ],
            }
        ),
        pd.DataFrame(
            {
                "set": [
                    "off",
                    "pt",
                    "off",
                    "pt",
                    "off",
                    "pt",
                    "off",
                    "pt",
                    "off",
                    "pt",
                    "off",
                    "pt",
                ],
                "loc": [
                    "A",
                    "G",
                    "B",
                    "H",
                    "C",
                    "I",
                    "D",
                    "J",
                    "E",
                    "K",
                    "F",
                    "L",
                ],
                "lat": [
                    121.271083,
                    100.07548220000001,
                    75.93845266,
                    75.191326,
                    135.043791,
                    122.65134479999999,
                    134.51128400000002,
                    124.13553329999999,
                    134.484374,
                    124.13553329999999,
                    137.962195,
                    124.01028909999998,
                ],
                "long": [
                    -7.188632000000001,
                    4.472089953,
                    -143.2288569,
                    -144.387785,
                    21.242563,
                    -40.45611048,
                    40.937416999999996,
                    -46.07156181,
                    40.78472,
                    -46.07156181,
                    22.905889000000002,
                    -46.01594293,
                ],
            }
        ),
        ("set", ".value"),
        "(.+)_(.+)",
        True,
    )
]


@pytest.mark.parametrize(
    """
    df_in,df_out,names_to,names_pattern,
    sort_by_appearance
    """,
    paired_columns_no_index_pattern,
)
def test_paired_columns_no_index_pattern(
    df_in, df_out, names_to, names_pattern, sort_by_appearance
):
    """
    Test function where `.value` is in the `names_to` argument,
    names_pattern is used and no index is supplied.
    """
    result = df_in.pivot_longer(
        names_to=names_to,
        names_pattern=names_pattern,
        sort_by_appearance=sort_by_appearance,
    )
    assert_frame_equal(result, df_out)


names_single_value = [
    (
        pd.DataFrame(
            {
                "event": [1, 2],
                "url_1": ["g1", "g3"],
                "name_1": ["dc", "nyc"],
                "url_2": ["g2", "g4"],
                "name_2": ["sf", "la"],
            }
        ),
        pd.DataFrame(
            {
                "event": [1, 1, 2, 2],
                "url": ["g1", "g2", "g3", "g4"],
                "name": ["dc", "sf", "nyc", "la"],
            }
        ),
        "event",
        "(.+)_.",
        True,
        True,
    ),
    (
        pd.DataFrame(
            {
                "id": [1, 2, 3],
                "x1": [4, 5, 6],
                "x2": [5, 6, 7],
                "y1": [7, 8, 9],
                "y2": [10, 11, 12],
            }
        ),
        pd.DataFrame(
            {
                "id": [1, 1, 2, 2, 3, 3],
                "x": [4, 5, 5, 6, 6, 7],
                "y": [7, 10, 8, 11, 9, 12],
            },
        ),
        "id",
        "(.).",
        True,
        True,
    ),
    (
        pd.DataFrame(
            {
                "x1": [4, 5, 6],
                "x2": [5, 6, 7],
                "y1": [7, 8, 9],
                "y2": [10, 11, 12],
            }
        ),
        pd.DataFrame({"x": [4, 5, 5, 6, 6, 7], "y": [7, 10, 8, 11, 9, 12]}),
        None,
        "(.).",
        True,
        True,
    ),
    (
        pd.DataFrame(
            {
                "x1": [4, 5, 6],
                "x2": [5, 6, 7],
                "y1": [7, 8, 9],
                "y2": [10, 11, 12],
            }
        ),
        pd.DataFrame(
            {"x": [4, 5, 5, 6, 6, 7], "y": [7, 10, 8, 11, 9, 12]},
            index=[0, 0, 1, 1, 2, 2],
        ),
        None,
        "(.).",
        False,
        True,
    ),
]


@pytest.mark.parametrize(
    """
    df_in,df_out,index, names_pattern,
    ignore_index,sort_by_appearance
    """,
    names_single_value,
)
def test_single_value(
    df_in, df_out, index, names_pattern, ignore_index, sort_by_appearance,
):
    """Test function where names_to is a string and == `.value`."""
    result = df_in.pivot_longer(
        index=index,
        names_to=".value",
        names_pattern=names_pattern,
        ignore_index=ignore_index,
        sort_by_appearance=sort_by_appearance,
    )
    assert_frame_equal(result, df_out)


single_column_names_pattern = [
    (
        pd.DataFrame(
            {
                "sepal_length": [5.1, 7.0],
                "sepal_width": [3.5, 3.2],
                "petal_length": [1.4, 4.7],
                "petal_width": [0.2, 1.4],
                "species": ["setosa", "versicolor"],
            }
        ),
        pd.DataFrame(
            [
                {"species": "setosa", "part": "sepal", "value": 5.1},
                {"species": "versicolor", "part": "sepal", "value": 7.0},
                {"species": "setosa", "part": "sepal", "value": 3.5},
                {"species": "versicolor", "part": "sepal", "value": 3.2},
                {"species": "setosa", "part": "petal", "value": 1.4},
                {"species": "versicolor", "part": "petal", "value": 4.7},
                {"species": "setosa", "part": "petal", "value": 0.2},
                {"species": "versicolor", "part": "petal", "value": 1.4},
            ]
        ),
        "species",
        None,
        "part",
        r"(.*)[_].*",
    ),
    (
        pd.DataFrame(
            {
                "sepal_length": [5.1, 7.0],
                "sepal_width": [3.5, 3.2],
                "petal_length": [1.4, 4.7],
                "petal_width": [0.2, 1.4],
                "species": ["setosa", "versicolor"],
                "jargon": ["blabla", "blacksheep"],
            }
        ),
        pd.DataFrame(
            [
                {"species": "setosa", "part": "sepal", "value": 5.1},
                {"species": "versicolor", "part": "sepal", "value": 7.0},
                {"species": "setosa", "part": "sepal", "value": 3.5},
                {"species": "versicolor", "part": "sepal", "value": 3.2},
                {"species": "setosa", "part": "petal", "value": 1.4},
                {"species": "versicolor", "part": "petal", "value": 4.7},
                {"species": "setosa", "part": "petal", "value": 0.2},
                {"species": "versicolor", "part": "petal", "value": 1.4},
            ]
        ),
        "species",
        patterns("_"),
        "part",
        r"(.*)[_].*",
    ),
]


@pytest.mark.parametrize(
    "df_in,df_out,index, column_names,names_to,names_pattern",
    single_column_names_pattern,
)
def test_single_column_names_pattern(
    df_in, df_out, index, column_names, names_to, names_pattern
):
    """
    Test output if `names_to` is a string and
    `names_pattern` returns a single column.
    Also tests when both `index` and `column_names`
    is supplied, and only a subset of the dataframe
    is transformed.
    """
    result = df_in.pivot_longer(
        index=index,
        column_names=column_names,
        names_to=names_to,
        names_pattern=names_pattern,
    )

    assert_frame_equal(result, df_out)


def test_group_present():
    """Test output if 'group' is in `names_to`."""
    df_in = pd.DataFrame(
        {
            "id": [1, 2, 3],
            "x1": [4, 5, 6],
            "x2": [5, 6, 7],
            "y1": [7, 8, 9],
            "y2": [10, 11, 12],
        }
    )

    df_out = pd.DataFrame(
        {
            "id": [1, 2, 3, 1, 2, 3],
            "group": ["1", "1", "1", "2", "2", "2"],
            "x": [4, 5, 6, 5, 6, 7],
            "y": [7, 8, 9, 10, 11, 12],
        }
    )

    result = df_in.pivot_longer(
        index="id", names_to=[".value", "group"], names_pattern="(.)(.)",
    )

    assert_frame_equal(result, df_out)


# copied from :
# https://github.com/pandas-dev/pandas/blob/master/pandas/tests/reshape/test_melt.py
# interesting edge case
def test_float_suffix_irregular():
    """
    Test output for floating suffixes for stubnames;
    the suffixes are unique and differ for the stubnames.
    """
    df = pd.DataFrame(
        {
            "treatment_1.1": [1.0, 2.0],
            "treatment_2.1": [3.0, 4.0],
            "result_1.2": [5.0, 6.0],
            "result_1": [0, 9],
            "A": ["X1", "X2"],
        }
    )

    expected = pd.DataFrame(
        {
            "A": ["X1", "X1", "X1", "X1", "X2", "X2", "X2", "X2"],
            "colname": ["1.1", "2.1", "1.2", "1", "1.1", "2.1", "1.2", "1"],
            "treatment": [1.0, 3.0, np.nan, np.nan, 2.0, 4.0, np.nan, np.nan],
            "result": [np.nan, np.nan, 5.0, 0.0, np.nan, np.nan, 6.0, 9.0],
        }
    )

    result = df.pivot_longer(
        index="A",
        names_to=(".value", "colname"),
        names_sep="_",
        sort_by_appearance=True,
    )
    assert_frame_equal(result, expected)
