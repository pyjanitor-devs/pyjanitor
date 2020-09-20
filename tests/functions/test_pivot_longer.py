from itertools import product

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

df_checks = pd.DataFrame(
    [
        {"region": "Pacific", "2007": 1039, "2009": 2587},
        {"region": "Southwest", "2007": 51, "2009": 176},
        {"region": "Rocky Mountains and Plains", "2007": 200, "2009": 338},
    ]
)
index_labels = [pd.Index(["region"]), {"2007", "region"}]
column_labels = [{"region": 2007}, {"2007", "2009"}]
names_to_labels = [1, {12, "newnames"}]

index_does_not_exist = ["Region", [2007, "region"]]
column_does_not_exist = ["two thousand and seven", ("2007", 2009)]


index_error_checks = [
    (frame, index) for frame, index in product([df_checks], index_labels)
]
column_error_checks = [
    (frame, column_name)
    for frame, column_name in product([df_checks], column_labels)
]
names_to_error_checks = [
    (frame, names_to)
    for frame, names_to in product([df_checks], names_to_labels)
]

index_presence_checks = [
    (frame, index)
    for frame, index in product([df_checks], index_does_not_exist)
]
column_presence_checks = [
    (frame, column_name)
    for frame, column_name in product([df_checks], column_does_not_exist)
]

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
    (df_checks, ["rar"], ["\\d+"]),
]

multi_index_df = [
    pd.DataFrame(
        pd.DataFrame(
            {
                "name": {
                    (67, 56): "Wilbur",
                    (80, 90): "Petunia",
                    (64, 50): "Gregory",
                }
            }
        )
    ),
    pd.DataFrame(
        {
            ("name", "a"): {0: "Wilbur", 1: "Petunia", 2: "Gregory"},
            ("names", "aa"): {0: 67, 1: 80, 2: 64},
            ("more_names", "aaa"): {0: 56, 1: 90, 2: 50},
        }
    ),
    pd.DataFrame(
        {
            ("name", "a"): {
                (0, 2): "Wilbur",
                (1, 3): "Petunia",
                (2, 4): "Gregory",
            },
            ("names", "aa"): {(0, 2): 67, (1, 3): 80, (2, 4): 64},
            ("more_names", "aaa"): {(0, 2): 56, (1, 3): 90, (2, 4): 50},
        }
    ),
]


@pytest.mark.parametrize("df,index", index_error_checks)
def test_type_index(df, index):
    """Raise TypeError if wrong type is provided for index label.'"""
    with pytest.raises(TypeError):
        df.pivot_longer(index=index)


@pytest.mark.parametrize("df,column", column_error_checks)
def test_type_column_names(df, column):
    """Raise TypeError if wrong type is provided for the column label.'"""
    with pytest.raises(TypeError):
        df.pivot_longer(column_names=column)


@pytest.mark.parametrize("df,names_to", names_to_error_checks)
def test_type_names_to(df, names_to):
    """Raise TypeError if wrong type is provided for `names_to`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to)


@pytest.mark.parametrize("df,index", index_presence_checks)
def test_presence_index(df, index):
    """Raise ValueError if index does not exist."""
    with pytest.raises(ValueError):
        df.pivot_longer(index=index)


@pytest.mark.parametrize("df,column", column_presence_checks)
def test_presence_columns(df, column):
    """Raise ValueError if column does not exist."""
    with pytest.raises(ValueError):
        df.pivot_longer(column_names=column)


@pytest.mark.parametrize("df,names_to, names_sep", names_sep_not_required)
def test_name_sep_names_to_len(df, names_to, names_sep):
    """
    Raise ValueError if the `names_to` is a string, or `names_to` is a
    list/tuple and its length is one, and `names_sep` is provided."""
    with pytest.raises(ValueError):
        df.pivot_longer(names_to=names_to, names_sep=names_sep)


@pytest.mark.parametrize("df,names_to, names_sep", names_sep_type_check)
def test_name_sep_wrong_type(df, names_to, names_sep):
    """
    Raise TypeError if wrong type provided for `names_sep`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to, names_sep=names_sep)


@pytest.mark.parametrize(
    "df,names_to, names_pattern", names_pattern_type_check
)
def test_name_pattern_wrong_type(df, names_to, names_pattern):
    """
    Raise TypeError if wrong type provided for `names_pattern`."""
    with pytest.raises(TypeError):
        df.pivot_longer(names_to=names_to, names_pattern=names_pattern)

@pytest.mark.parametrize("df", multi_index_df)
def test_warning_multi_index(df):
    """Raise Warning if dataframe is a MultiIndex."""
    with pytest.warns(UserWarning):
        df.pivot_longer()

def test_both_names_sep_and_pattern():
    """Raise ValueError if `names_sep` and `names_pattern` is provided."""
    with pytest.raises(ValueError):
        df_checks.pivot_longer(
            names_to=["rar", "bar"], names_sep="-", names_pattern=r"\\d+"
        )


def test_values_to():
    """Raise TypeError if wrong type is provided for`values_to`."""
    with pytest.raises(TypeError):
        df_checks.pivot_longer(values_to=["salvo"])

# print(df1.pivot_longer(names_to=["year"], index=janitor.patterns(r"[^\d+]")))


# df2 = pd.DataFrame(
#     {
#         "country": ["United States", "Russia", "China"],
#         "vault_2012": [48.132, 46.36600000000001, 44.266000000000005],
#         "floor_2012": [45.36600000000001, 41.599, 40.833],
#         "vault_2016": [46.86600000000001, 45.733000000000004, 44.332],
#         "floor_2016": [45.998999999999995, 42.032, 42.066],
#     }
# )

# print(df2)

# print(
#     df2.pivot_longer(
#         index="country", names_to=("event", "year"), names_sep="_"
#     )
# )

# df3 = pd.DataFrame({'country': ['United States', 'Russia', 'China'],
# 'vault_2012_f': [48.132, 46.36600000000001, 44.266000000000005],
# 'vault_2012_m': [46.632, 46.86600000000001, 48.316],
# 'vault_2016_f': [46.86600000000001, 45.733000000000004, 44.332],
# 'vault_2016_m': [45.865, 46.033, 45.0],
# 'floor_2012_f': [45.36600000000001, 41.599, 40.833],
# 'floor_2012_m': [45.266000000000005, 45.308, 45.133],
# 'floor_2016_f': [45.998999999999995, 42.032, 42.066],
# 'floor_2016_m': [43.757, 44.766000000000005, 43.799]})

# print(df3)

# print(
#    df3.pivot_longer(
#        index="country", names_to=("event", "year", "gender"), names_sep="_"
#    )
# )

# df4 = pd.DataFrame(
#    {
#       "country": ["United States", "Russia", "China"],
#     "floor2012": [45.36600000000001, 41.599, 40.833],
#     "vault2016": [46.86600000000001, 45.733000000000004, 44.332],
#    "floor2016": [45.998999999999995, 42.032, 42.066],
# }
# )

# print(df4)

# print(
#    df4.pivot_longer(
#        index="country",
#        names_to=("event", "year"),
#        names_sep="_",
#        names_pattern="([A-Za-z]+)(\\d+)",
#    )
# )

# df5 = pd.DataFrame({'country': ['United States', 'Russia', 'China'],
# 'score_vault': [46.86600000000001, 45.733000000000004, 44.332],
# 'score_floor': [45.998999999999995, 42.032, 42.066]})

# print(df5)

# print(df5.pivot_longer(index='country', names_to='event', names_sep="_")


# df6 = pd.DataFrame(
#     {
#         "family": [1, 2, 3, 4, 5],
#         "dob_child1": [
#             "1998-11-26",
#             "1996-06-22",
#             "2002-07-11",
#             "2004-10-10",
#             "2000-12-05",
#         ],
#         "dob_child2": [
#             "2000-01-29",
#             np.nan,
#             "2004-04-05",
#             "2009-08-27",
#             "2005-02-28",
#         ],
#         "gender_child1": [1, 2, 2, 1, 2],
#         "gender_child2": [2.0, np.nan, 2.0, 1.0, 1.0],
#     }
# )

# print(df6)

# print(
#     df6.pivot_longer(
#         index="family", names_to=(".value", "child"), names_sep="_"
#     )
# )

# df7 = pd.DataFrame([{' id': 1,
#   'a1': ' a',
#   'a2': ' b',
#   'a3': ' c',
#   'A1': ' A',
#   'A2': ' B',
#   'A3': ' C'}])

# print(df7)

# print(df7.pivot_longer(column_names= janitor.patterns("^a|^A"), names_to=(".value", "instance"), names_pattern = "(\\w)(\\d)"))


# df8 = pd.DataFrame(
#     {
#         "off_loc": ["A", "B", "C", "D", "E", "F"],
#         "pt_loc": ["G", "H", "I", "J", "K", "L"],
#         "pt_lat": [
#             100.07548220000001,
#             75.191326,
#             122.65134479999999,
#             124.13553329999999,
#             124.13553329999999,
#             124.01028909999998,
#         ],
#         "off_lat": [
#             121.271083,
#             75.93845266,
#             135.043791,
#             134.51128400000002,
#             134.484374,
#             137.962195,
#         ],
#         "pt_long": [
#             4.472089953,
#             -144.387785,
#             -40.45611048,
#             -46.07156181,
#             -46.07156181,
#             -46.01594293,
#         ],
#         "off_long": [
#             -7.188632000000001,
#             -143.2288569,
#             21.242563,
#             40.937416999999996,
#             40.78472,
#             22.905889000000002,
#         ],
#     }
# )

# print(df8)

# print(df8.pivot_longer(names_to=("set", ".value"), names_pattern="(.+)_(.+)"))

# df9 = pd.DataFrame({'A1970': ['a', 'b', 'c'],
#  'A1980': ['d', 'e', 'f'],
#  'B1970': [2.5, 1.2, 0.7],
#  'B1980': [3.2, 1.3, 0.1],
#  'X': [-1.085631, 0.997345, 0.282978]})

# print(df9)

# print(df9.pivot_longer(index='X', names_to=(".value", "year"), names_pattern="([A-Z])(.+)"))

# df10 = pd.DataFrame(
#     {
#         "famid": [1, 1, 1, 2, 2, 2, 3, 3, 3],
#         "birth": [1, 2, 3, 1, 2, 3, 1, 2, 3],
#         "ht1": [2.8, 2.9, 2.2, 2, 1.8, 1.9, 2.2, 2.3, 2.1],
#         "ht2": [3.4, 3.8, 2.9, 3.2, 2.8, 2.4, 3.3, 3.4, 2.9],
#     }
# )


# print(df10)

# print(
#     df10.pivot_longer(
#         index=["famid", "birth"],
#         names_to=(".value", "age"),
#         names_pattern=r"(ht)(\d)",
#     )
# )

# df11 = pd.DataFrame(
#     {
#         "A(weekly)-2010": [0.548814, 0.7151890000000001, 0.602763],
#         "A(weekly)-2011": [0.544883, 0.423655, 0.645894],
#         "B(weekly)-2010": [0.437587, 0.8917729999999999, 0.9636629999999999],
#         "B(weekly)-2011": [0.383442, 0.791725, 0.528895],
#         "X": [0, 1, 1],

#     }
# )

# print(df11)

# print(
#     df11.pivot_longer(
#         index=["X"], names_to=(".value", "year"), names_sep="-"
#     )
# )

# df12 = pd.DataFrame({'id': ['A', 'B', 'C', 'D', 'E', 'F'],
#  'f_start': ['p', 'i', 'i', 'p', 'p', 'i'],
#  'd_start': ['2018-01-01',
#   '2019-04-01',
#   '2018-06-01',
#   '2019-12-01',
#   '2019-02-01',
#   '2018-04-01'],
#  'f_end': ['p', 'p', 'i', 'p', 'p', 'i'],
#  'd_end': ['2018-02-01',
#   '2020-01-01',
#   '2019-03-01',
#   '2020-05-01',
#   '2019-05-01',
#   '2018-07-01']})

# print(df12)

# print(df12.pivot_longer(index='id', names_to = ('.value' , 'status') ,
#                names_pattern = '(.*)_(.*)'))


# df13 = pd.DataFrame({'commune': ['A', 'B', 'C'],
#  'nuance_1': ['X', 'X', 'Z'],
#  'votes_1': [12, 10, 7],
#  'nuance_2': ['Y', 'Y', 'X'],
#  'votes_2': [20, 5, 2],
#  'nuance_3': ['Z', None, None],
#  'votes_3': [5.0, np.nan, np.nan]})

# print(df13)

# print(df13.pivot_longer(index='commune', names_to = ('.value' , 'numbers') ,
#                names_pattern = '(.*)_(.*)'))

# df14 = pd.DataFrame({'name': ['Wilbur', 'Petunia', 'Gregory'],
#  'a': [67, 80, 64],
#  'b': [56, 90, 50]})

# print(df14)

# print(df14.pivot_longer(column_names=['a', 'b'], names_to='drug',   values_to='heartrate'))
