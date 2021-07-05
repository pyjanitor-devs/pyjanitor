import numpy as np
import pandas as pd
import pytest
import janitor
from pandas.testing import assert_frame_equal


@pytest.fixture
def left_df():
    return pd.DataFrame({"col_a": [1, 2, 3], "col_b": ["A", "B", "C"]})


@pytest.fixture
def right_df():
    return pd.DataFrame({"col_a": [0, 2, 3], "col_c": ["Z", "X", "Y"]})


@pytest.fixture
def sequence():
    return [1, 2, 3]


@pytest.fixture
def multiIndex_df():
    frame = pd.DataFrame({"col_a": [0, 2, 3], "col_b": ["Z", "X", "Y"]})
    frame.columns = [["A", "col_a"], ["B", "col_b"]]
    return frame


def test_df_MultiIndex(multiIndex_df, right_df):
    """Raise ValueError if `df` has MultiIndex columns"""
    with pytest.raises(
        ValueError, match="MultiIndex columns are not supported for non-equi joins."
    ):
        multiIndex_df.le_join(right_df, "col_a", "col_a")


def test_right_MultiIndex(left_df, multiIndex_df):
    """Raise ValueError if `right` has MultiIndex columns"""
    with pytest.raises(
        ValueError, match="MultiIndex columns are not supported for non-equi joins."
    ):
        left_df.le_join(multiIndex_df, "col_a", "col_a")


def test_right_not_Series(left_df, sequence):
    """Raise TypeError if `right` is not DataFrame/Series"""
    with pytest.raises(TypeError):
        left_df.le_join(sequence, "col_a", "col_a")


def test_right_unnamed_Series(left_df, sequence):
    """Raise ValueError if `right` is not a named Series"""
    sequence = pd.Series(sequence)
    with pytest.raises(
        ValueError, match="Unnamed Series are not supported for non-equi joins."
    ):
        left_df.le_join(sequence, ("col_a", "col_a", "le"))


def test_wrong_type_suffixes(left_df, right_df):
    """Raise TypeError if `suffixes` is not a tuple."""
    with pytest.raises(TypeError):
        left_df.le_join(right_df, "col_a", "col_a", suffixes=None)


def test_wrong_length_suffixes(left_df, right_df):
    """Raise TypeError if `suffixes` length != 2."""
    with pytest.raises(
        ValueError, match="`suffixes` argument must be a 2-length tuple"
    ):
        left_df.le_join(
            right_df, "col_a", "col_a", suffixes=("_x",)
        )


def test_suffixes_None(left_df, right_df):
    """Raise ValueError if `suffixes` is (None, None)."""
    with pytest.raises(
        ValueError, match="At least one of the suffixes should be non-null."
    ):
        left_df.le_join(
            right_df, "col_a", "col_a", suffixes=(None, None)
        )


def test_wrong_type_suffix(left_df, right_df):
    """Raise TypeError if one of the `suffixes` is not None or a string type."""
    with pytest.raises(TypeError):
        left_df.le_join(
            right_df, "col_a", "col_a", suffixes=("_x", 1)
        )

# test that left and right column r strings, and exist in df and right respectively.
# check that left and right are both either numeric/string/datetime
# check if suffix already exists in df and right
# check for less than if left_min > right_max
# check for greater than if left_max < right_min

# from janitor.utils import (
#     _generic_less_than_inequality,
#     _generic_greater_than_inequality,
# )

# df1 = pd.DataFrame(
#     [
#         {"x": "b", "y": 1, "v": 1},
#         {"x": "b", "y": 3, "v": 2},
#         {"x": "b", "y": 6, "v": 3},
#         {"x": "a", "y": 1, "v": 4},
#         {"x": "a", "y": 3, "v": 5},
#         {"x": "a", "y": 6, "v": 6},
#         {"x": "c", "y": 1, "v": 7},
#         {"x": "c", "y": 3, "v": 8},
#         {"x": "c", "y": 6, "v": 9},
#         {"x": "c", "y": np.nan, "v": 9},
#     ]
# )

# df2 = pd.DataFrame(
#     [
#         {"x": "c", "v": 8, "foo": 4},
#         {"x": "b", "v": 7, "foo": 12},
#         {"x": "b", "v": 7, "foo": None},
#         {"x": "b", "v": 7, "foo": None},
#         {"x": "b", "v": 7, "foo": 4},
#         {"x": "b", "v": 7, "foo": 3},
#     ]
# )

# # outcome = _generic_greater_than_inequality(df1, df2, 'y', 'foo', strict = True)
# outcome = janitor.gt_join(df1, df2, "v", "foo")#, ("v", "v", "gt"), ("x", "x", "le"))

# print(outcome, end='\n\n')
# print(df1)

# print(df1.le_join(df3, ("col_a", "col_a", "le")))

# def test_type_right():
#     """Raise TypeError if wrong type is provided for `right`."""
#     with pytest.raises(TypeError):
#         df1.conditional_join(df2, le_join("col_a", "col_c"))


# def test_right_unnamed_Series():
#     """Raise ValueError if `right` is an unnamed Series."""
#     with pytest.raises(ValueError):
#         df1.conditional_join(df_2, le_join("col_a", "col_c"))


# def test_wrong_condition_type():
#     """Raise TypeError if wrong type is provided for condition."""
#     with pytest.raises(TypeError):
#         df1.conditional_join(df3, lte_join("col_a", "col_c"))


# def test_wrong_column_type():
#     """Raise TypeError if wrong type is provided for the columns."""
#     with pytest.raises(TypeError):
#         df1.conditional_join(df3, le_join(1, "col_c"))


# def test_wrong_column_presence():
#     """Raise ValueError if column is not found in the dataframe."""
#     with pytest.raises(ValueError):
#         df1.conditional_join(df3, le_join("col_a", "col_b"))


# def test_no_condition():
#     """Raise ValueError if no condition is provided."""
#     with pytest.raises(ValueError):
#         df1.conditional_join(df3)


# def test_wrong_column_dtype():
#     """
#     Raise ValueError if dtypes of columns
#     is not one of numeric, date, or string.
#     """
#     with pytest.raises(ValueError):
#         df1.conditional_join(df_3, le_join("col_a", "col_c"))


# def test_more_than_two_conditions():
#     """Raise ValueError if len(conditions) > 2"""
#     with pytest.raises(ValueError):
#         df1.conditional_join(
#             df3,
#             lt_join("col_a", "col_a"),
#             le_join("col_a", "col_c"),
#             ge_join("col_b", "col_c"),
#         )


# df_multi = df1.copy()
# df_multi.columns = [list("AB"), list("CD")]

# multi_index_columns = [
#     (df_multi, le_join(("A", "B"), "col_c"), df2),
#     (df1, le_join("col_a", ("A", "B")), df_multi),
# ]


# @pytest.mark.parametrize("df_left,condition,df_right", multi_index_columns)
# def test_multiIndex_columns(df_left, condition, df_right):
#     """Raise ValueError if columns are MultiIndex."""
#     with pytest.raises(ValueError):
#         df_left.conditional_join(df_right, condition)


# @pytest.fixture
# def df_left():
#     return pd.DataFrame(
#         [
#             {"x": "b", "y": 1, "v": 1},
#             {"x": "b", "y": 3, "v": 2},
#             {"x": "b", "y": 6, "v": 3},
#             {"x": "a", "y": 1, "v": 4},
#             {"x": "a", "y": 3, "v": 5},
#             {"x": "a", "y": 6, "v": 6},
#             {"x": "c", "y": 1, "v": 7},
#             {"x": "c", "y": 3, "v": 8},
#             {"x": "c", "y": 6, "v": 9},
#         ]
#     )


# @pytest.fixture
# def df_right():
#     return pd.DataFrame(
#         [{"x": "c", "v": 8, "foo": 4}, {"x": "b", "v": 7, "foo": 2}]
#     )


# def test_less_than_join(df_left, df_right):
#     """Test output of less than join."""
#     result = df_left.conditional_join(df_right, le_join("y", "foo"))
#     pass
# df1 = pd.DataFrame({'col_a': [1,2,3], 'col_b': ["A", "B", "C"]})
# df2 = pd.DataFrame({'col_a': [0, 2, 3], 'col_c': ["Z", "X", "Y"]})
# df2 = pd.Series([1,2,3], name='ragnar').astype('category')

# df1 = pd.DataFrame(dict(col_a = [1,2,5,np.nan], col_b=pd.Series(['A','B','B','C'], dtype='string')))
# df2 = pd.DataFrame({'col_a': [2,0, 3,np.nan], 'col_c': pd.Series(["Z", "X", "Y","A"], dtype='string')})
# print(df1)
# print(df2)
# print(df1.dtypes)
# print(df2.dtypes)
# print(df1.conditional_join(df2, le_join("col_a", "col_a")))


# df1 = pd.DataFrame(
#     [
#         {"x": "b", "y": 1, "v": 1},
#         {"x": "b", "y": 3, "v": 2},
#         {"x": "b", "y": 6, "v": 3},
#         {"x": "a", "y": 1, "v": 4},
#         {"x": "a", "y": 3, "v": 5},
#         {"x": "a", "y": 6, "v": 6},
#         {"x": "c", "y": 1, "v": 7},
#         {"x": "c", "y": 3, "v": 8},
#         {"x": "c", "y": 6, "v": 9},
#         {"x": "c", "y": np.nan, "v": 9},
#     ]
# )

# df2 = pd.DataFrame(
#     [{"x": "c", "v": 8, "foo": 4}, {"x": "b", "v": 7, "foo": 2}, {"x": "b", "v": 7, "foo": None}]
# )

# result = janitor.lt_join(df1, df2, 'v', 'v')

# print(df1, end='\n\n')
# print(result)
# result = df1.conditional_join(df2, le_join("v", "v"))

# print(df1, end="\n")
# print(df2, end="\n")
# print(result)

# df = pd.DataFrame(
#     {
#         "Origin": {
#             1: "A",
#             6: "A",
#             11: "A",
#             16: "A",
#             21: "B",
#             26: "B",
#             31: "C",
#             36: "C",
#         },
#         "Destination": {
#             1: "B",
#             6: "B",
#             11: "C",
#             16: "C",
#             21: "Z",
#             26: "Z",
#             31: "Z",
#             36: "Z",
#         },
#         "Dept_Time": {
#             1: pd.Timestamp("2019-03-30 17:31:00"),
#             6: pd.Timestamp("2019-05-16 17:32:00"),
#             11: pd.Timestamp("2019-04-01 08:30:00"),
#             16: pd.Timestamp("2019-06-09 08:20:00"),
#             21: pd.Timestamp("2019-07-26 08:31:00"),
#             26: pd.Timestamp("2019-03-31 06:16:00"),
#             31: pd.Timestamp("2019-07-03 23:52:00"),
#             36: pd.Timestamp("2019-03-27 17:31:00"),
#         },
#         "Arrv_Time": {
#             1: pd.Timestamp("2019-03-30 23:23:00"),
#             6: pd.Timestamp("2019-05-16 23:22:00"),
#             11: pd.Timestamp("2019-04-01 14:22:00"),
#             16: pd.Timestamp("2019-06-09 14:18:00"),
#             21: pd.Timestamp("2019-07-26 14:23:00"),
#             26: pd.Timestamp("2019-06-18 05:00:00"),
#             31: pd.Timestamp("2019-07-04 05:36:00"),
#             36: pd.Timestamp("2019-03-27 23:23:00"),
#         },
#     }
# )


# print(df.lt_join(df, 'Dept_Time', 'Arrv_Time'))

# df_1 = pd.DataFrame(
#     [
#         {
#             "timestamp": pd.Timestamp("2016-05-14 10:54:33"),
#             "A": 0.020228,
#             "B": 0.026572,
#         },
#         {
#             "timestamp": pd.Timestamp("2016-05-14 10:54:34"),
#             "A": 0.05778,
#             "B": 0.175499,
#         },
#         {
#             "timestamp": pd.Timestamp("2016-05-14 10:54:35"),
#             "A": 0.098808,
#             "B": 0.620986,
#         },
#         {
#             "timestamp": pd.Timestamp("2016-05-14 10:54:36"),
#             "A": 0.158789,
#             "B": 1.014819,
#         },
#         {
#             "timestamp": pd.Timestamp("2016-05-14 10:54:39"),
#             "A": 0.038129,
#             "B": 2.38459,
#         },
#     ]
# )


# df_2 = pd.DataFrame(
#     [
#         {
#             "start": pd.Timestamp("2016-05-14 10:54:31"),
#             "end": pd.Timestamp("2016-05-14 10:54:33"),
#             "event": "E1",
#         },
#         {
#             "start": pd.Timestamp("2016-05-14 10:54:34"),
#             "end": pd.Timestamp("2016-05-14 10:54:37"),
#             "event": "E2",
#         },
#         {
#             "start": pd.Timestamp("2016-05-14 10:54:38"),
#             "end": pd.Timestamp("2016-05-14 10:54:42"),
#             "event": "E3",
#         },
#     ]
# )

