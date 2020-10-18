from itertools import product
from janitor.functions import collapse_levels

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.fixture
def df_checks_output():
    return pd.DataFrame(
        {
            "geoid": [1, 1, 13, 13],
            "name": ["Alabama", "Alabama", "Georgia", "Georgia"],
            "variable": [
                "pop_renter",
                "median_rent",
                "pop_renter",
                "median_rent",
            ],
            "estimate": [1434765, 747, 3592422, 927],
            "error": [16736, 3, 33385, 3],
        }
    )


combinations = [(pd.DataFrame({'geoid': [1, 1, 1, 1, 13, 13, 13, 13],
 'name': ['Alabama',
  'Alabama',
  'Alabama',
  'Alabama',
  'Georgia',
  'Georgia',
  'Georgia',
  'Georgia'],
 'variable': ['pop_renter',
  'pop_renter',
  'median_rent',
  'median_rent',
  'pop_renter',
  'pop_renter',
  'median_rent',
  'median_rent'],
 'measure': ['estimate',
  'error',
  'estimate',
  'error',
  'estimate',
  'error',
  'estimate',
  'error'],
 'value': [1434765, 16736, 747, 3, 3592422, 33385, 927, 3]}), 

pd.DataFrame({'geoid': [1, 13],
 'name': ['Alabama', 'Georgia'],
 'pop_renter_estimate': [1434765, 3592422],
 'pop_renter_error': [16736, 33385],
 'median_rent_estimate': [747, 927],
 'median_rent_error': [3, 3]}),

  ['geoid','name'],
  ['variable','measure'],
  'value',
  None
  
  
  
  
  ),

(pd.DataFrame({'geoid': [1, 1, 1, 1, 13, 13, 13, 13],
 'name': ['Alabama',
  'Alabama',
  'Alabama',
  'Alabama',
  'Georgia',
  'Georgia',
  'Georgia',
  'Georgia'],
 'variable': ['pop_renter',
  'pop_renter',
  'median_rent',
  'median_rent',
  'pop_renter',
  'pop_renter',
  'median_rent',
  'median_rent'],
 'measure': ['estimate',
  'error',
  'estimate',
  'error',
  'estimate',
  'error',
  'estimate',
  'error'],
 'value': [1434765, 16736, 747, 3, 3592422, 33385, 927, 3]}), 

pd.DataFrame({'geoid': [1, 13],
 'name': ['Alabama', 'Georgia'],
 'pop_renter_estimate': [1434765, 3592422],
 'pop_renter_error': [16736, 33385],
 'median_rent_estimate': [747, 927],
 'median_rent_error': [3, 3]}),

  ['geoid','name'],
  ['variable','measure'],
  None, None
  
  
  
  
  ),


(pd.DataFrame({'family': ['Kelly', 'Kelly', 'Quin', 'Quin'],
 'name': ['Mark', 'Scott', 'Tegan', 'Sara'],
 'n': [1, 2, 1, 2]}),
 
 pd.DataFrame({'family': ['Kelly', 'Quin'], 1: ['Mark', 'Tegan'], 2: ['Scott', 'Sara']}
),

'family', 'n', 'name', None
 
 
 
 ),

(pd.DataFrame({'family': ['Kelly', 'Kelly', 'Quin', 'Quin'],
 'name': ['Mark', 'Scott', 'Tegan', 'Sara'],
 'n': [1, 2, 1, 2]}),
 
 pd.DataFrame({'family': ['Kelly', 'Quin'], 'name1': ['Mark', 'Tegan'], 'name2': ['Scott', 'Sara']}
),

'family', 'n', 'name', 'name'
 
 
 
 ),

 (pd.DataFrame({'geoid': [1, 1, 13, 13],
 'name': ['Alabama', 'Alabama', 'Georgia', 'Georgia'],
 'variable': ['pop_renter', 'median_rent', 'pop_renter', 'median_rent'],
 'estimate': [1434765, 747, 3592422, 927],
 'error': [16736, 3, 33385, 3]}),

 pd.DataFrame({'geoid': [1, 13],
 'name': ['Alabama', 'Georgia'],
 'estimate_pop_renter': [1434765, 3592422],
 'estimate_median_rent': [747, 927],
 'error_pop_renter': [16736, 33385],
 'error_median_rent': [3, 3]}),

 ['geoid', 'name'], 'variable', ['estimate','error'], None
 
 )


]





def test_type_index1(df_checks_output, index={"geoid"}):
    "Raise TypeError if wrong type is provided for the `index`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index=index, names_from="variable")


def test_type_index2(df_checks_output, index=("geoid", "name")):
    "Raise TypeError if wrong type is provided for the `index`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index=index, names_from="variable")


def test_type_names_from1(df_checks_output, names_from={"variable"}):
    "Raise TypeError if wrong type is provided for `names_from`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_type_names_from2(df_checks_output, names_from=("variable",)):
    "Raise TypeError if wrong type is provided for `names_from`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_names_from_None(df_checks_output, names_from=None):
    "Raise ValueError if no value is provided for ``names_from``."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_presence_index1(df_checks_output, index="geo"):
    "Raise ValueError if labels in `index` do not exist."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index=index, names_from="variable")


def test_presence_index2(df_checks_output, index=["geoid", "Name"]):
    "Raise ValueError if labels in `index` do not exist."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index=index, names_from="variable")


def test_presence_names_from1(df_checks_output, names_from="estmt"):
    "Raise ValueError if labels in `names_from` do not exist."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_presence_names_from2(df_checks_output, names_from=["estimat"]):
    "Raise ValueError if labels in `names_from` do not exist."
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(index="geoid", names_from=names_from)


def test_values_from_first_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], values_from_first=2
):
    "Raise TypeError if the wrong type is provided for `values_from_first`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=names_from,
            values_from_first=values_from_first,
        )

def test_collapse_levels_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], collapse_levels=2
):
    "Raise TypeError if the wrong type is provided for `collapse_levels`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=names_from,
            collapse_levels = collapse_levels,
        )


def test_name_prefix_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], names_prefix=1
):
    "Raise TypeError if the wrong type is provided for `names_prefix`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, names_prefix=names_prefix
        )


def test_name_suffix_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], names_suffix=1
):
    "Raise TypeError if the wrong type is provided for `names_prefix`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, names_suffix=names_suffix
        )


def test_name_sep_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], names_sep=1
):
    "Raise TypeError if the wrong type is provided for `names_sep`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, names_sep=names_sep
        )


def test_values_from_len_equal_1(
    df_checks_output,
):
    """
    Raise ValueError if the length of `values_from` is 1 and
    `values_from_first` is False.
    """
    with pytest.raises(ValueError):
        df_checks_output.pivot_wider(
            index="name",
            names_from=["estimate", "variable"],
            values_from="error",
            values_from_first=False,
        )


def test_fill_value_wrong_type(
    df_checks_output, names_from=["estimate", "variable"], fill_value={2}
):
    "Raise TypeError if the wrong type is provided for `fill_value`."
    with pytest.raises(TypeError):
        df_checks_output.pivot_wider(
            index="name", names_from=names_from, fill_value=fill_value
        )


def test_non_unique_index_names_from_combination():
    """
    Raise ValueError for non-unique combination of
    `index` and `names_from`.
    """
    df = pd.DataFrame(
        {"A": ["A", "A", "A"], "L": ["L", "L", "L"], "numbers": [30, 54, 25]}
    )
    with pytest.raises(ValueError):
        df.pivot_wider(index="A", names_from="L")


def pivot_longer_wider_longer():
    """
    Test that transformation from pivot_longer to wider and
    back to longer returns the same source dataframe.
    """
    df = pd.DataFrame(
        {
            "name": ["Wilbur", "Petunia", "Gregory"],
            "a": [67, 80, 64],
            "b": [56, 90, 50],
        }
    )

    result = df.pivot_longer(
        column_names=["a", "b"], names_to="drug", values_to="heartrate"
    ).pivot_wider(index="name", names_from="drug", values_from="heartrate")

    assert_frame_equal(result, df)

@pytest.mark.parametrize(
    "df_in,df_out,index,names_from,values_from, names_prefix", combinations
)
def test_pivot_wider_various(
    df_in, df_out, index, names_from, values_from, names_prefix
):
    """
    Test `pivot_wider` function with various combinations.
    """
    result = df_in.pivot_wider(
        index=index, names_from=names_from, values_from=values_from, names_prefix= names_prefix
    )
    assert_frame_equal(result, df_out)


def test_collapse_levels_false():
    "Test output if `collapse_levels` is False."
    
    df_collapse = pd.DataFrame({'foo': ['one', 'one', 'one', 'two', 'two',
                           'two'],
                   'bar': ['A', 'B', 'C', 'A', 'B', 'C'],
                   'baz': [1, 2, 3, 4, 5, 6],
                   'zoo': ['x', 'y', 'z', 'q', 'w', 't']})


    result = df_collapse.pivot_wider(index='foo', names_from='bar', values_from=["baz","zoo"], collapse_levels=False)

    assert_frame_equal(result, df_collapse.pivot(index='foo', columns='bar', values=["baz","zoo"]), check_dtype=False)


def test_fill_values():
    "Test output if `fill_value` is provided."

    df_fill_value = pd.DataFrame(
    {
        "lev1": [1, 1, 1, 2, 2, 2],
        "lev2": [1, 1, 2, 1, 1, 2],
        "lev3": [1, 2, 1, 2, 1, 2],
        "lev4": [1, 2, 3, 4, 5, 6],
        "values": [0, 1, 2, 3, 4, 5],
    }
)

    result = df_fill_value.pivot_wider(index=["lev1", "lev2"],
    names_from=["lev3"],
    values_from="values",
    collapse_levels=False, fill_value = 0)

    expected_output = pd.DataFrame(
    {
        1: {(1, 1): 0, (1, 2): 2, (2, 1): 4, (2, 2): 0},
        2: {(1, 1): 1, (1, 2): 0, (2, 1): 3, (2, 2): 5},
    },
    index=pd.MultiIndex.from_tuples(
        [(1, 1), (1, 2), (2, 1), (2, 2)], names=["lev1", "lev2"]
    ),
    columns=pd.Int64Index([1, 2], dtype="int64", name="lev3"),
)

    assert_frame_equal(result, expected_output)


df_fill_value = pd.DataFrame(
    {
        "lev1": [1, 1, 1, 2, 2, 2],
        "lev2": [1, 1, 2, 1, 1, 2],
        "lev3": [1, 2, 1, 2, 1, 2],
        "lev4": [1, 2, 3, 4, 5, 6],
        "values": [0, 1, 2, 3, 4, 5],
    }
)

result = df_fill_value.pivot_wider(index=["lev1", "lev2"],
    names_from=["lev3"],
    values_from="values",
    collapse_levels=False, fill_value = 0)


print(result)