import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


df = pd.DataFrame(
    [
        {"rank": 1, "pet_type": np.nan, "breed": "Boston Terrier", "owner": "sam"},
        {"rank": 2, "pet_type": np.nan, "breed": "Retrievers (Labrador)", "owner": "ogor"},
        {"rank": 3, "pet_type": np.nan, "breed": "Retrievers (Golden)", "owner": "nathan"},
        {"rank": 4, "pet_type": np.nan, "breed": "French Bulldogs", "owner": np.nan},
        {"rank": 5, "pet_type": np.nan, "breed": "Bulldogs", "owner": np.nan},
        {"rank": 6, "pet_type": "Dog", "breed": "Beagles", "owner": np.nan},
        {"rank": 1, "pet_type": np.nan, "breed": "Persian", "owner": np.nan},
        {"rank": 2, "pet_type": np.nan, "breed": "Maine Coon", "owner":"ragnar"},
        {"rank": 3, "pet_type": np.nan, "breed": "Ragdoll", "owner":np.nan},
        {"rank": 4, "pet_type": np.nan, "breed": "Exotic", "owner":np.nan},
        {"rank": 5, "pet_type": np.nan, "breed": "Siamese", "owner": np.nan},
        {"rank": 6, "pet_type": "Cat", "breed": "American Short", "owner": "adaora"},
    ]
)

def test_fill_column():
    """ Fill down on a single column with default direction."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].ffill()
    result = df.fill(columns = "pet_type")
    assert_frame_equal(result, expected)

def test_fill_column_up():
    """ Fill up on a single column."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].bfill()
    result = df.fill(columns = "pet_type", directions = "up")
    assert_frame_equal(result, expected)

def test_fill_column_updown():
    """ Fill upwards, then downwards on a single column."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].bfill().ffill()
    result = df.fill(columns = "pet_type", directions = "updown")
    assert_frame_equal(result, expected)

def test_fill_column_down_up():
    """ Fill downwards, then upwards on a single column."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].ffill().bfill()
    result = df.fill(columns = "pet_type", directions = "downup")
    assert_frame_equal(result, expected)

def test_fill_multiple_columns():
    """ Fill on multiple columns with a single direction."""
    expected = df.copy()
    expected.loc[:, ["pet_type", "owner"]] = expected.loc[:, ["pet_type","owner"]].ffill()
    result = df.fill(columns = "pet_type,owner")
    assert_frame_equal(result, expected)

def test_fill_multiple_columns_multiple_directions():
    """ Fill on multiple columns with different directions."""
    expected = df.copy()
    expected.loc[:, "pet_type"] = expected.loc[:, "pet_type"].ffill()
    expected.loc[:, "owner"] = expected.loc[:, "owner"].bfill()
    result = df.fill(columns = ("pet_type", "owner"), directions= "down,up")
    assert_frame_equal(result, expected)

def test_fill_uneven_lengths():
    """ Raise Value Error if number of directions is greater than one and 
    is unequal to number of columns."""
    with pytest.raises(ValueError):
        df.fill(columns = ["pet_type", "owner"], directions = ["up", "up", "down"])

def test_wrong_column_instance():
    """ Raise Type Error if columns argument is not a List/Tuple/str."""
    with pytest.raises(TypeError):
        df.fill(columns = {"pet_type"})   

def test_wrong_direction_instance():
    """ Raise Type Error if directions argument is not a List/Tuple/str."""
    with pytest.raises(TypeError):
        df.fill(columns = ["pet_type"], directions = {"pet_type" : "up"})   

def test_wrong_column_name():
    """ Raise Value Error if wrong column name is provided."""
    with pytest.raises(ValueError):
        df.fill(columns = "PetType")   

def test_wrong_direction():
    """ Raise Value Error if wrong direction is provided."""
    with pytest.raises(ValueError):
        df.fill(columns = ("pet_type", ), directions = "upanddown")   