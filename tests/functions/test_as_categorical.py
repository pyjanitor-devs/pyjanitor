from janitor.functions import as_categorical
import pandas as pd
import pytest

def test_check_type_dataframe():
    "Raise TypeError if `df` is not a dataframe."
    df = pd.Series([60, 70])
    with pytest.raises(TypeError):
        as_categorical(df)


def test_check_type_column_name():
    "Raise TypeError if `column_name` is not a string type."
    df = pd.DataFrame({"col1": [60,70]})
    with pytest.raises(TypeError):
        as_categorical(df = df, column_name=1)
