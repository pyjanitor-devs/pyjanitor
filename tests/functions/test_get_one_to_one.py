import pandas as pd
import pytest

import janitor


@pytest.mark.functions
def test_get_one_to_one_basic():
    df = pd.DataFrame(
        {
            "Lab_Test_Long": ["Cholesterol, LDL", "Cholesterol, LDL", "Glucose"],
            "Lab_Test_Short": ["CLDL", "CLDL", "GLUC"],
            "LOINC": [12345, 12345, 54321],
            "Person": ["Sam", "Bill", "Sam"],
        }
    )
    assert janitor.get_one_to_one(df) == [["Lab_Test_Long", "Lab_Test_Short", "LOINC"]]


@pytest.mark.functions
def test_get_one_to_one_no_matches():
    df = pd.DataFrame({"a": [1, 2, 1], "b": [1, 2, 2]})
    assert janitor.get_one_to_one(df) == []


@pytest.mark.functions
def test_get_one_to_one_duplicate_columns_error():
    df = pd.DataFrame([[1, 2]], columns=["a", "a"])
    with pytest.raises(ValueError, match="unique column names"):
        janitor.get_one_to_one(df)


@pytest.mark.functions
def test_get_one_to_one_empty_dataframe_error():
    with pytest.raises(ValueError, match="at least one column"):
        janitor.get_one_to_one(pd.DataFrame())
