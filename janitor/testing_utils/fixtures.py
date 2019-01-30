from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from janitor.testing_utils import date_data

TEST_DATA_DIR = (
    Path(__file__).resolve().parent.parent.parent / "tests" / "test_data"
)


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
def date_dataframe():
    df = pd.DataFrame(date_data.date_list, columns=["AMOUNT", "DATE"])
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


@pytest.fixture
def biodf():
    df = pd.read_csv(TEST_DATA_DIR / "sequences.tsv", sep="\t").clean_names()
    return df
