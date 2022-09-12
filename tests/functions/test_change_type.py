import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal


@pytest.mark.functions
def test_change_type(dataframe):
    df = dataframe.change_type(column_name="a", dtype=float)
    assert df["a"].dtype == float


@pytest.mark.functions
def test_change_type_keep_values():
    df = pd.DataFrame(["a", 1, True], columns=["col1"])
    df = df.change_type(
        column_name="col1", dtype=float, ignore_exception="keep_values"
    )
    assert df.equals(pd.DataFrame(["a", 1, True], columns=["col1"]))


@pytest.mark.functions
def test_change_type_fillna():
    df = pd.DataFrame(["a", 1, True], columns=["col1"])
    df = df.change_type(
        column_name="col1", dtype=float, ignore_exception="fillna"
    )
    assert np.isnan(df.col1[0])


@pytest.mark.functions
def test_change_type_unknown_option():
    df = pd.DataFrame(["a", 1, True], columns=["col1"])
    with pytest.raises(Exception):
        df = df.change_type(
            column_name="col1", dtype=float, ignore_exception="blabla"
        )


@pytest.mark.functions
def test_change_type_raise_exception():
    df = pd.DataFrame(["a", 1, True], columns=["col1"])
    with pytest.raises(Exception):
        df = df.change_type(
            column_name="col1", dtype=float, ignore_exception=False
        )


@pytest.mark.functions
def test_original_data_type(dataframe):
    df = pd.DataFrame(range(3), columns=["col1"])
    df_original = df.copy()

    df.change_type("col1", dtype=str)

    # 'cols' is still int type not str type
    assert_frame_equal(df, df_original)
