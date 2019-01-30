import pandas as pd
import pytest

from janitor.errors import JanitorError
from janitor.testing_utils.fixtures import dataframe


@pytest.mark.functions
def test_single_column_label_encode():
    df = pd.DataFrame(
        {"a": ["hello", "hello", "sup"], "b": [1, 2, 3]}
    ).label_encode(columns="a")
    assert "a_enc" in df.columns


@pytest.mark.functions
def test_single_column_fail_label_encode():
    with pytest.raises(AssertionError):
        pd.DataFrame(
            {"a": ["hello", "hello", "sup"], "b": [1, 2, 3]}
        ).label_encode(
            columns="c"
        )  # noqa: 841


@pytest.mark.functions
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


@pytest.mark.functions
def test_label_encode_invalid_input(dataframe):
    with pytest.raises(JanitorError):
        dataframe.label_encode(1)
