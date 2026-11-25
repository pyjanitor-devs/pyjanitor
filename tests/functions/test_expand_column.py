import pandas as pd
import pytest


@pytest.mark.functions
def test_expand_column():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data)
    expanded_df = df.expand_column(column_name="col1", sep=", ", concat=False)
    assert expanded_df.shape[1] == 6


@pytest.mark.functions
def test_expand_column_drop_first():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data)
    expanded_df = df.expand_column(
        column_name="col1", sep=", ", concat=False, drop_first=True
    )

    assert list(expanded_df.columns) == ["B", "C", "D", "E", "F"]
    assert expanded_df.shape[1] == 5


@pytest.mark.functions
def test_expand_and_concat():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data).expand_column(column_name="col1", sep=", ", concat=True)
    assert df.shape[1] == 8


@pytest.mark.functions
def test_expand_and_concat_drop_first():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data).expand_column(
        column_name="col1", sep=", ", concat=True, drop_first=True
    )

    assert list(df.columns) == ["col1", "col2", "B", "C", "D", "E", "F"]
    assert df.shape[1] == 7


@pytest.mark.functions
def test_sep_default_parameter():
    """Test that the default parameter is a pipe character `|`."""
    df = pd.DataFrame(
        {
            "col1": ["A|B", "B|C|D", "E|F", "A|E|F"],
            "col2": [1, 2, 3, 4],
        }
    )
    result = df.expand_column("col1")

    assert result.shape[1] == 8


@pytest.mark.functions
def test_expand_and_concat_duplicate_index():
    index = pd.Index([2, 1, 2], name="row")
    df = pd.DataFrame({"tags": ["A", "A|B", "B"]}, index=index)

    result = df.expand_column("tags")

    expected = df.assign(A=[1, 1, 0], B=[0, 1, 1])
    pd.testing.assert_frame_equal(result, expected)


@pytest.mark.functions
def test_expand_duplicate_index_overlapping_columns():
    df = pd.DataFrame({"tags": ["A", "B"], "A": [3, 4]}, index=[1, 1])

    with pytest.raises(ValueError, match="columns overlap"):
        df.expand_column("tags")
