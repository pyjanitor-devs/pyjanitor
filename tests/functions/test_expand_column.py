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
def test_expand_and_concat():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data).expand_column(column_name="col1", sep=", ", concat=True)
    assert df.shape[1] == 8


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
def test_expand_column_drop_first_concat_false():
    """``drop_first=True`` drops the first (lex-sorted) dummy column.

    Mirrors ``pandas.get_dummies(drop_first=True)``. Useful before linear
    regression to avoid the dummy-variable trap. See issue #368.
    """
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }
    df = pd.DataFrame(data)
    expanded = df.expand_column(
        column_name="col1", sep=", ", concat=False, drop_first=True
    )
    # Pre-fix: would have 6 columns; with drop_first the leading "A" goes.
    assert expanded.shape[1] == 5
    assert "A" not in expanded.columns
    assert list(expanded.columns) == ["B", "C", "D", "E", "F"]


@pytest.mark.functions
def test_expand_column_drop_first_concat_true():
    """drop_first composes with concat=True (the default)."""
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }
    df = pd.DataFrame(data).expand_column(
        column_name="col1", sep=", ", drop_first=True
    )
    # original 2 columns + 5 dummies (one dropped)
    assert df.shape[1] == 7
    assert "A" not in df.columns


@pytest.mark.functions
def test_expand_column_drop_first_default_off():
    """``drop_first`` defaults to ``False`` so existing callers see no change."""
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }
    df = pd.DataFrame(data)
    default = df.expand_column(column_name="col1", sep=", ", concat=False)
    explicit = df.expand_column(
        column_name="col1", sep=", ", concat=False, drop_first=False
    )
    pd.testing.assert_frame_equal(default, explicit)
