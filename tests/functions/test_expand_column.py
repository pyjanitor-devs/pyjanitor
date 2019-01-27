import pandas as pd


def test_expand_column():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data)
    expanded = df.expand_column("col1", sep=", ", concat=False)
    assert expanded.shape[1] == 6


def test_expand_and_concat():
    data = {
        "col1": ["A, B", "B, C, D", "E, F", "A, E, F"],
        "col2": [1, 2, 3, 4],
    }

    df = pd.DataFrame(data).expand_column("col1", sep=", ", concat=True)
    assert df.shape[1] == 8
