import pytest
from pandas import testing


@pytest.mark.functions
def test_move_row(dataframe):
    """
    Test function move() for rows with defaults.
    Case with row labels being integers.
    """
    # Setup
    source = 1
    target = 3
    row = dataframe.loc[source, :]

    # Exercise
    result = dataframe.move(source=source, target=target, axis=0)

    # Verify
    testing.assert_series_equal(result.iloc[target - 1, :], row)


@pytest.mark.functions
def test_move_row_after(dataframe):
    """
    Test function move() for rows with position = 'after'.
    Case with row labels being integers.
    """
    # Setup
    source = 1
    target = 3
    row = dataframe.loc[source, :]

    # Exercise
    result = dataframe.move(
        source=source, target=target, position="after", axis=0
    )

    # Verify
    testing.assert_series_equal(result.iloc[target, :], row)


@pytest.mark.functions
def test_move_row_strings(dataframe):
    """
    Test function move() for rows with defaults.
    Case with row labels being strings.
    """
    # Setup
    dataframe = dataframe.set_index("animals@#$%^").drop_duplicates()
    rows = dataframe.index
    source_index = 1
    target_index = 2
    source = rows[source_index]
    target = rows[target_index]
    row = dataframe.loc[source, :]

    # Exercise
    result = dataframe.move(source=source, target=target, axis=0)

    # Verify
    testing.assert_series_equal(result.iloc[target_index - 1, :], row)


@pytest.mark.functions
def test_move_row_after_strings(dataframe):
    """
    Test function move() for rows with position = 'after'.
    Case with row labels being strings.
    """
    # Setup
    dataframe = dataframe.set_index("animals@#$%^").drop_duplicates()
    rows = dataframe.index
    source_index = 1
    target_index = 2
    source = rows[source_index]
    target = rows[target_index]
    row = dataframe.loc[source, :]

    # Exercise
    result = dataframe.move(
        source=source, target=target, position="after", axis=0
    )

    # Verify
    testing.assert_series_equal(result.iloc[target_index, :], row)


@pytest.mark.functions
def test_move_col(dataframe):
    """
    Test function move() for columns with defaults.
    """
    # Setup
    columns = dataframe.columns
    source_index = 1
    target_index = 3
    source = columns[source_index]
    target = columns[target_index]
    col = dataframe[source]

    # Exercise
    result = dataframe.move(source=source, target=target, axis=1)

    # Verify
    testing.assert_series_equal(result.iloc[:, target_index - 1], col)


@pytest.mark.functions
def test_move_col_after(dataframe):
    """
    Test function move() for columns with position = 'after'.
    """
    # Setup
    columns = dataframe.columns
    source_index = 1
    target_index = 3
    source = columns[source_index]
    target = columns[target_index]
    col = dataframe[source]

    # Exercise
    result = dataframe.move(
        source=source, target=target, position="after", axis=1
    )

    # Verify
    testing.assert_series_equal(result.iloc[:, target_index], col)
