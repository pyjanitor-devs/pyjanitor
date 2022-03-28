"""
Testing strategies are placed here.
"""

import numpy as np
from hypothesis import strategies as st
from hypothesis.extra.pandas import column, data_frames, range_indexes, series


def nulldf_strategy():
    return data_frames(
        columns=[
            column("1", st.floats(allow_nan=True, allow_infinity=True)),
            column("2", st.sampled_from([np.nan])),
            column("3", st.sampled_from([np.nan])),
        ],
        index=range_indexes(min_size=3, max_size=20),
    )


def df_strategy():
    """
    A convenience function for generating a dataframe as a hypothesis strategy.

    Should be treated like a fixture, but should not be passed as a fixture
    into a test function. Instead::

        @given(df=dataframe())
        def test_function(df):
            # test goes here

    .. # noqa: DAR201
    """
    return data_frames(
        columns=[
            column("a", elements=st.integers()),
            column("Bell__Chart", elements=st.floats()),
            column("decorated-elephant", elements=st.integers()),
            column("animals@#$%^", elements=st.text()),
            column("cities", st.text()),
        ],
        index=range_indexes(min_size=1, max_size=20),
    )


def categoricaldf_strategy():
    return data_frames(
        columns=[
            column("names", st.sampled_from(names)),
            column("numbers", st.sampled_from(range(3))),
        ],
        index=range_indexes(min_size=1, max_size=20),
    )


names = [
    "John",
    "Mark",
    "Luke",
    "Matthew",
    "Peter",
    "Adam",
    "Eve",
    "Mary",
    "Ruth",
    "Esther",
]


def names_strategy():
    return st.lists(elements=st.sampled_from(names))


def conditional_df():
    """Dataframe used in tests_conditional_join."""
    return data_frames(
        [
            column(name="A", dtype=int),
            column(name="B", elements=st.floats(allow_nan=True)),
            column(name="C", elements=st.text(max_size=10)),
            column(name="D", dtype=bool),
            column(name="E", dtype="datetime64[ns]"),
        ],
        index=range_indexes(min_size=1, max_size=10),
    )


def conditional_series():
    """Series used in tests_conditional_join"""
    return series(dtype=int, index=range_indexes(min_size=1, max_size=10))


def conditional_right():
    """Dataframe used in tests_conditional_join."""
    return data_frames(
        [
            column(name="Integers", dtype=int),
            column(name="Numeric", elements=st.floats(allow_nan=True)),
            column(name="Floats", elements=st.floats(max_value=10)),
            column(name="Strings", dtype=str),
            column(name="Booleans", dtype=np.bool_),
            column(name="Dates", dtype="datetime64[ns]"),
            column(name="Dates_Right", dtype="datetime64[ns]"),
        ],
        index=range_indexes(min_size=1, max_size=10),
    )
