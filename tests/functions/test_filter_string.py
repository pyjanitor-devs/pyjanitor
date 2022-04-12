import pytest


@pytest.mark.functions
def test_filter_string(dataframe):
    df = dataframe.filter_string(
        column_name="animals@#$%^",
        search_string="bbit",
    )

    assert len(df) == 3


def test_filter_string_complement(dataframe):
    df = dataframe.filter_string(
        column_name="cities",
        search_string="hang",
        complement=True,
    )

    assert len(df) == 6


def test_filter_string_case(dataframe):
    df = dataframe.filter_string(
        column_name="cities",
        search_string="B",
        case=False,
    )

    assert len(df) == 6


def test_filter_string_regex(dataframe):
    df = dataframe.change_type("Bell__Chart", str).filter_string(
        column_name="Bell__Chart",
        search_string="1.",
        regex=False,
    )

    assert len(df) == 3
