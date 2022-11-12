"""Tests for inflate_currency() in finance module."""
import pytest
import requests

from janitor.finance import _inflate_currency, inflate_currency  # noqa: F401


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_make_currency_inflator_api_request():
    """Test for currency inflator API request.

    This test exists because we rely solely on the service by
    the World Bank's Indicator API:
    https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
    """
    r = requests.get(
        "https://api.worldbank.org/v2/country/USA/indicator/"
        "FP.CPI.TOTL.ZG?date=2010:2018&format=json"
    )
    assert r.status_code == 200


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_make_new_inflated_currency_col(dataframe):
    """Test currency inflation for same year added as a new column."""
    df = dataframe.inflate_currency(
        "a",
        country="USA",
        currency_year=2018,
        to_year=2018,
        make_new_column=True,
    )
    assert all(df["a"] == df["a_2018"])


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_inflate_existing_currency_col(dataframe):
    """Test currency inflation updates existing column."""
    initialval = dataframe["a"].sum()
    # Pulled raw values from API website for USA 2018 and 2015
    inflator = _inflate_currency("USA", currency_year=2018, to_year=2015)
    df = dataframe.inflate_currency(
        "a",
        country="USA",
        currency_year=2018,
        to_year=2015,
        make_new_column=False,
    )
    updatedval = df["a"].sum()
    assert (initialval * inflator) == pytest.approx(updatedval)


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_expected_result(dataframe):
    """Test inflation calculation gives expected value."""
    initialval = dataframe["a"].sum()
    # Pulled raw values from API website for USA 2018 and 2015
    inflator = _inflate_currency("USA", currency_year=2018, to_year=2015)
    df = dataframe.inflate_currency(
        "a",
        country="USA",
        currency_year=2018,
        to_year=2015,
        make_new_column=True,
    )
    updatedval = df["a_2015"].sum()
    assert (initialval * inflator) == pytest.approx(updatedval)


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_expected_result_with_full_country_name(dataframe):
    """Test inflation calculation works when providing country name."""
    initialval = dataframe["a"].sum()
    # Pulled raw values from API website for USA 2018 and 2015
    inflator = _inflate_currency(
        "United States", currency_year=2018, to_year=2015
    )
    df = dataframe.inflate_currency(
        "a",
        country="United States",
        currency_year=2018,
        to_year=2015,
        make_new_column=True,
    )
    updatedval = df["a_2015"].sum()
    assert (initialval * inflator) == pytest.approx(updatedval)


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_wb_country_check(dataframe):
    """Test inflation calculation fails when providing invalid country name."""
    with pytest.raises(ValueError):
        assert dataframe.inflate_currency(
            "a", country="INVALID-COUNTRY", currency_year=2018, to_year=2018
        )


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_year_check(dataframe):
    """Test inflation calculation fails with year outside valid range."""
    with pytest.raises(ValueError):
        assert dataframe.inflate_currency(
            "a", country="USA", currency_year=1950, to_year=2018
        )


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_datatypes_check(dataframe):
    """Test inflation calculation fails when provided invalid types."""
    with pytest.raises(TypeError):
        assert dataframe.inflate_currency(
            "a", country=123, currency_year=1960, to_year=2018
        )
        assert dataframe.inflate_currency(
            "a", country="USA", currency_year="b", to_year=2018
        )
        assert dataframe.inflate_currency(
            "a", country="USA", currency_year=1960, to_year="b"
        )


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_api_result_check(dataframe):
    """Test inflation calculation fails with year outside API's valid range."""
    with pytest.raises(ValueError):
        assert dataframe.inflate_currency(
            "a", country="USA", currency_year=2030, to_year=2050
        )


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_to_year_available(dataframe):
    """Test inflation calculation fails with unavailable to_year."""
    with pytest.raises(ValueError):
        assert dataframe.inflate_currency(
            "a", country="GHA", currency_year=2010, to_year=1962
        )


@pytest.mark.xfail(reason="Relies on external API call.")
@pytest.mark.finance
def test_currency_year_available(dataframe):
    """Test inflation calculation fails with unavailable currency_year."""
    with pytest.raises(ValueError):
        assert dataframe.inflate_currency(
            "a", country="GHA", currency_year=1962, to_year=2010
        )
