import pytest
import requests
from datetime import date, datetime

from janitor.finance import convert_currency


@pytest.mark.finance
@pytest.mark.xfail
def test_make_currency_api_request():
    """
    Test for currency API request.

    This test exists because we rely solely on the service by
    exchangeratesapi. That said, we also mark it as expected to fail because
    it sometimes pings the exchange rates API a too frequently and causes
    tests to fail.

    For an example of how this test fails, see:
    https://github.com/ericmjl/pyjanitor/issues/147
    """
    r = requests.get("https://api.exchangeratesapi.io")
    assert r.status_code == 200


@pytest.mark.finance
def test_make_new_currency_col(dataframe):
    df = dataframe.convert_currency("a", "USD", "USD", make_new_column=True)
    assert all(df["a"] == df["a_USD"])


@pytest.mark.finance
def test_historical_datetime(dataframe):
    with pytest.raises(ValueError):
        assert dataframe.convert_currency(
            "a",
            "USD",
            "AUD",
            make_new_column=True,
            historical_date=datetime(1982, 10, 27),
        )


@pytest.mark.finance
def test_historical_date(dataframe):
    with pytest.raises(ValueError):
        assert dataframe.convert_currency(
            "a",
            "USD",
            "AUD",
            make_new_column=True,
            historical_date=date(1982, 10, 27),
        )


@pytest.mark.finance
def test_currency_check(dataframe):
    with pytest.raises(ValueError):
        assert dataframe.convert_currency("a", "USD", "ASDF")
