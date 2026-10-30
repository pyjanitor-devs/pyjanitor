"""Tests for convert_currency() in finance module."""

import json
from datetime import date, datetime

import pandas as pd
import pytest
import requests

from janitor import finance
from janitor.finance import convert_currency  # noqa: F401


class _FakeResponse:
    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.text = json.dumps(payload)


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the rates API with a stub that records each request."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params))
        return _FakeResponse(
            {
                "base": params["base"],
                "date": "2018-01-02",
                "rates": {params["symbols"]: 0.5},
            }
        )

    finance._convert_currency.cache_clear()
    monkeypatch.setattr(finance.requests, "get", fake_get)
    yield calls
    finance._convert_currency.cache_clear()


@pytest.mark.finance
def test_make_new_currency_col(dataframe, fake_api):
    """Test a new column is added with converted values."""
    df = dataframe.convert_currency(
        "a", "USD", "EUR", make_new_column=True
    )
    assert "a_EUR" in df.columns
    pd.testing.assert_series_equal(
        df["a_EUR"], dataframe["a"] * 0.5, check_names=False
    )
    url, params = fake_api[0]
    assert url == finance.CURRENCY_API_URL + "/latest"
    assert params == {"base": "USD", "symbols": "EUR"}


@pytest.mark.finance
def test_convert_in_place(dataframe, fake_api):
    """Test the column is overwritten and the input frame is untouched."""
    original = dataframe["a"].copy()
    df = dataframe.convert_currency("a", "USD", "EUR")
    pd.testing.assert_series_equal(df["a"], original * 0.5)
    pd.testing.assert_series_equal(dataframe["a"], original)


@pytest.mark.finance
def test_same_currency_skips_request(dataframe, fake_api):
    """Test converting to the same currency returns the original values."""
    df = dataframe.convert_currency("a", "USD", "USD", make_new_column=True)
    assert all(df["a"] == df["a_USD"])
    assert fake_api == []


@pytest.mark.finance
@pytest.mark.parametrize(
    "historical_date", [date(2018, 1, 2), datetime(2018, 1, 2, 15, 30)]
)
def test_historical_url(dataframe, fake_api, historical_date):
    """Test a historical date is sent as a YYYY-MM-DD path segment."""
    dataframe.convert_currency(
        "a", "USD", "EUR", historical_date=historical_date
    )
    assert fake_api[0][0] == finance.CURRENCY_API_URL + "/2018-01-02"


@pytest.mark.finance
def test_api_key_deprecated(dataframe, fake_api):
    """Test passing `api_key` warns and is otherwise ignored."""
    with pytest.warns(FutureWarning):
        dataframe.convert_currency("a", "USD", "EUR", api_key="abc")
    assert "access_key" not in fake_api[0][1]


@pytest.mark.finance
def test_missing_rate(dataframe, monkeypatch):
    """Test a currency without a published rate raises ValueError."""
    finance._convert_currency.cache_clear()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse(
            {"base": "USD", "date": "2026-10-30", "rates": {}}
        ),
    )
    with pytest.raises(ValueError):
        dataframe.convert_currency("a", "USD", "BGN")


@pytest.mark.finance
def test_server_error(dataframe, monkeypatch):
    """Test a non-200 response raises ConnectionError."""
    finance._convert_currency.cache_clear()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse({}, 503),
    )
    with pytest.raises(ConnectionError):
        dataframe.convert_currency("a", "USD", "EUR")


@pytest.mark.finance
def test_historical_datetime(dataframe):
    """Test conversion raises exception for datetime outside API range."""
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
    """Test conversion raises exception for date outside API range."""
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
    """Test conversion raises exception for invalid currency."""
    with pytest.raises(ValueError):
        assert dataframe.convert_currency("a", "USD", "INVALID-CURRENCY")


@pytest.mark.finance
@pytest.mark.xfail(
    reason="hits the live rates API, which may be rate limited",
    raises=(requests.exceptions.RequestException, ConnectionError),
)
def test_live_historical_rate():
    """Test a known historical rate from the live API."""
    finance._convert_currency.cache_clear()
    rate = finance._convert_currency("USD", "EUR", date(2018, 1, 2))
    assert rate == pytest.approx(0.82884376)
