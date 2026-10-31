"""Tests for convert_currency() in the finance module."""

import json
from datetime import date, datetime

import pandas as pd
import pytest
import requests

from janitor import finance
from janitor.finance import convert_currency  # noqa: F401


class _FakeResponse:
    """Minimal ``requests`` response stub."""

    def __init__(self, payload, status_code=200):
        self.status_code = status_code
        self.text = json.dumps(payload)


def _clear_rate_cache():
    """Drop cached exchange rates when the helper is present."""
    fetch = getattr(finance, "fetch_exchange_rate", None)
    clear = getattr(fetch, "cache_clear", None)
    if clear is not None:
        clear()


def _rate_payload(params, rate=0.5):
    """Frankfurter v2 rates body, including the base identity row."""
    return [
        {
            "date": params.get("date", "2026-10-30"),
            "base": params["base"],
            "quote": params["base"],
            "rate": 1.0,
        },
        {
            "date": params.get("date", "2026-10-30"),
            "base": params["base"],
            "quote": params["quotes"],
            "rate": rate,
        },
    ]


@pytest.fixture
def fake_api(monkeypatch):
    """Replace the rates API with a stub that records each request."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        return _FakeResponse(_rate_payload(params))

    _clear_rate_cache()
    monkeypatch.setattr(finance.requests, "get", fake_get)
    yield calls
    _clear_rate_cache()


@pytest.mark.finance
def test_requests_frankfurter_latest_rate(dataframe, fake_api):
    """Test a new column uses the Frankfurter base and quotes parameters."""
    df = dataframe.convert_currency("a", "USD", "EUR", make_new_column=True)
    assert "a_EUR" in df.columns
    assert "column_name" not in df.columns
    pd.testing.assert_series_equal(df["a_EUR"], dataframe["a"] * 0.5, check_names=False)
    url, params, timeout = fake_api[0]
    assert url == finance.FRANKFURTER_API_URL
    assert params == {"base": "USD", "quotes": "EUR"}
    assert timeout is not None and timeout > 0


@pytest.mark.finance
def test_convert_in_place(dataframe, fake_api):
    """Test the column is overwritten and the input frame is untouched."""
    original = dataframe["a"].copy()
    df = dataframe.convert_currency("a", "USD", "EUR")
    pd.testing.assert_series_equal(df["a"], original * 0.5)
    pd.testing.assert_series_equal(dataframe["a"], original)
    assert len(fake_api) == 1


@pytest.mark.finance
def test_same_currency_skips_request(dataframe, fake_api):
    """Test converting to the same currency returns the original values."""
    df = dataframe.convert_currency("a", "USD", "USD", make_new_column=True)
    assert all(df["a"] == df["a_USD"])
    assert fake_api == []


@pytest.mark.finance
@pytest.mark.parametrize(
    "historical_date",
    [date(2018, 1, 2), datetime(2018, 1, 2, 15, 30)],
)
def test_historical_date_and_datetime(dataframe, fake_api, historical_date):
    """Test a historical date or datetime is sent as a date query parameter."""
    dataframe.convert_currency("a", "USD", "EUR", historical_date=historical_date)
    url, params, _timeout = fake_api[0]
    assert url == finance.FRANKFURTER_API_URL
    assert params == {
        "base": "USD",
        "quotes": "EUR",
        "date": "2018-01-02",
    }


@pytest.mark.finance
def test_base_url_override(dataframe, monkeypatch):
    """Test JANITOR_CURRENCY_API_URL replaces the Frankfurter endpoint."""
    override = "https://example.test/rates"
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append((url, params, timeout))
        return _FakeResponse(_rate_payload(params))

    monkeypatch.setenv("JANITOR_CURRENCY_API_URL", override)
    _clear_rate_cache()
    monkeypatch.setattr(finance.requests, "get", fake_get)
    dataframe.convert_currency("a", "USD", "EUR")
    assert calls[0][0] == override
    assert calls[0][1] == {"base": "USD", "quotes": "EUR"}
    _clear_rate_cache()


@pytest.mark.finance
def test_api_key_deprecated(dataframe, fake_api):
    """Test passing api_key warns and is otherwise ignored."""
    with pytest.warns(FutureWarning, match="api_key"):
        dataframe.convert_currency("a", "USD", "EUR", api_key="abc")
    assert "access_key" not in fake_api[0][1]
    assert "api_key" not in fake_api[0][1]


@pytest.mark.finance
def test_missing_rate(dataframe, monkeypatch):
    """Test a response without the requested rate raises ValueError."""
    _clear_rate_cache()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse([]),
    )
    with pytest.raises(ValueError, match="No USD/EUR rate"):
        dataframe.convert_currency("a", "USD", "EUR")
    _clear_rate_cache()


@pytest.mark.finance
def test_server_error(dataframe, monkeypatch):
    """Test a non-200 response raises ConnectionError."""
    _clear_rate_cache()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse(
            {"status": 503, "message": "unavailable"},
            status_code=503,
        ),
    )
    with pytest.raises(ConnectionError):
        dataframe.convert_currency("a", "USD", "EUR")
    _clear_rate_cache()


@pytest.mark.finance
def test_currency_check(dataframe, fake_api):
    """Test conversion raises exception for an invalid currency."""
    with pytest.raises(ValueError, match="INVALID-CURRENCY"):
        dataframe.convert_currency("a", "USD", "INVALID-CURRENCY")
    assert fake_api == []


@pytest.mark.finance
@pytest.mark.parametrize(
    "historical_date",
    [date(1982, 10, 27), datetime(1982, 10, 27, 1, 2, 3)],
)
def test_historical_before_ecb_history(dataframe, fake_api, historical_date):
    """Test dates before the ECB series raises ValueError without a request."""
    with pytest.raises(ValueError, match="1999-01-04"):
        dataframe.convert_currency(
            "a",
            "USD",
            "AUD",
            make_new_column=True,
            historical_date=historical_date,
        )
    assert fake_api == []


@pytest.mark.finance
@pytest.mark.xfail(
    reason="hits the live Frankfurter API, which may be unreachable",
    raises=(requests.exceptions.RequestException, ConnectionError),
)
def test_live_historical_rate():
    """Test a known ECB reference rate from the live Frankfurter API."""
    _clear_rate_cache()
    rate = finance.fetch_exchange_rate("USD", "EUR", date(2018, 1, 2))
    assert rate == pytest.approx(0.82884)
