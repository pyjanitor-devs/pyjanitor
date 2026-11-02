"""Tests for convert_currency() in the finance module."""

import json
from datetime import date, datetime, timedelta, timezone

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
    """Drop cached historical exchange rates when the helper is present."""
    for name in ("fetch_exchange_rate", "cached_historical_exchange_rate"):
        fetch = getattr(finance, name, None)
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
def test_latest_rate_is_not_cached(dataframe, monkeypatch):
    """Test a second latest-rate call hits the service again."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(
            _rate_payload(params, rate=0.5 if len(calls) == 1 else 0.25)
        )

    _clear_rate_cache()
    monkeypatch.setattr(finance.requests, "get", fake_get)
    first = dataframe.convert_currency("a", "USD", "EUR")
    second = dataframe.convert_currency("a", "USD", "EUR")
    assert len(calls) == 2
    pd.testing.assert_series_equal(first["a"], dataframe["a"] * 0.5)
    pd.testing.assert_series_equal(second["a"], dataframe["a"] * 0.25)
    _clear_rate_cache()


@pytest.mark.finance
def test_historical_rate_is_cached(dataframe, monkeypatch):
    """Test a repeated historical date reuses the first response."""
    calls = []

    def fake_get(url, params=None, timeout=None):
        calls.append(params)
        return _FakeResponse(_rate_payload(params))

    _clear_rate_cache()
    monkeypatch.setattr(finance.requests, "get", fake_get)
    historical_date = date(2018, 1, 2)
    dataframe.convert_currency("a", "USD", "EUR", historical_date=historical_date)
    dataframe.convert_currency("a", "USD", "EUR", historical_date=historical_date)
    assert len(calls) == 1
    _clear_rate_cache()


@pytest.mark.finance
@pytest.mark.parametrize("code", ["HRK", "RUB"])
def test_retired_ecb_currencies_rejected(dataframe, fake_api, code):
    """Test HRK and RUB are no longer accepted currency codes."""
    assert code not in finance.currency_set
    with pytest.raises(ValueError, match=code):
        dataframe.convert_currency("a", "USD", code)
    assert fake_api == []


@pytest.mark.finance
def test_unprocessable_entity_is_value_error(dataframe, monkeypatch):
    """Test HTTP 422 raises ValueError with the service message."""
    _clear_rate_cache()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse(
            {"status": 422, "message": "invalid currency: ABC"},
            status_code=422,
        ),
    )
    with pytest.raises(ValueError, match="invalid currency: ABC"):
        dataframe.convert_currency("a", "USD", "EUR")
    _clear_rate_cache()


@pytest.mark.finance
@pytest.mark.parametrize(
    "historical_date",
    [
        datetime(2018, 1, 2, 23, 30, tzinfo=timezone(timedelta(hours=14))),
        pd.Timestamp("2018-01-02 23:30", tz="UTC"),
    ],
)
def test_timezone_aware_historical_date(dataframe, fake_api, historical_date):
    """Test an aware datetime uses its calendar date."""
    dataframe.convert_currency("a", "USD", "EUR", historical_date=historical_date)
    assert fake_api[0][1]["date"] == "2018-01-02"


@pytest.mark.finance
@pytest.mark.parametrize(
    "historical_date",
    [
        datetime(1998, 12, 31, 23, 0, tzinfo=timezone.utc),
        pd.Timestamp("1998-12-31 23:00", tz="UTC"),
    ],
)
def test_timezone_aware_date_before_ecb_history(dataframe, fake_api, historical_date):
    """Test an aware date before 1999-01-04 raises ValueError."""
    with pytest.raises(ValueError, match="1999-01-04"):
        dataframe.convert_currency(
            "a",
            "USD",
            "EUR",
            historical_date=historical_date,
        )
    assert fake_api == []


@pytest.mark.finance
def test_legacy_positional_api_key_still_converts(dataframe, fake_api):
    """Test an api_key-first positional call still converts the column."""
    with pytest.warns(FutureWarning, match="positional"):
        df = dataframe.convert_currency("KEY", "a", "USD", "EUR", make_new_column=True)
    assert "a_EUR" in df.columns
    assert fake_api[0][1] == {"base": "USD", "quotes": "EUR"}


@pytest.mark.finance
def test_legacy_positional_api_key_with_date(dataframe, fake_api):
    """Test an api_key-first call keeps a following historical date."""
    with pytest.warns(FutureWarning, match="positional"):
        dataframe.convert_currency("KEY", "a", "USD", "EUR", date(2018, 1, 2))
    assert fake_api[0][1] == {
        "base": "USD",
        "quotes": "EUR",
        "date": "2018-01-02",
    }


@pytest.mark.finance
def test_legacy_positional_unknown_column_is_clear(dataframe, fake_api):
    """Test an old positional call with a missing column explains the shift."""
    with pytest.raises(TypeError, match="api_key"):
        dataframe.convert_currency("KEY", "missing", "USD", "EUR")
    assert fake_api == []


@pytest.mark.finance
def test_response_must_be_a_list(dataframe, monkeypatch):
    """Test a JSON object body is not reported as a missing rate."""
    _clear_rate_cache()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse(
            {"base": "USD", "rates": {"EUR": 0.5}}
        ),
    )
    with pytest.raises(ValueError, match="list"):
        dataframe.convert_currency("a", "USD", "EUR")
    _clear_rate_cache()


@pytest.mark.finance
@pytest.mark.parametrize("rate", [0, -1, float("nan"), "nope", None, True])
def test_non_positive_or_non_numeric_rate_rejected(dataframe, monkeypatch, rate):
    """Test a zero, non-finite, or non-numeric rate raises ValueError."""
    _clear_rate_cache()
    payload = [
        {
            "date": "2018-01-02",
            "base": "USD",
            "quote": "EUR",
            "rate": rate,
        }
    ]
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse(payload),
    )
    with pytest.raises(ValueError, match="positive finite number"):
        dataframe.convert_currency("a", "USD", "EUR")
    _clear_rate_cache()


@pytest.mark.finance
def test_missing_quote_among_other_rows(dataframe, monkeypatch):
    """Test rows for a different quote do not satisfy the request."""
    _clear_rate_cache()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse(
            [
                {
                    "date": "2018-01-02",
                    "base": "USD",
                    "quote": "GBP",
                    "rate": 0.75,
                }
            ]
        ),
    )
    with pytest.raises(ValueError, match="No USD/EUR rate"):
        dataframe.convert_currency("a", "USD", "EUR")
    _clear_rate_cache()


@pytest.mark.finance
def test_non_object_rate_row_rejected(dataframe, monkeypatch):
    """Test a list row that is not an object raises ValueError."""
    _clear_rate_cache()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _FakeResponse(
            ["bad", {"date": "2018-01-02", "base": "USD", "quote": "EUR", "rate": 0.5}]
        ),
    )
    with pytest.raises(ValueError, match="not an object"):
        dataframe.convert_currency("a", "USD", "EUR")
    _clear_rate_cache()


@pytest.mark.finance
def test_invalid_json_is_connection_error(dataframe, monkeypatch):
    """Test a 200 response that is not JSON raises ConnectionError."""

    class _BadBody:
        status_code = 200
        text = "not-json"

    _clear_rate_cache()
    monkeypatch.setattr(
        finance.requests,
        "get",
        lambda url, params=None, timeout=None: _BadBody(),
    )
    with pytest.raises(ConnectionError):
        dataframe.convert_currency("a", "USD", "EUR")
    _clear_rate_cache()


@pytest.mark.finance
@pytest.mark.parametrize(
    "override",
    ["file:///tmp/rates", "ftp://example.test/rates", "example.test/rates"],
)
def test_currency_api_url_requires_http(dataframe, monkeypatch, override):
    """Test JANITOR_CURRENCY_API_URL rejects a non-http(s) URL."""
    monkeypatch.setenv("JANITOR_CURRENCY_API_URL", override)
    with pytest.raises(ValueError, match="http"):
        dataframe.convert_currency("a", "USD", "EUR")


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
