"""Finance-specific data cleaning functions."""

import json
import math
import os
from datetime import date, datetime
from functools import lru_cache
from typing import Optional
from urllib.parse import urlparse
from warnings import warn

import pandas as pd
import pandas_flavor as pf
import requests

from .utils import check, deprecated_alias, find_stack_level, is_connected

__all__ = [
    "FRANKFURTER_API_URL",
    "convert_currency",
    "convert_stock",
    "currency_set",
    "get_symbol",
    "inflate_currency",
]

currency_set = {
    "AUD",
    "BGN",
    "BRL",
    "CAD",
    "CHF",
    "CNY",
    "CZK",
    "DKK",
    "EUR",
    "GBP",
    "HKD",
    "HRK",
    "HUF",
    "IDR",
    "ILS",
    "INR",
    "ISK",
    "JPY",
    "KRW",
    "MXN",
    "MYR",
    "NOK",
    "NZD",
    "PHP",
    "PLN",
    "RON",
    "RUB",
    "SEK",
    "SGD",
    "THB",
    "TRY",
    "USD",
    "ZAR",
}

# Dictionary of recognized World Bank countries and their abbreviations
wb_country_dict = {
    "Aruba": "ABW",
    "Afghanistan": "AFG",
    "Angola": "AGO",
    "Albania": "ALB",
    "Andorra": "AND",
    "Arab World": "ARB",
    "United Arab Emirates": "ARE",
    "Argentina": "ARG",
    "Armenia": "ARM",
    "American Samoa": "ASM",
    "Antigua and Barbuda": "ATG",
    "Australia": "AUS",
    "Austria": "AUT",
    "Azerbaijan": "AZE",
    "Burundi": "BDI",
    "Belgium": "BEL",
    "Benin": "BEN",
    "Burkina Faso": "BFA",
    "Bangladesh": "BGD",
    "Bulgaria": "BGR",
    "Bahrain": "BHR",
    "Bahamas, The": "BHS",
    "Bosnia and Herzegovina": "BIH",
    "Belarus": "BLR",
    "Belize": "BLZ",
    "Bermuda": "BMU",
    "Bolivia": "BOL",
    "Brazil": "BRA",
    "Barbados": "BRB",
    "Brunei Darussalam": "BRN",
    "Bhutan": "BTN",
    "Botswana": "BWA",
    "Central African Republic": "CAF",
    "Canada": "CAN",
    "Central Europe and the Baltics": "CEB",
    "Switzerland": "CHE",
    "Channel Islands": "CHI",
    "Chile": "CHL",
    "China": "CHN",
    "Cote d'Ivoire": "CIV",
    "Cameroon": "CMR",
    "Congo, Dem. Rep.": "COD",
    "Congo, Rep.": "COG",
    "Colombia": "COL",
    "Comoros": "COM",
    "Cabo Verde": "CPV",
    "Costa Rica": "CRI",
    "Caribbean small states": "CSS",
    "Cuba": "CUB",
    "Curacao": "CUW",
    "Cayman Islands": "CYM",
    "Cyprus": "CYP",
    "Czech Republic": "CZE",
    "Germany": "DEU",
    "Djibouti": "DJI",
    "Dominica": "DMA",
    "Denmark": "DNK",
    "Dominican Republic": "DOM",
    "Algeria": "DZA",
    "East Asia & Pacific (excluding high income)": "EAP",
    "Early-demographic dividend": "EAR",
    "East Asia & Pacific": "EAS",
    "Europe & Central Asia (excluding high income)": "ECA",
    "Europe & Central Asia": "ECS",
    "Ecuador": "ECU",
    "Egypt, Arab Rep.": "EGY",
    "Euro area": "EMU",
    "Eritrea": "ERI",
    "Spain": "ESP",
    "Estonia": "EST",
    "Ethiopia": "ETH",
    "European Union": "EUU",
    "Fragile and conflict affected situations": "FCS",
    "Finland": "FIN",
    "Fiji": "FJI",
    "France": "FRA",
    "Faroe Islands": "FRO",
    "Micronesia, Fed. Sts.": "FSM",
    "Gabon": "GAB",
    "United Kingdom": "GBR",
    "Georgia": "GEO",
    "Ghana": "GHA",
    "Gibraltar": "GIB",
    "Guinea": "GIN",
    "Gambia, The": "GMB",
    "Guinea-Bissau": "GNB",
    "Equatorial Guinea": "GNQ",
    "Greece": "GRC",
    "Grenada": "GRD",
    "Greenland": "GRL",
    "Guatemala": "GTM",
    "Guam": "GUM",
    "Guyana": "GUY",
    "High income": "HIC",
    "Hong Kong SAR, China": "HKG",
    "Honduras": "HND",
    "Heavily indebted poor countries (HIPC)": "HPC",
    "Croatia": "HRV",
    "Haiti": "HTI",
    "Hungary": "HUN",
    "IBRD only": "IBD",
    "IDA & IBRD total": "IBT",
    "IDA total": "IDA",
    "IDA blend": "IDB",
    "Indonesia": "IDN",
    "IDA only": "IDX",
    "Isle of Man": "IMN",
    "India": "IND",
    "Not classified": "INX",
    "Ireland": "IRL",
    "Iran, Islamic Rep.": "IRN",
    "Iraq": "IRQ",
    "Iceland": "ISL",
    "Israel": "ISR",
    "Italy": "ITA",
    "Jamaica": "JAM",
    "Jordan": "JOR",
    "Japan": "JPN",
    "Kazakhstan": "KAZ",
    "Kenya": "KEN",
    "Kyrgyz Republic": "KGZ",
    "Cambodia": "KHM",
    "Kiribati": "KIR",
    "St. Kitts and Nevis": "KNA",
    "Korea, Rep.": "KOR",
    "Kuwait": "KWT",
    "Latin America & Caribbean (excluding high income)": "LAC",
    "Lao PDR": "LAO",
    "Lebanon": "LBN",
    "Liberia": "LBR",
    "Libya": "LBY",
    "St. Lucia": "LCA",
    "Latin America & Caribbean": "LCN",
    "Least developed countries: UN classification": "LDC",
    "Low income": "LIC",
    "Liechtenstein": "LIE",
    "Sri Lanka": "LKA",
    "Lower middle income": "LMC",
    "Low & middle income": "LMY",
    "Lesotho": "LSO",
    "Late-demographic dividend": "LTE",
    "Lithuania": "LTU",
    "Luxembourg": "LUX",
    "Latvia": "LVA",
    "Macao SAR, China": "MAC",
    "St. Martin (French part)": "MAF",
    "Morocco": "MAR",
    "Monaco": "MCO",
    "Moldova": "MDA",
    "Madagascar": "MDG",
    "Maldives": "MDV",
    "Middle East & North Africa": "MEA",
    "Mexico": "MEX",
    "Marshall Islands": "MHL",
    "Middle income": "MIC",
    "North Macedonia": "MKD",
    "Mali": "MLI",
    "Malta": "MLT",
    "Myanmar": "MMR",
    "Middle East & North Africa (excluding high income)": "MNA",
    "Montenegro": "MNE",
    "Mongolia": "MNG",
    "Northern Mariana Islands": "MNP",
    "Mozambique": "MOZ",
    "Mauritania": "MRT",
    "Mauritius": "MUS",
    "Malawi": "MWI",
    "Malaysia": "MYS",
    "North America": "NAC",
    "Namibia": "NAM",
    "New Caledonia": "NCL",
    "Niger": "NER",
    "Nigeria": "NGA",
    "Nicaragua": "NIC",
    "Netherlands": "NLD",
    "Norway": "NOR",
    "Nepal": "NPL",
    "Nauru": "NRU",
    "New Zealand": "NZL",
    "OECD members": "OED",
    "Oman": "OMN",
    "Other small states": "OSS",
    "Pakistan": "PAK",
    "Panama": "PAN",
    "Peru": "PER",
    "Philippines": "PHL",
    "Palau": "PLW",
    "Papua New Guinea": "PNG",
    "Poland": "POL",
    "Pre-demographic dividend": "PRE",
    "Puerto Rico": "PRI",
    "Korea, Dem. People's Rep.": "PRK",
    "Portugal": "PRT",
    "Paraguay": "PRY",
    "West Bank and Gaza": "PSE",
    "Pacific island small states": "PSS",
    "Post-demographic dividend": "PST",
    "French Polynesia": "PYF",
    "Qatar": "QAT",
    "Romania": "ROU",
    "Russian Federation": "RUS",
    "Rwanda": "RWA",
    "South Asia": "SAS",
    "Saudi Arabia": "SAU",
    "Sudan": "SDN",
    "Senegal": "SEN",
    "Singapore": "SGP",
    "Solomon Islands": "SLB",
    "Sierra Leone": "SLE",
    "El Salvador": "SLV",
    "San Marino": "SMR",
    "Somalia": "SOM",
    "Serbia": "SRB",
    "Sub-Saharan Africa (excluding high income)": "SSA",
    "South Sudan": "SSD",
    "Sub-Saharan Africa": "SSF",
    "Small states": "SST",
    "Sao Tome and Principe": "STP",
    "Suriname": "SUR",
    "Slovak Republic": "SVK",
    "Slovenia": "SVN",
    "Sweden": "SWE",
    "Eswatini": "SWZ",
    "Sint Maarten (Dutch part)": "SXM",
    "Seychelles": "SYC",
    "Syrian Arab Republic": "SYR",
    "Turks and Caicos Islands": "TCA",
    "Chad": "TCD",
    "East Asia & Pacific (IDA & IBRD countries)": "TEA",
    "Europe & Central Asia (IDA & IBRD countries)": "TEC",
    "Togo": "TGO",
    "Thailand": "THA",
    "Tajikistan": "TJK",
    "Turkmenistan": "TKM",
    "Latin America & the Caribbean (IDA & IBRD countries)": "TLA",
    "Timor-Leste": "TLS",
    "Middle East & North Africa (IDA & IBRD countries)": "TMN",
    "Tonga": "TON",
    "South Asia (IDA & IBRD)": "TSA",
    "Sub-Saharan Africa (IDA & IBRD countries)": "TSS",
    "Trinidad and Tobago": "TTO",
    "Tunisia": "TUN",
    "Turkey": "TUR",
    "Tuvalu": "TUV",
    "Tanzania": "TZA",
    "Uganda": "UGA",
    "Ukraine": "UKR",
    "Upper middle income": "UMC",
    "Uruguay": "URY",
    "United States": "USA",
    "Uzbekistan": "UZB",
    "St. Vincent and the Grenadines": "VCT",
    "Venezuela, RB": "VEN",
    "British Virgin Islands": "VGB",
    "Virgin Islands (U.S.)": "VIR",
    "Vietnam": "VNM",
    "Vanuatu": "VUT",
    "World": "WLD",
    "Samoa": "WSM",
    "Kosovo": "XKX",
    "Yemen, Rep.": "YEM",
    "South Africa": "ZAF",
    "Zambia": "ZMB",
    "Zimbabwe": "ZWE",
}


def _check_currency(currency: str):
    """Check that currency is in supported set."""
    if currency not in currency_set:
        raise ValueError(
            f"currency {currency} not in supported currency set, {currency_set}"
        )


def _check_wb_country(country: str):
    """Check that world bank country is in supported set."""
    if (country not in wb_country_dict.keys()) & (
        country not in wb_country_dict.values()  # noqa: PD011
    ):
        raise ValueError(
            f"country {country} not in supported World Bank country dict, "
            f"{wb_country_dict}"
        )


def _check_wb_years(year: int):
    """Check that year is in world bank dataset years."""
    if year < 1960:
        raise ValueError("year value must be 1960 or later")


# Frankfurter v2 (https://api.frankfurter.dev). Currency filters are
# `base` and `quotes`; `from` and `to` bound a date range. Official ECB
# reference rates come from the provider route below.
# JANITOR_CURRENCY_API_URL replaces the whole URL.
FRANKFURTER_API_URL = "https://api.frankfurter.dev/v2/providers/ecb/rates"

# First date with no ECB observation. Later requests still repeat the last
# published rate, so availability is decided here.
retired_currencies = {
    "BGN": (date(2026, 1, 1), "EUR"),
    "HRK": (date(2023, 1, 1), "EUR"),
    "RUB": (date(2022, 3, 2), None),
}


def currency_api_url() -> str:
    """Return the exchange-rate endpoint used by ``convert_currency``.

    The default is Frankfurter's ECB provider route. Set the
    ``JANITOR_CURRENCY_API_URL`` environment variable to use another
    endpoint that accepts the same ``base``, ``quotes``, and ``date``
    query parameters and returns a JSON list of objects with ``quote``
    and ``rate`` fields. The value must be an absolute ``http`` or
    ``https`` URL. Currency codes and the requested date are sent to
    that host.

    Returns:
        The rates endpoint URL.

    Raises:
        ValueError: If ``JANITOR_CURRENCY_API_URL`` is set and is not
            an absolute http or https URL.
    """
    configured = os.environ.get("JANITOR_CURRENCY_API_URL")
    if not configured or not configured.strip():
        return FRANKFURTER_API_URL
    candidate = configured.strip()
    parsed = urlparse(candidate)
    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        raise ValueError(
            "JANITOR_CURRENCY_API_URL must be an absolute http or https URL. "
            f"Got {configured!r}."
        )
    return candidate


def positive_exchange_rate(rate: object) -> float:
    """Return ``rate`` when it is a positive finite number.

    Args:
        rate: Candidate rate from one response row.

    Returns:
        The rate as a float.

    Raises:
        ValueError: If ``rate`` is missing, non-numeric, non-finite,
            or not positive.
    """
    if isinstance(rate, bool) or not isinstance(rate, (int, float, str)):
        raise ValueError(f"Exchange rate {rate!r} is not a positive finite number.")
    try:
        value = float(rate)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Exchange rate {rate!r} is not a positive finite number."
        ) from exc
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"Exchange rate {rate!r} is not a positive finite number.")
    return value


def request_exchange_rate(
    from_currency: str,
    to_currency: str,
    quote_date: Optional[str],
    api_url: str,
) -> float:
    """Request one uncached rate from a Frankfurter-compatible endpoint.

    Args:
        from_currency: Currency code to convert from.
        to_currency: Currency code to convert to.
        quote_date: ISO date ``YYYY-MM-DD``, or None for the latest rate.
        api_url: Rates endpoint.

    Returns:
        Units of ``to_currency`` per one unit of ``from_currency``.

    Raises:
        ValueError: If the service returns HTTP 422, the body is not a
            list of rate objects, or the requested rate is missing or
            not a positive finite number.
        ConnectionError: If the request times out, cannot connect, the
            server does not return HTTP 200 or 422, or the body is not JSON.
    """
    params = {"base": from_currency, "quotes": to_currency}
    if quote_date is not None:
        params["date"] = quote_date

    try:
        result = requests.get(api_url, params=params, timeout=30)
    except requests.Timeout as exc:
        raise ConnectionError(
            "Exchange rate request timed out after 30 seconds."
        ) from exc
    except requests.ConnectionError as exc:
        raise ConnectionError(
            "Exchange rate request could not connect to the server."
        ) from exc
    if result.status_code == 422:
        detail = "the rates service rejected the request"
        try:
            body = json.loads(result.text)
        except json.JSONDecodeError:
            body = None
        if isinstance(body, dict) and body.get("message"):
            detail = str(body["message"])
        raise ValueError(f"Exchange rate request was rejected: {detail}.")
    if result.status_code != 200:
        raise ConnectionError(
            "Exchange Rate API failed to receive a 200 "
            "response from the server. "
            "Please try again later."
        )

    try:
        payload = json.loads(result.text)
    except json.JSONDecodeError as exc:
        raise ConnectionError(
            "Exchange Rate API returned a response that is not valid JSON."
        ) from exc
    if not isinstance(payload, list):
        raise ValueError("Exchange rate response must be a list of rate objects.")

    published = quote_date or "the latest date"
    for row in payload:
        if not isinstance(row, dict):
            raise ValueError(
                "Exchange rate response contained a row that is not an object."
            )
        if row.get("date"):
            published = str(row["date"])
        if row.get("quote") != to_currency:
            continue
        return positive_exchange_rate(row.get("rate"))

    raise ValueError(
        f"No {from_currency}/{to_currency} rate was published for {published}."
    )


@lru_cache(maxsize=32)
def cached_historical_exchange_rate(
    from_currency: str,
    to_currency: str,
    quote_date: str,
    api_url: str,
) -> float:
    """Return a cached ECB rate for one calendar date.

    Latest rates are not stored. ``quote_date`` is ``YYYY-MM-DD``.

    Args:
        from_currency: Currency code to convert from.
        to_currency: Currency code to convert to.
        quote_date: ISO date ``YYYY-MM-DD``.
        api_url: Rates endpoint.

    Returns:
        Units of ``to_currency`` per one unit of ``from_currency``.
    """
    return request_exchange_rate(from_currency, to_currency, quote_date, api_url)


def is_currency_code(value: object) -> bool:
    """Return whether ``value`` is a three-letter currency code.

    Args:
        value: Candidate currency code.

    Returns:
        True when ``value`` is three letters.
    """
    return isinstance(value, str) and len(value) == 3 and value.isalpha()


def currency_code(currency: object) -> str:
    """Return a three-letter currency code.

    Args:
        currency: Candidate ISO code.

    Returns:
        The upper-case code.

    Raises:
        ValueError: If ``currency`` is not three letters.
    """
    if not is_currency_code(currency):
        raise ValueError(
            f"currency {currency} is not a three-letter ISO currency code."
        )
    return str(currency).upper()


def require_currency_available(currency: str, on_date: Optional[date]) -> None:
    """Reject a currency after the ECB stopped publishing it.

    Args:
        currency: Three-letter currency code.
        on_date: Calendar date, or None for the latest rate.

    Raises:
        ValueError: If ``currency`` was replaced or withdrawn on or
            before ``on_date``.
    """
    retirement = retired_currencies.get(currency)
    if retirement is None:
        return
    unavailable_from, replaced_by = retirement
    as_of = date.today() if on_date is None else on_date
    if as_of < unavailable_from:
        return
    when = "the latest rate" if on_date is None else as_of.isoformat()
    if replaced_by:
        raise ValueError(
            f"{currency} was replaced by {replaced_by} on "
            f"{unavailable_from.isoformat()}, so no {currency} rate is "
            f"available for {when}."
        )
    raise ValueError(
        f"{currency} is not available for {when}. The ECB stopped "
        f"publishing it on {unavailable_from.isoformat()}."
    )


def normalize_convert_currency_arguments(
    df: pd.DataFrame,
    column_name: Optional[str],
    from_currency: Optional[str],
    to_currency: Optional[str],
    historical_date: Optional[date],
    make_new_column: bool,
    api_key: Optional[str],
) -> tuple:
    """Map an api_key-first positional call onto the current arguments.

    Args:
        df: DataFrame whose columns identify the old column argument.
        column_name: Current column argument, or the old api key.
        from_currency: Current source currency, or the old column name.
        to_currency: Current target currency, or the old source currency.
        historical_date: Current rate date, or the old target currency.
        make_new_column: Current flag, or the old rate date.
        api_key: Current key, or the old new-column flag.

    Returns:
        ``column_name``, ``from_currency``, ``to_currency``,
        ``historical_date``, ``make_new_column``, and whether the old
        positional order was detected.

    Raises:
        TypeError: If a currency code lands in the date position and
            the call is not the old positional order.
    """
    legacy_call = (
        isinstance(from_currency, str)
        and from_currency in df.columns
        and is_currency_code(to_currency)
        and is_currency_code(historical_date)
    )
    if legacy_call:
        if isinstance(make_new_column, date):
            shifted_date = make_new_column
            shifted_make_new = api_key if isinstance(api_key, bool) else False
        else:
            shifted_date = None
            shifted_make_new = make_new_column
        return (
            from_currency,
            to_currency,
            historical_date,
            shifted_date,
            shifted_make_new,
            True,
        )
    if is_currency_code(historical_date):
        raise TypeError(
            "convert_currency received a currency code where a date was "
            "expected. api_key is no longer the first positional "
            "argument. Pass the column name first: "
            "df.convert_currency(column_name, from_currency, to_currency). "
            "api_key is ignored."
        )
    return (
        column_name,
        from_currency,
        to_currency,
        historical_date,
        make_new_column,
        False,
    )


def fetch_exchange_rate(
    from_currency: Optional[str] = None,
    to_currency: Optional[str] = None,
    historical_date: Optional[date] = None,
    api_url: str = FRANKFURTER_API_URL,
) -> float:
    """Fetch one ECB reference rate from Frankfurter.

    [Frankfurter](https://www.frankfurter.app) republishes European
    Central Bank reference rates without an API key. The series starts
    on 1999-01-04. A date with no published observation, such as a
    weekend or a TARGET holiday, uses the previous published rate.

    A date before today is cached. The latest rate and today's date
    are requested every time.

    Args:
        from_currency: Currency code to convert from.
        to_currency: Currency code to convert to.
        historical_date: Rate date. ``datetime`` values, including
            timezone-aware values and ``pandas.Timestamp``, use their
            calendar date and ignore the time. Omit it for the latest
            rate. ``NaT`` is rejected.
        api_url: Rates endpoint. Defaults to ``FRANKFURTER_API_URL``.
            ``convert_currency`` passes ``currency_api_url``, which
            honors ``JANITOR_CURRENCY_API_URL``.

    Returns:
        Units of ``to_currency`` per one unit of ``from_currency``.

    Raises:
        TypeError: If ``historical_date`` is not a ``date`` or
            ``datetime``.
        ValueError: If a currency code is not three letters, the date
            is before 1999-01-04 or ``NaT``, the currency is no longer
            published on that date, or the response has no usable rate.
        ConnectionError: If the request times out, cannot connect, the
            server does not return HTTP 200 or 422, or the body is not
            JSON.
    """
    quote_day = None
    if historical_date is not None:
        if pd.isna(historical_date):
            raise ValueError(
                "historical_date is NaT and cannot be used as a rate date."
            )
        check("historical_date", historical_date, [datetime, date])
        # Compare calendar dates. An aware datetime is not ordered
        # against a naive datetime, and Timestamp is a datetime.
        quote_day = (
            historical_date.date()
            if isinstance(historical_date, datetime)
            else historical_date
        )
        if quote_day < date(1999, 1, 4):
            raise ValueError("historical_date must be on or after 1999-01-04.")

    from_currency = currency_code(from_currency)
    to_currency = currency_code(to_currency)
    if from_currency == to_currency:
        return 1.0
    require_currency_available(from_currency, quote_day)
    require_currency_available(to_currency, quote_day)
    quote_date = None if quote_day is None else quote_day.isoformat()
    if quote_date is None or quote_day >= date.today():
        return request_exchange_rate(from_currency, to_currency, quote_date, api_url)
    return cached_historical_exchange_rate(
        from_currency, to_currency, quote_date, api_url
    )


@pf.register_dataframe_method
@deprecated_alias(colname="column_name")
def convert_currency(
    df: pd.DataFrame,
    column_name: Optional[str] = None,
    from_currency: Optional[str] = None,
    to_currency: Optional[str] = None,
    historical_date: Optional[date] = None,
    make_new_column: bool = False,
    api_key: Optional[str] = None,
) -> pd.DataFrame:
    """Convert a column from one currency to another.

    Rates are European Central Bank reference rates, fetched without an
    API key from [Frankfurter](https://www.frankfurter.app). The series
    starts on 1999-01-04. When a date has no published observation
    (weekends and TARGET holidays), the previous published rate is used.

    Set ``JANITOR_CURRENCY_API_URL`` to replace the default endpoint,
    ``FRANKFURTER_API_URL``. The value must be an absolute ``http`` or
    ``https`` URL. It must accept ``base``, ``quotes``, and an optional
    ``date`` query parameter, and return a JSON list of objects with
    ``quote`` and ``rate`` fields. Currency codes and the date are sent
    to that host.

    Latest rates and today's date are requested on every call. An
    earlier historical date is cached for the life of the process.
    ``HRK`` was replaced by EUR on 2023-01-01, ``RUB`` stopped on
    2022-03-02, and ``BGN`` was replaced by EUR on 2026-01-01. Those
    codes still convert on earlier dates.

    The returned DataFrame is a copy. The input is left unchanged.
    Converting a currency to itself keeps the column's dtype.

    Args:
        df: A pandas DataFrame.
        column_name: Name of the column containing monetary values
            to convert.
        from_currency: The base currency to convert from. A three-letter
            ISO code.
        to_currency: The target currency to convert to. A three-letter
            ISO code.
        historical_date: Exchange-rate date. Omit it for the latest
            rate. ``datetime`` values, including timezone-aware values,
            use their calendar date. Rates go back to 1999-01-04.
        make_new_column: When True, store the converted values in
            ``{column_name}_{to_currency}``. When False, replace
            ``column_name`` on the returned copy.
        api_key: Deprecated and ignored. Frankfurter does not require
            a key. A call that still passes the key as the first
            positional argument is accepted with a ``FutureWarning``.

    Returns:
        A DataFrame with the converted currency column.

    Raises:
        TypeError: If ``historical_date`` is not a ``date`` or
            ``datetime``, or an old positional call cannot be mapped
            back onto the current arguments.
        ValueError: If a currency code is not three letters, the date
            is before 1999-01-04 or ``NaT``, the currency is no longer
            published on that date, the service returns HTTP 422, the
            response has no usable rate, or
            ``JANITOR_CURRENCY_API_URL`` is not an http(s) URL.
        ConnectionError: If the request times out, cannot connect, the
            rates endpoint does not return HTTP 200 or 422, or the body
            is not JSON.
        KeyError: If ``column_name`` is not in the DataFrame.

    Examples:
        >>> import pandas as pd
        >>> import janitor.finance
        >>> from datetime import date
        >>> df = pd.DataFrame({"price": [10.0, 20.0, 30.0]})
        >>> df.convert_currency(
        ...     column_name="price",
        ...     from_currency="USD",
        ...     to_currency="EUR",
        ...     historical_date=date(2018, 1, 2),
        ...     make_new_column=True,
        ... )  # doctest: +SKIP
           price  price_EUR
        0   10.0     8.2884
        1   20.0    16.5768
        2   30.0    24.8652
    """
    (
        column_name,
        from_currency,
        to_currency,
        historical_date,
        make_new_column,
        legacy_call,
    ) = normalize_convert_currency_arguments(
        df,
        column_name,
        from_currency,
        to_currency,
        historical_date,
        make_new_column,
        api_key,
    )
    if legacy_call:
        warn(
            "Passing api_key as the first positional argument of "
            "convert_currency is deprecated and ignored. Pass the "
            "column name first: df.convert_currency(column_name, "
            "from_currency, to_currency). api_key will be removed in "
            "a future release.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )
    elif api_key is not None:
        warn(
            "The `api_key` argument of `convert_currency` is deprecated "
            "and ignored. It will be removed in a future release.",
            FutureWarning,
            stacklevel=find_stack_level(),
        )

    if column_name not in df.columns:
        raise KeyError(f"column_name {column_name!r} is not in the DataFrame.")
    from_currency = currency_code(from_currency)
    to_currency = currency_code(to_currency)
    if from_currency == to_currency:
        converted = df[column_name]
    else:
        rate = fetch_exchange_rate(
            from_currency,
            to_currency,
            historical_date,
            currency_api_url(),
        )
        converted = df[column_name] * rate
    new_column_name = column_name
    if make_new_column:
        new_column_name = f"{column_name}_{to_currency}"
    return df.assign(**{new_column_name: converted})


@lru_cache(maxsize=32)
def _inflate_currency(
    country: str = None, currency_year: int = None, to_year: int = None
) -> float:
    """
    Currency inflation for Pandas DataFrame column.
    Helper function for `inflate_currency` method.
    The API used is the World Bank Indicator API:
    https://datahelpdesk.worldbank.org/knowledgebase/articles/889392-about-the-indicators-api-documentation
    """

    # Check all inputs are correct data type
    check("country", country, [str])
    check("currency_year", currency_year, [int])
    check("to_year", to_year, [int])

    # Get WB country abbreviation
    _check_wb_country(country)
    if country in wb_country_dict.keys():
        country = wb_country_dict[country]
    else:
        # `country` is already a correct abbreviation; do nothing
        pass

    _check_wb_years(currency_year)
    _check_wb_years(to_year)

    url = (
        "https://api.worldbank.org/v2/country/"
        + country
        + "/indicator/FP.CPI.TOTL?date="
        + str(min(currency_year, to_year))
        + ":"
        + str(max(currency_year, to_year))
        + "&format=json"
    )

    result = requests.get(url)

    if result.status_code != 200:
        raise ConnectionError(
            "WB Indicator API failed to receive a 200 "
            "response from the server. "
            "Please try again later."
        )

    # The API returns a list of two items;
    # the second item in the list is what we want
    inflation_dict = json.loads(result.text)[1]

    # Error checking
    if inflation_dict is None:
        raise ValueError(
            "The WB Indicator API returned nothing. "
            "This likely means the currency_year and "
            "to_year are outside of the year range for "
            "which the WB has inflation data for the "
            "specified country."
        )

    # Create new dict with only the year and inflation values
    inflation_dict_ready = {
        int(inflation_dict[i]["date"]): float(inflation_dict[i]["value"])
        for i in range(len(inflation_dict))
        if inflation_dict[i]["value"] is not None
    }

    # Error catching
    if currency_year not in inflation_dict_ready.keys():
        raise ValueError(
            f"The WB Indicator API does not have inflation "
            f"data for {currency_year} for {country}."
        )
    if to_year not in inflation_dict_ready.keys():
        raise ValueError(
            f"The WB Indicator API does not have inflation "
            f"data for {to_year} for {country}."
        )

    inflator = inflation_dict_ready[to_year] / inflation_dict_ready[currency_year]
    return inflator


@pf.register_dataframe_method
def inflate_currency(
    df: pd.DataFrame,
    column_name: str = None,
    country: str = None,
    currency_year: int = None,
    to_year: int = None,
    make_new_column: bool = False,
) -> pd.DataFrame:
    """
    Inflates a column of monetary values from one year to another, based on
    the currency's country.

    The provided country can be any economy name or code from the World Bank
    [list of economies](https://databank.worldbank.org/data/download/site-content/CLASS.xls).

    **Note**: This method mutates the original DataFrame.

    Examples:

        >>> import pandas as pd
        >>> import janitor.finance
        >>> df = pd.DataFrame({"profit": [100.10, 200.20, 300.30, 400.40, 500.50]})
        >>> df
           profit
        0   100.1
        1   200.2
        2   300.3
        3   400.4
        4   500.5
        >>> df.inflate_currency(
        ...     column_name="profit",
        ...     country="USA",
        ...     currency_year=2015,
        ...     to_year=2018,
        ...     make_new_column=True,
        ... )
           profit  profit_2018
        0   100.1   106.050596
        1   200.2   212.101191
        2   300.3   318.151787
        3   400.4   424.202382
        4   500.5   530.252978

    Args:
        df: A pandas DataFrame.
        column_name: Name of the column containing monetary
            values to inflate.
        country: The country associated with the currency being inflated.
            May be any economy or code from the World Bank
            [List of economies](https://databank.worldbank.org/data/download/site-content/CLASS.xls).
        currency_year: The currency year to inflate from.
            The year should be 1960 or later.
        to_year: The currency year to inflate to.
            The year should be 1960 or later.
        make_new_column: Generates new column for inflated currency if
            True, otherwise, inflates currency in place.

    Returns:
        The DataFrame with inflated currency column.
    """  # noqa: E501

    inflator = _inflate_currency(country, currency_year, to_year)

    if make_new_column:
        new_column_name = column_name + "_" + str(to_year)
        df[new_column_name] = df[column_name] * inflator

    else:
        df[column_name] = df[column_name] * inflator

    return df


def convert_stock(stock_symbol: str) -> str:
    """
    This function takes in a stock symbol as a parameter,
    queries an API for the companies full name and returns
    it

    Examples:

        ```python
        import janitor.finance

        janitor.finance.convert_stock("aapl")
        ```

    Args:
        stock_symbol: Stock ticker Symbol

    Raises:
        ConnectionError: Internet connection is not available

    Returns:
        Full company name
    """
    if is_connected("www.google.com"):
        stock_symbol = stock_symbol.upper()
        return get_symbol(stock_symbol)
    else:
        raise ConnectionError("Connection Error: Client Not Connected to Internet")


def get_symbol(symbol: str) -> Optional[str]:
    """
    This is a helper function to get a companies full
    name based on the stock symbol.

    Examples:

        ```python
        import janitor.finance

        janitor.finance.get_symbol("aapl")
        ```

    Args:
        symbol: This is our stock symbol that we use
            to query the api for the companies full name.

    Returns:
        Company full name
    """
    result = requests.get(
        "http://d.yimg.com/autoc."
        + "finance.yahoo.com/autoc?query={}&region=1&lang=en".format(symbol)
    ).json()

    for x in result["ResultSet"]["Result"]:
        if x["symbol"] == symbol:
            return x["name"]
        else:
            return None
