"""
Finance-specific data cleaning functions.
"""

import json
from datetime import date
from functools import lru_cache

import pandas as pd
import pandas_flavor as pf
import requests

from janitor.errors import JanitorError

from .utils import check, deprecated_alias, is_connected


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
            f"currency {currency} not in supported currency set, "
            f"{currency_set}"
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


# @lru_cache(maxsize=32)
# def _convert_currency(
#     api_key: str,
#     from_currency: str = None,
#     to_currency: str = None,
#     historical_date: Optional[date] = None,
# ) -> float:
#     """
#     Currency conversion for Pandas DataFrame column.

#     Helper function for `convert_currency` method.

#     The API used is https://exchangeratesapi.io/.
#     """

#     url = "http://api.exchangeratesapi.io"

#     if historical_date:
#         check("historical_date", historical_date, [datetime, date])
#         if isinstance(historical_date, datetime):
#             if historical_date < datetime(1999, 1, 4):
#                 raise ValueError(
#                     "historical_date:datetime must be later than 1999-01-04!"
#                 )
#             string_date = str(historical_date)[:10]
#         else:
#             if historical_date < date(1999, 1, 4):
#                 raise ValueError(
#                     "historical_date:date must be later than 1999-01-04!"
#                 )
#             string_date = str(historical_date)
#         url = url + "/%s" % string_date
#     else:
#         url = url + "/latest"

#     _check_currency(from_currency)
#     _check_currency(to_currency)

#     payload = {
#         # "base": from_currency,
#         "symbols": to_currency,
#         "access_key": api_key,
#     }

#     result = requests.get(url, params=payload)

#     if result.status_code != 200:
#         raise ConnectionError(
#             "Exchange Rate API failed to receive a 200 "
#             "response from the server. "
#             "Please try again later."
#         )

#     currency_dict = json.loads(result.text)
#     rate = currency_dict["rates"][to_currency]
#     return rate


@pf.register_dataframe_method
@deprecated_alias(colname="column_name")
def convert_currency(
    df: pd.DataFrame,
    api_key: str,
    column_name: str = None,
    from_currency: str = None,
    to_currency: str = None,
    historical_date: date = None,
    make_new_column: bool = False,
) -> pd.DataFrame:
    """Deprecated function."""
    raise JanitorError(
        "The `convert_currency` function has been temporarily disabled due to "
        "exchangeratesapi.io disallowing free pinging of its API. "
        "(Our tests started to fail due to this issue.) "
        "There is no easy way around this problem "
        "except to find a new API to call on."
        "Please comment on issue #829 "
        "(https://github.com/pyjanitor-devs/pyjanitor/issues/829) "
        "if you know of an alternative API that we can call on, "
        "otherwise the function will be removed in pyjanitor's 1.0 release."
    )


# @pf.register_dataframe_method
# @deprecated_alias(colname="column_name")
# def convert_currency(
#     df: pd.DataFrame,
#     api_key: str,
#     column_name: str = None,
#     from_currency: str = None,
#     to_currency: str = None,
#     historical_date: date = None,
#     make_new_column: bool = False,
# ) -> pd.DataFrame:
#     """
#     Converts a column from one currency to another, with an option to
#     convert based on historical exchange values.

#     On April 10 2021,
#     we discovered that there was no more free API available.
#     Thus, an API key is required to perform currency conversion.
#     API keys should be set as an environment variable,
#     for example, `EXCHANGE_RATE_API_KEY``,
#     and then passed into the function
#     by calling on `os.getenv("EXCHANGE_RATE_APIKEY")``.

#     :param df: A pandas dataframe.
#     :param api_key: exchangeratesapi.io API key.
#     :param column_name: Name of the new column. Should be a string, in order
#         for the column name to be compatible with the Feather binary
#         format (this is a useful thing to have).
#     :param from_currency: The base currency to convert from.
#         May be any of: currency_set = {"AUD", "BGN", "BRL", "CAD", "CHF",
#         "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HRK", "HUF", "IDR",
#         "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD",
#         "PHP", "PLN", "RON", "RUB", "SEK", "SGD", "THB", "TRY", "USD",
#         "ZAR"}
#     :param to_currency: The target currency to convert to.
#         May be any of: currency_set = {"AUD", "BGN", "BRL", "CAD", "CHF",
#         "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HRK", "HUF", "IDR",
#         "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD",
#         "PHP", "PLN", "RON", "RUB", "SEK", "SGD", "THB", "TRY", "USD",
#         "ZAR"}
#     :param historical_date: If supplied,
#         get exchange rate on a certain date.
#         If not supplied, get the latest exchange rate.
#         The exchange rates go back to Jan. 4, 1999.
#     :param make_new_column: Generates new column
#         for converted currency if True,
#         otherwise, converts currency in place.
#     :returns: The dataframe with converted currency column.

#     .. code-block:: python

#         import pandas as pd
#         import janitor
#         from datetime import date

#         data_dict = {
#             "a": [1.23452345, 2.456234, 3.2346125] * 3,
#             "Bell__Chart": [1/3, 2/7, 3/2] * 3,
#             "decorated-elephant": [1/234, 2/13, 3/167] * 3,
#             "animals": ["rabbit", "leopard", "lion"] * 3,
#             "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
#         }

#         example_dataframe = pd.DataFrame(data_dict)

#     Example: Converting a column from one currency to another
#     using rates from 01/01/2018.

#     .. code-block:: python

#         example_dataframe.convert_currency('a', from_currency='USD',
#         to_currency='EUR', historical_date=date(2018,1,1))

#     Output:

#     .. code-block:: python

#                     a  Bell__Chart  decorated-elephant  animals     cities
#         0  1.029370     0.333333            0.004274   rabbit  Cambridge
#         1  2.048056     0.285714            0.153846  leopard   Shanghai
#         2  2.697084     1.500000            0.017964     lion      Basel
#         3  1.029370     0.333333            0.004274   rabbit  Cambridge
#         4  2.048056     0.285714            0.153846  leopard   Shanghai
#         5  2.697084     1.500000            0.017964     lion      Basel
#         6  1.029370     0.333333            0.004274   rabbit  Cambridge
#         7  2.048056     0.285714            0.153846  leopard   Shanghai
#         8  2.697084     1.500000            0.017964     lion      Basel
#     """

#     rate = _convert_currency(
#         api_key, from_currency, to_currency, historical_date
#     )

#     if make_new_column:
#         # new_column_name = column_name + "_" + to_currency
#         column_name = column_name + "_" + to_currency

#     df = df.assign(column_name=df[column_name] * rate)

#     return df


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

    inflator = (
        inflation_dict_ready[to_year] / inflation_dict_ready[currency_year]
    )
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
    [list of economies]
    (https://databank.worldbank.org/data/download/site-content/CLASS.xls).

    **Note**: This method mutates the original DataFrame.

    Method chaining usage example:

    >>> import pandas as pd
    >>> import janitor.finance
    >>> df = pd.DataFrame({"profit":[100.10, 200.20, 300.30, 400.40, 500.50]})
    >>> df
       profit
    0   100.1
    1   200.2
    2   300.3
    3   400.4
    4   500.5
    >>> df.inflate_currency(
    ...    column_name='profit',
    ...    country='USA',
    ...    currency_year=2015,
    ...    to_year=2018,
    ...    make_new_column=True
    ... )
       profit  profit_2018
    0   100.1   106.050596
    1   200.2   212.101191
    2   300.3   318.151787
    3   400.4   424.202382
    4   500.5   530.252978


    :param df: A pandas DataFrame.
    :param column_name: Name of the column containing monetary
        values to inflate.
    :param country: The country associated with the currency being inflated.
        May be any economy or code from the World Bank [List of economies]
        (https://databank.worldbank.org/data/download/site-content/CLASS.xls).
    :param currency_year: The currency year to inflate from.
        The year should be 1960 or later.
    :param to_year: The currency year to inflate to.
        The year should be 1960 or later.
    :param make_new_column: Generates new column for inflated currency if
        True, otherwise, inflates currency in place.
    :returns: The dataframe with inflated currency column.
    """

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

    Functional usage example:

    ```python
    import janitor.finance

    janitor.finance.convert_stock("aapl")
    ```

    :param stock_symbol: Stock ticker Symbol
    :raises ConnectionError: Internet connection is not available
    :returns: Full company name
    """
    if is_connected("www.google.com"):
        stock_symbol = stock_symbol.upper()
        return get_symbol(stock_symbol)
    else:
        raise ConnectionError(
            "Connection Error: Client Not Connected to Internet"
        )


def get_symbol(symbol: str):
    """
    This is a helper function to get a companies full
    name based on the stock symbol.

    Functional usage example:

    ```python
    import janitor.finance

    janitor.finance.get_symbol("aapl")
    ```

    :param symbol: This is our stock symbol that we use
        to query the api for the companies full name.
    :return: Company full name
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
