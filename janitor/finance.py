"""
Finance-specific data cleaning functions.
"""

import json
from datetime import date, datetime
from functools import lru_cache, partial

import pandas as pd
import pandas_flavor as pf
import requests

from janitor import check

from .utils import (
    _clean_accounting_column,
    _currency_column_to_numeric,
    _replace_empty_string_with_none,
    _replace_original_empty_string_with_none,
    deprecated_alias,
)

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
    if currency not in currency_set:
        raise ValueError(
            f"currency {currency} not in supported currency set, "
            f"{currency_set}"
        )


def _check_wb_country(country: str):
    if (country not in wb_country_dict.keys()) & (
        country not in wb_country_dict.values()
    ):
        raise ValueError(
            f"country {country} not in supported World Bank country dict, "
            f"{wb_country_dict}"
        )


def _check_wb_years(year: int):
    if year < 1960:
        raise ValueError(f"year value must be 1960 or later")


@lru_cache(maxsize=32)
def _convert_currency(
    from_currency: str = None,
    to_currency: str = None,
    historical_date: date = None,
):
    """
    Currency conversion for Pandas DataFrame column.

    Helper function for `convert_currency` method.

    The API used is: https://exchangeratesapi.io/
    """

    url = "https://api.exchangeratesapi.io"

    if historical_date:
        check("historical_date", historical_date, [datetime, date])
        if isinstance(historical_date, datetime):
            if historical_date < datetime(1999, 1, 4):
                raise ValueError(
                    "historical_date:datetime must be later than 1999-01-04!"
                )
            string_date = str(historical_date)[:10]
        else:
            if historical_date < date(1999, 1, 4):
                raise ValueError(
                    "historical_date:date must be later than 1999-01-04!"
                )
            string_date = str(historical_date)
        url = url + "/%s" % string_date
    else:
        url = url + "/latest"

    _check_currency(from_currency)
    _check_currency(to_currency)

    payload = {"base": from_currency, "symbols": to_currency}

    result = requests.get(url, params=payload)

    if result.status_code != 200:
        raise ConnectionError(
            "Exchange Rate API failed to receive a 200 "
            "response from the server. "
            "Please try again later."
        )

    currency_dict = json.loads(result.text)
    rate = currency_dict["rates"][to_currency]
    return rate


@pf.register_dataframe_method
@deprecated_alias(col_name="column_name", type="cleaning_style")
def currency_column_to_numeric(
    df: pd.DataFrame,
    column_name,
    cleaning_style: str = None,
    cast_non_numeric: dict = None,
    fill_all_non_numeric: float = None,
    remove_non_numeric: bool = False,
) -> pd.DataFrame:
    """
    Convert currency column to numeric.

    This method does not mutate the original DataFrame.

    This method allows one to take a column containing currency values,
    inadvertently imported as a string, and cast it as a float. This is
    usually the case when reading CSV files that were modified in Excel.
    Empty strings (i.e. `''`) are retained as `NaN` values.

    :param df: The DataFrame
    :param column_name: The column to modify
    :param cleaning_style: What style of cleaning to perform. If None, standard
        cleaning is applied. Options are:

            * 'accounting':
            Replaces numbers in parentheses with negatives, removes commas.

    :param cast_non_numeric: A dict of how to coerce certain strings. For
        example, if there are values of 'REORDER' in the DataFrame,
        {'REORDER': 0} will cast all instances of 'REORDER' to 0.
    :param fill_all_non_numeric: Similar to `cast_non_numeric`, but fills all
        strings to the same value. For example,  fill_all_non_numeric=1, will
        make everything that doesn't coerce to a currency 1.
    :param remove_non_numeric: Will remove rows of a DataFrame that contain
        non-numeric values in the `column_name` column. Defaults to `False`.
    :returns: A pandas DataFrame.
    """
    # TODO: Convert this to a notebook.
    # :Example Setup:

    # .. code-block:: python

    #     import pandas as pd
    #     import janitor
    #     data = {
    #         "a": ["-$1.00", "", "REPAY"] * 2 + ["$23.00", "",
    # "Other Account"],
    #         "Bell__Chart": [1.234_523_45, 2.456_234, 3.234_612_5] * 3,
    #         "decorated-elephant": [1, 2, 3] * 3,
    #         "animals@#$%^": ["rabbit", "leopard", "lion"] * 3,
    #         "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
    #     }
    #     df = pd.DataFrame(data)

    # :Example 1: Coerce numeric values in column to float:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a")

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2   NaN     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5   NaN     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai
    #     8   NaN     3.234612                   3         lion      Basel

    # :Example 2: Coerce numeric values in column to float, and replace a
    # string\
    # value with a specific value:

    # .. code-block:: python

    #     cast_non_numeric = {"REPAY": 22}
    #     df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2  22.0     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5  22.0     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai
    #     8   NaN     3.234612                   3         lion      Basel

    # :Example 3: Coerce numeric values in column to float, and replace all\
    #     string value with a specific value:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a", fill_all_non_numeric=35)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2  35.0     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5  35.0     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai
    #     8  35.0     3.234612                   3         lion      Basel

    # :Example 4: Coerce numeric values in column to float, replace a string\
    #     value with a specific value, and replace remaining string values
    # with\
    #     a specific value:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric,
    #     fill_all_non_numeric=35)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2  22.0     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5  22.0     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai
    #     8  35.0     3.234612                   3         lion      Basel

    # :Example 5: Coerce numeric values in column to float, and remove string\
    #     values:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a", remove_non_numeric=True)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai

    # :Example 6: Coerce numeric values in column to float, replace a string\
    #     value with a specific value, and remove remaining string values:

    # .. code-block:: python

    #     df.currency_column_to_numeric("a", cast_non_numeric=cast_non_numeric,
    #     remove_non_numeric=True)

    # :Output:

    # .. code-block:: python

    #           a  Bell__Chart  decorated-elephant animals@#$%^     cities
    #     0  -1.0     1.234523                   1       rabbit  Cambridge
    #     1   NaN     2.456234                   2      leopard   Shanghai
    #     2  22.0     3.234612                   3         lion      Basel
    #     3  -1.0     1.234523                   1       rabbit  Cambridge
    #     4   NaN     2.456234                   2      leopard   Shanghai
    #     5  22.0     3.234612                   3         lion      Basel
    #     6  23.0     1.234523                   1       rabbit  Cambridge
    #     7   NaN     2.456234                   2      leopard   Shanghai

    check("column_name", column_name, [str])

    column_series = df[column_name]
    if cleaning_style == "accounting":
        df.loc[:, column_name] = df[column_name].apply(
            _clean_accounting_column
        )
        return df

    if cast_non_numeric:
        check("cast_non_numeric", cast_non_numeric, [dict])

    _make_cc_patrial = partial(
        _currency_column_to_numeric, cast_non_numeric=cast_non_numeric
    )

    column_series = column_series.apply(_make_cc_patrial)

    if remove_non_numeric:
        df = df.loc[column_series != "", :]

    # _replace_empty_string_with_none is applied here after the check on
    # remove_non_numeric since "" is our indicator that a string was coerced
    # in the original column
    column_series = column_series.apply(_replace_empty_string_with_none)

    if fill_all_non_numeric is not None:
        check("fill_all_non_numeric", fill_all_non_numeric, [int, float])
        column_series = column_series.fillna(fill_all_non_numeric)

    column_series = column_series.apply(
        _replace_original_empty_string_with_none
    )

    df = df.assign(**{column_name: pd.to_numeric(column_series)})

    return df


@pf.register_dataframe_method
@deprecated_alias(colname="column_name")
def convert_currency(
    df: pd.DataFrame,
    column_name: str = None,
    from_currency: str = None,
    to_currency: str = None,
    historical_date: date = None,
    make_new_column: bool = False,
) -> pd.DataFrame:
    """
    Converts a column from one currency to another, with an option to
    convert based on historical exchange values.

    This method mutates the original DataFrame.

    :param df: A pandas dataframe.
    :param column_name: Name of the new column. Should be a string, in order
        for the column name to be compatible with the Feather binary
        format (this is a useful thing to have).
    :param from_currency: The base currency to convert from.
        May be any of: currency_set = {"AUD", "BGN", "BRL", "CAD", "CHF",
        "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HRK", "HUF", "IDR",
        "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD",
        "PHP", "PLN", "RON", "RUB", "SEK", "SGD", "THB", "TRY", "USD",
        "ZAR"}
    :param to_currency: The target currency to convert to.
        May be any of: currency_set = {"AUD", "BGN", "BRL", "CAD", "CHF",
        "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HRK", "HUF", "IDR",
        "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD",
        "PHP", "PLN", "RON", "RUB", "SEK", "SGD", "THB", "TRY", "USD",
        "ZAR"}
    :param historical_date: If supplied, get exchange rate on a certain\
        date. If not supplied, get the latest exchange rate. The exchange\
        rates go back to Jan. 4, 1999.
    :param make_new_column: Generates new column for converted currency if
        True, otherwise, converts currency in place.

    :Setup:

    .. code-block:: python

        import pandas as pd
        import janitor
        from datetime import date

        data_dict = {
            "a": [1.23452345, 2.456234, 3.2346125] * 3,
            "Bell__Chart": [1/3, 2/7, 3/2] * 3,
            "decorated-elephant": [1/234, 2/13, 3/167] * 3,
            "animals": ["rabbit", "leopard", "lion"] * 3,
            "cities": ["Cambridge", "Shanghai", "Basel"] * 3,
        }

        example_dataframe = pd.DataFrame(data_dict)

    :Example: Converting a column from one currency to another using rates

    from 01/01/2018:

    .. code-block:: python

        example_dataframe.convert_currency('a', from_currency='USD',
        to_currency='EUR', historical_date=date(2018,1,1))

    :Output:

    .. code-block:: python

                    a  Bell__Chart  decorated-elephant  animals     cities
        0  1.029370     0.333333            0.004274   rabbit  Cambridge
        1  2.048056     0.285714            0.153846  leopard   Shanghai
        2  2.697084     1.500000            0.017964     lion      Basel
        3  1.029370     0.333333            0.004274   rabbit  Cambridge
        4  2.048056     0.285714            0.153846  leopard   Shanghai
        5  2.697084     1.500000            0.017964     lion      Basel
        6  1.029370     0.333333            0.004274   rabbit  Cambridge
        7  2.048056     0.285714            0.153846  leopard   Shanghai
        8  2.697084     1.500000            0.017964     lion      Basel

    """

    rate = _convert_currency(from_currency, to_currency, historical_date)

    if make_new_column:
        new_column_name = column_name + "_" + to_currency
        df[new_column_name] = df[column_name] * rate

    else:
        df[column_name] = df[column_name] * rate

    return df


@lru_cache(maxsize=32)
def _inflate_currency(
    country: str = None, currency_year: int = None, to_year: int = None
):
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
    list of economies:
    https://databank.worldbank.org/data/download/site-content/CLASS.xls.

    This method mutates the original DataFrame.

    Functional usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.finance

        df = pd.DataFrame(...)

        df = janitor.finance.inflate_currency(
            df=df,
            column_name='profit',
            country='USA',
            currency_year=2015,
            to_year=2018,
            make_new_column=True
        )

    Method chaining usage example:

    .. code-block:: python

        import pandas as pd
        import janitor.finance

        df = pd.DataFrame(...)
        df = df.inflate_currency(
            column_name='profit',
            country='USA',
            currency_year=2015,
            to_year=2018,
            make_new_column=True
        )

    :param df: A pandas dataframe.
    :param column_name: Name of the column containing monetary
        values to inflate.
    :param country: The country associated with the currency being inflated.
        May be any economy or code from the World Bank list of economies:
        https://databank.worldbank.org/data/download/site-content/CLASS.xls.
    :param currency_year: The currency year to inflate from.
        The year should be 1960 or later.
    :param to_year: The currency year to inflate to.
        The year should be 1960 or later.
    :param make_new_column: Generates new column for inflated currency if
        True, otherwise, inflates currency in place.
    """

    inflator = _inflate_currency(country, currency_year, to_year)

    if make_new_column:
        new_column_name = column_name + "_" + str(to_year)
        df[new_column_name] = df[column_name] * inflator

    else:
        df[column_name] = df[column_name] * inflator

    return df
