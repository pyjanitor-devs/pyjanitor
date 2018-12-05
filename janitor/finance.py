"""Finance Submodule for PyJanitor functions """

from functools import lru_cache
from datetime import datetime
from datetime import date
import json
import pandas_flavor as pf

from janitor import check

import requests

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


def _check_currency(currency):
    if currency not in currency_set:
        raise ValueError(
            f"currency {currency} not in supported currency set, "
            f"{currency_set}"
        )


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
def convert_currency(
    df,
    colname: str = None,
    from_currency: str = None,
    to_currency: str = None,
    historical_date: date = None,
    make_new_column: bool = False,
):
    """
        Converts a column from one currency to another, with an option to
        convert based on historical exchange values.

        :param df: A pandas dataframe.
        :param colname: Name of the new column. Should be a string, in order
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
        new_col_name = colname + "_" + to_currency
        df[new_col_name] = df[colname] * rate

    else:
        df[colname] = df[colname] * rate

    return df
