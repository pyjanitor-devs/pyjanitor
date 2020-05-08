# df.convert_currency()

## Description
This method converts a column from one currency to another, with an option to convert based on historical exchange values.

## Parameters
### df
A pandas dataframe.

### colname
The column which to apply the currency conversion.

### from_currency
The base currency to convert from.

May be any of: currency_set = {"AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HRK", "HUF", "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PLN", "RON", "RUB", "SEK", "SGD", "THB", "TRY", "USD", "ZAR"}

### to_currency
The target currency to convert to.

May be any of: currency_set = {"AUD", "BGN", "BRL", "CAD", "CHF", "CNY", "CZK", "DKK", "EUR", "GBP", "HKD", "HRK", "HUF", "IDR", "ILS", "INR", "ISK", "JPY", "KRW", "MXN", "MYR", "NOK", "NZD", "PHP", "PLN", "RON", "RUB", "SEK", "SGD", "THB", "TRY", "USD", "ZAR"}

### historical_date
If supplied, get exchange rate on a certain date. If not supplied, get the latest exchange rate. The exchange rates go back to Jan. 4, 1999.

## Setup
```python
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
```

## Example: Converting a column from one currency to another using rates from 01/01/2018
 ```python
example_dataframe = pd.DataFrame(data_dict)

example_dataframe.convert_currency(
    'a',
    from_currency='USD',
    to_currency='EUR',
    historical_date=date(2018, 1, 1)
)
```

### Output

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

__Note:__ Since this hits an API for the currency conversions, you need internet access to fetch the results. However, results are locally cached for future calls, so if you need to run the same code again, but briefly lost internet access, this should still work.
