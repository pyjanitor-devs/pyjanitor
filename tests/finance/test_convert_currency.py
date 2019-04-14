import pytest
import requests

import janitor.finance


@pytest.mark.finance
@pytest.mark.xfail
def test_make_currency_api_request():
    r = requests.get("https://api.exchangeratesapi.io")
    assert r.status_code == 200


@pytest.mark.finance
def test_make_new_currency_col(dataframe):
    df = dataframe.convert_currency("a", "USD", "USD", make_new_column=True)
    assert all(df["a"] == df["a_USD"])
