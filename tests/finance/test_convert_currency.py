import pytest
import requests

import janitor.finance


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
