import pandas as pd


def test_convert_unix_date():
    unix = [
        "1284101485",
        1_284_101_486,
        "1284101487000",
        1_284_101_488_000,
        "1284101489",
        "1284101490",
        -2_147_483_648,
        2_147_483_648,
    ]
    df = pd.DataFrame(unix, columns=["dates"]).convert_unix_date("dates")

    assert df["dates"].dtype == "M8[ns]"
