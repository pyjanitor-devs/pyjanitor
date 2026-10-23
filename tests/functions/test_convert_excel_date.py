from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.functions
def test_convert_excel_date():
    # using openpyxl as the engine staves off an error that crops up
    # during the CI build up with xlrd
    df = (
        pd.read_excel(
            Path(pytest.EXAMPLES_DIR) / "notebooks" / "dirty_data.xlsx",
            engine="openpyxl",
        )
        .clean_names()
        .convert_excel_date("hire_date")
    )

    assert pd.api.types.is_datetime64_dtype(df["hire_date"])
    expected = pd.Series(
        [
            "2008-08-30",
            "2008-08-30",
            "2001-08-15",
            "1975-05-01",
            "2013-06-06",
            "1930-03-20",
            "1930-03-20",
            None,
            "1990-05-01",
            "1976-06-08",
            "2015-08-05",
            "1995-01-01",
            "2009-09-15",
        ],
        name="hire_date",
    )
    pd.testing.assert_series_equal(
        df["hire_date"].dt.strftime("%Y-%m-%d"), expected, check_dtype=False
    )
