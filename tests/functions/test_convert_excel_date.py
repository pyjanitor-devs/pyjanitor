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

    assert df["hire_date"].dtype == "M8[ns]"
