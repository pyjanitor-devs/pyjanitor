from pathlib import Path

import pandas as pd
import pytest


@pytest.mark.functions
def test_convert_excel_date():
    df = (
        pd.read_excel(
            Path(pytest.EXAMPLES_DIR) / "notebooks" / "dirty_data.xlsx",
            engine="openpyxl",
        )
        .clean_names()
        .convert_excel_date("hire_date")
    )

    assert df["hire_date"].dtype == "M8[ns]"
