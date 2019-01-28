import pandas as pd


def test_convert_excel_date():
    df = (
        pd.read_excel("examples/dirty_data.xlsx")
        .clean_names()
        .convert_excel_date("hire_date")
    )

    assert df["hire_date"].dtype == "M8[ns]"
