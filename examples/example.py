import pandas as pd

import janitor as jn

df = (
    pd.read_excel("dirty_data.xlsx")
    .clean_names()
    .remove_empty()
    .rename_column("%_allocated", "percent_allocated")
    .rename_column("full_time_", "full_time")
    .coalesce(["certification", "certification_1"], "certification")
    .encode_categorical(["subject", "employee_status", "full_time"])
    .convert_excel_date("hire_date")
)

print(df)
print(df.original_names)
