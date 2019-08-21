""" General purpose data cleaning functions. """

from .. import functions as janitor_func
from . import backend


@backend.register_dataframe_method
def clean_names(df):
    """Clean column names."""

    cols = [
        f"`{col}` AS {janitor_func._remove_special(col.replace(' ', '_'))}"
        for col in df.columns
    ]
    return df.selectExpr(*cols)
