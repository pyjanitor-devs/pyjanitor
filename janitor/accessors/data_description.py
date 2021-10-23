from typing import Dict, List, Union
import pandas_flavor as pf
import pandas as pd


@pf.register_dataframe_accessor("data_description")
class DataDescription:
    """High-level description of data present in this DataFrame.

    This is a custom data accessor.
    """

    def __init__(self, data):
        """Initialize DataDescription class."""
        self._data = data
        self._desc = {}

    def _get_data_df(self) -> pd.DataFrame:
        df = self._data

        data_dict = {}
        data_dict["column_name"] = df.columns.tolist()
        data_dict["type"] = df.dtypes.tolist()
        data_dict["count"] = df.count().tolist()
        data_dict["pct_missing"] = (1 - (df.count() / len(df))).tolist()
        data_dict["description"] = [self._desc.get(c, "") for c in df.columns]

        return pd.DataFrame(data_dict).set_index("column_name")

    @property
    def df(self) -> pd.DataFrame:
        """Get a table of descriptive information in a DataFrame format."""
        return self._get_data_df()

    def __repr__(self):
        """Human-readable representation of the `DataDescription` object."""
        return str(self._get_data_df())

    def display(self):
        """Print the table of descriptive information about this DataFrame."""
        print(self)

    def set_description(self, desc: Union[List, Dict]):
        """Update the description for each of the columns in the DataFrame.

        :param desc: The structure containing the descriptions to update
        :raises ValueError: if length of description list does not match
            number of columns in DataFrame.
        """
        if isinstance(desc, list):
            if len(desc) != len(self._data.columns):
                raise ValueError(
                    "Length of description list "
                    f"({len(desc)}) does not match number of columns in "
                    f"DataFrame ({len(self._data.columns)})"
                )

            self._desc = dict(zip(self._data.columns, desc))

        elif isinstance(desc, dict):
            self._desc = desc
