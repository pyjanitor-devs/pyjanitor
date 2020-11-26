import numpy as np
import pandas as pd
import pytest
from pandas.api.types import CategoricalDtype
from pandas.testing import assert_frame_equal


import janitor

df = pd.DataFrame({"col1": [60, 70], "col2": [80, 90]})

print(df, end="\n\n")

print(df.as_categorical(col1 = (None, "sort")))