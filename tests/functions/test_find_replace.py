import numpy as np
import pandas as pd
import pytest


@pytest.mark.test
@pytest.mark.functions
def test_find_replace():
    df= pd.DataFrame({"a": [1, np.nan, 3], "b": [2, 3, np.nan], "c": [2, np.nan, 9]})
    df1= df.find_replace(["a","c"], {1: 2, 3: 4, np.nan: 5})
    df2= pd.DataFrame(
        {"a": [2, 5, 4], "b": [2, 3, np.nan], "c": [2, 5, 9]})
    assert (df1[["a","c"]]==df2[["a","c"]]).all().all()
