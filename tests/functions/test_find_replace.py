import numpy as np
import pandas as pd
import pytest


@pytest.mark.test
@pytest.mark.functions
def test_find_replace():
    df = pd.DataFrame(
        {"a": [1, np.nan, 3], "b": [2, 3, 1], "c": [2, np.nan, 9]}
    ).find_replace("a", {1: 2, 3: 4, np.nan: 5})

    assert not all(pd.isnull(df['a']))