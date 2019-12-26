import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.pandas import series


@given(s=series(dtype=np.number))
def test_ecdf(s):
    x, y = s.ecdf()


@given(s=series(dtype=str))
def test_ecdf_string(s):
    with pytest.raises(TypeError):
        x, y = s.ecdf()
