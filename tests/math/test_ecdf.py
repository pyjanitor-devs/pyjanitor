import numpy as np
import pytest
from hypothesis import given
from hypothesis import settings
from hypothesis.extra.pandas import series


@given(s=series(dtype=np.number))
@settings(deadline=None)
def test_ecdf(s):
    """A simple execution test."""
    if s.isna().sum() > 0:
        with pytest.raises(ValueError):
            x, y = s.ecdf()
    else:
        x, y = s.ecdf()
        assert len(x) == len(y)


@given(s=series(dtype=str))
def test_ecdf_string(s):
    """Test that type enforcement is in place."""
    with pytest.raises(TypeError):
        x, y = s.ecdf()
