import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis.extra.pandas import series


@given(s=series(dtype=np.float64))
@settings(deadline=None)
def test_ecdf(s):
    """A simple execution test."""
    if s.isna().sum() > 0:
        with pytest.raises(ValueError):
            x, y = s.ecdf()
    else:
        x, y = s.ecdf()
        assert len(x) == len(y)


@given(s=series(dtype=object))
@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None)
def test_ecdf_string(s):
    """Test that type enforcement is in place."""
    with pytest.raises(TypeError):
        x, y = s.ecdf()
