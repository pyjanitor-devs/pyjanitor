import numpy as np
import pandas as pd
import pytest

import janitor


@pytest.mark.functions
def test_round_half_up_scalar():
    assert janitor.round_half_up(12.5) == 13.0
    assert janitor.round_half_up(1.125, digits=2) == pytest.approx(1.13)
    assert janitor.round_half_up(-0.5) == -1.0


@pytest.mark.functions
def test_round_half_up_array():
    result = janitor.round_half_up([0.5, 1.5, 2.5])
    assert np.allclose(result, [1.0, 2.0, 3.0])


@pytest.mark.functions
def test_round_half_up_series_preserves_index():
    series = pd.Series([0.5, 1.5], name="vals")
    result = janitor.round_half_up(series)
    assert isinstance(result, pd.Series)
    assert result.name == "vals"
    assert result.tolist() == [1.0, 2.0]


@pytest.mark.functions
def test_signif_half_up_scalar():
    assert janitor.signif_half_up(12.5, digits=2) == 13.0
    assert janitor.signif_half_up(1.125, digits=3) == pytest.approx(1.13)
    assert janitor.signif_half_up(-2.5, digits=1) == -3.0


@pytest.mark.functions
def test_signif_half_up_array_with_nan_inf():
    values = np.array([0.0, np.nan, np.inf, 123.4])
    result = janitor.signif_half_up(values, digits=2)
    assert result[0] == 0
    assert np.isnan(result[1])
    assert np.isinf(result[2])
