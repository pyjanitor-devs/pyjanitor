import numpy as np, pandas as pd, pytest
from janitor.functions.scale_mad import scale_mad

def test_scales_numeric_columns_default():
    df = pd.DataFrame({"x":[1,2,3,4], "y":[10,10,10,10]})
    res = scale_mad(df)
    assert set(res.columns) == {"x","y"}
    assert (res["y"] == 10).all()
    assert abs(res["x"].median()) < 1e-9

def test_zero_mad_center_only():
    df = pd.DataFrame({"y":[10,10,10,10]})
    res = scale_mad(df, zero_mad="one")
    assert np.isclose(res["y"].mean(), 0.0)

def test_suffix_and_clip():
    df = pd.DataFrame({"x":[1,2,3,100]})
    res = scale_mad(df, columns=["x"], clip=3, suffix="_mad")
    assert "x_mad" in res.columns and (res["x_mad"].abs() <= 3).all()

def test_callable_column_selector():
    df = pd.DataFrame({"a":[1,2,3], "b":["x","y","z"]})
    res = scale_mad(df, columns=lambda d: d.select_dtypes("number").columns, suffix="_mad")
    assert "a_mad" in res.columns

def test_zero_mad_raise():
    df = pd.DataFrame({"y":[1,1,1]})
    with pytest.raises(ValueError):
        scale_mad(df, columns=["y"], zero_mad="raise")
