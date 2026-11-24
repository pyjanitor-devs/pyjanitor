"""Tests for the winsorize function."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal, assert_series_equal

import janitor  # noqa: F401


@pytest.fixture
def sales():
    """A small sales table with one obvious outlier per store."""
    return pd.DataFrame(
        {
            "store": ["a", "a", "a", "a", "b", "b", "b", "b"],
            "amount": [1.0, 2.0, 3.0, 40.0, 100.0, 200.0, 300.0, 4000.0],
            "units": [1, 2, 3, 4, 5, 6, 7, 8],
            "promo": [True, False, True, False, True, False, True, False],
        }
    )


@pytest.mark.functions
def test_winsorize_default_selects_numeric_columns(sales):
    """By default, numeric columns are capped; text and booleans are untouched."""
    original = sales.copy()

    result = sales.winsorize()

    for col in ["amount", "units"]:
        values = sales[col].astype(float)
        expected = values.clip(values.quantile(0.05), values.quantile(0.95))
        assert_series_equal(result[col], expected)
    assert_series_equal(result["store"], sales["store"])
    assert_series_equal(result["promo"], sales["promo"])
    assert result is not sales
    assert_frame_equal(sales, original)


@pytest.mark.functions
def test_winsorize_quantile_matches_manual_clip():
    """Quantile bounds match a manual clip on random data."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.standard_t(df=2, size=500)})

    result = df.winsorize("x", limits=(0.01, 0.99))

    expected = df["x"].clip(df["x"].quantile(0.01), df["x"].quantile(0.99))
    assert_series_equal(result["x"], expected)
    assert result["x"].min() == pytest.approx(df["x"].quantile(0.01))
    assert result["x"].max() == pytest.approx(df["x"].quantile(0.99))


@pytest.mark.functions
def test_winsorize_iqr_uses_tukey_fences():
    """method='iqr' caps at Q1 - 1.5 IQR and Q3 + 1.5 IQR."""
    df = pd.DataFrame({"x": [-500.0, 10.0, 11.0, 12.0, 13.0, 500.0]})
    q1, q3 = df["x"].quantile([0.25, 0.75])
    iqr = q3 - q1

    result = df.winsorize("x", method="iqr")

    expected = pd.Series(
        [q1 - 1.5 * iqr, 10.0, 11.0, 12.0, 13.0, q3 + 1.5 * iqr], name="x"
    )
    assert_series_equal(result["x"], expected)


@pytest.mark.functions
def test_winsorize_iqr_factor_controls_fence_width():
    """iqr_factor=0 caps values to the interquartile range itself."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 5.0]})

    result = df.winsorize("x", method="iqr", iqr_factor=0)

    assert result["x"].tolist() == [2.0, 2.0, 3.0, 4.0, 4.0]


@pytest.mark.functions
def test_winsorize_iqr_ignores_limits():
    """limits has no effect when method='iqr'."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

    default = df.winsorize("x", method="iqr")
    custom = df.winsorize("x", method="iqr", limits=(0.4, 0.6))

    assert_frame_equal(default, custom)


@pytest.mark.functions
def test_winsorize_upper_tail_only():
    """A None lower limit leaves the low values alone."""
    df = pd.DataFrame({"x": [-1000.0, 1.0, 2.0, 3.0, 1000.0]})

    result = df.winsorize("x", limits=(None, 0.75))

    assert result["x"].tolist() == [-1000.0, 1.0, 2.0, 3.0, 3.0]


@pytest.mark.functions
def test_winsorize_lower_tail_only():
    """A None upper limit leaves the high values alone."""
    df = pd.DataFrame({"x": [-1000.0, 1.0, 2.0, 3.0, 1000.0]})

    result = df.winsorize("x", limits=(0.25, None))

    assert result["x"].tolist() == [1.0, 1.0, 2.0, 3.0, 1000.0]


@pytest.mark.functions
def test_winsorize_suffix_keeps_original_column(sales):
    """suffix writes new columns at the end and keeps the originals."""
    result = sales.winsorize("amount", suffix="_capped")

    assert list(result.columns) == [*sales.columns, "amount_capped"]
    assert_series_equal(result["amount"], sales["amount"])
    assert result["amount_capped"].max() < sales["amount"].max()


@pytest.mark.functions
def test_winsorize_suffix_existing_column_raises(sales):
    """suffix may not overwrite a column that already exists."""
    df = sales.assign(amount_capped=0)

    with pytest.raises(ValueError, match="amount_capped"):
        df.winsorize("amount", suffix="_capped")


@pytest.mark.functions
def test_winsorize_by_group(sales):
    """Bounds are computed within each group when by is given."""
    result = sales.winsorize("amount", limits=(0, 0.75), by="store")

    expected = pd.Series(
        [1.0, 2.0, 3.0, 12.25, 100.0, 200.0, 300.0, 1225.0], name="amount"
    )
    assert_series_equal(result["amount"], expected)
    assert_series_equal(result["store"], sales["store"])


@pytest.mark.functions
def test_winsorize_by_group_differs_from_global(sales):
    """Grouping caps each store against its own values, not the whole table."""
    global_result = sales.winsorize("amount", limits=(0, 0.75))
    grouped_result = sales.winsorize("amount", limits=(0, 0.75), by="store")

    # Globally, the 75th percentile is 225, so store b's 300 gets capped.
    assert global_result.loc[6, "amount"] == 225.0
    # Within store b, 300 is below that store's own 75th percentile.
    assert grouped_result.loc[6, "amount"] == 300.0


@pytest.mark.functions
def test_winsorize_by_multiple_keys():
    """Several grouping columns can be used together."""
    df = pd.DataFrame(
        {
            "region": ["n", "n", "n", "n", "n", "n"],
            "store": ["a", "a", "a", "b", "b", "b"],
            "amount": [1.0, 2.0, 30.0, 10.0, 20.0, 300.0],
        }
    )

    result = df.winsorize("amount", limits=(0, 0.5), by=["region", "store"])

    assert result["amount"].tolist() == [1.0, 2.0, 2.0, 10.0, 20.0, 20.0]


@pytest.mark.functions
def test_winsorize_by_missing_key_left_unchanged():
    """Rows whose group key is missing are not capped."""
    df = pd.DataFrame(
        {
            "store": ["a", "a", "a", None],
            "amount": [1.0, 2.0, 30.0, 9999.0],
        }
    )

    result = df.winsorize("amount", limits=(0, 0.5), by="store")

    assert result["amount"].tolist() == [1.0, 2.0, 2.0, 9999.0]


@pytest.mark.functions
def test_winsorize_by_numeric_key_not_winsorized_by_default():
    """A numeric grouping column is excluded from the default selection."""
    df = pd.DataFrame({"year": [2020, 2020, 2021, 2021], "x": [1, 9, 1, 9]})

    result = df.winsorize(limits=(0.5, 1), by="year")

    assert_series_equal(result["year"], df["year"])
    assert result["x"].tolist() == [5.0, 9.0, 5.0, 9.0]


@pytest.mark.functions
def test_winsorize_by_column_in_selection_raises():
    """A column cannot be both a group key and winsorized."""
    df = pd.DataFrame({"year": [2020, 2021], "x": [1, 2]})

    with pytest.raises(ValueError, match="used in `by`"):
        df.winsorize(["year", "x"], by="year")


@pytest.mark.functions
def test_winsorize_by_with_duplicate_index_labels():
    """Grouping works when the index has repeated labels."""
    df = pd.DataFrame(
        {"g": ["a", "a", "a", "b", "b", "b"], "x": [1, 2, 30, 10, 20, 300]},
        index=[0, 0, 0, 1, 1, 1],
    )

    result = df.winsorize("x", limits=(0, 0.5), by="g")

    assert result["x"].tolist() == [1.0, 2.0, 2.0, 10.0, 20.0, 20.0]
    assert result.index.equals(df.index)


@pytest.mark.functions
def test_winsorize_missing_values_are_preserved():
    """NaN values are ignored for the bounds and stay NaN."""
    df = pd.DataFrame({"x": [1.0, np.nan, 2.0, 3.0, 100.0]})

    result = df.winsorize("x", limits=(0, 0.75))

    assert np.isnan(result.loc[1, "x"])
    assert result["x"].dropna().tolist() == [1.0, 2.0, 3.0, 27.25]


@pytest.mark.functions
def test_winsorize_all_missing_column():
    """A column with only missing values is returned unchanged."""
    df = pd.DataFrame({"x": [np.nan, np.nan, np.nan]})

    result = df.winsorize("x")

    assert_frame_equal(result, df)


@pytest.mark.functions
def test_winsorize_empty_dataframe():
    """An empty DataFrame is returned unchanged."""
    df = pd.DataFrame({"x": pd.Series([], dtype=float)})

    result = df.winsorize()

    assert_frame_equal(result, df)


@pytest.mark.functions
def test_winsorize_integer_column_becomes_float():
    """Integer columns are returned as float64."""
    df = pd.DataFrame({"x": [1, 2, 3, 4, 100]})

    result = df.winsorize("x", limits=(0, 0.75))

    assert result["x"].dtype == "float64"
    assert result["x"].tolist() == [1.0, 2.0, 3.0, 4.0, 4.0]


@pytest.mark.functions
def test_winsorize_nullable_integer_column():
    """Nullable Int64 columns become Float64 and keep their missing values."""
    df = pd.DataFrame({"x": pd.array([1, 2, None, 3, 100], dtype="Int64")})

    result = df.winsorize("x", limits=(0, 0.75))

    assert result["x"].dtype == "Float64"
    assert result["x"].isna().tolist() == [False, False, True, False, False]
    assert result["x"].dropna().tolist() == [1.0, 2.0, 3.0, 27.25]


@pytest.mark.functions
def test_winsorize_glob_column_selection():
    """column_names accepts the same selections as select, such as globs."""
    df = pd.DataFrame(
        {
            "price_usd": [1.0, 2.0, 3.0, 100.0],
            "price_eur": [1.0, 2.0, 3.0, 100.0],
            "qty": [1.0, 2.0, 3.0, 100.0],
        }
    )

    result = df.winsorize("price_*", limits=(0, 0.5))

    assert result["price_usd"].max() == 2.5
    assert result["price_eur"].max() == 2.5
    assert_series_equal(result["qty"], df["qty"])


@pytest.mark.functions
def test_winsorize_method_chaining(sales):
    """winsorize can be chained with other pyjanitor and pandas methods."""
    result = (
        sales.winsorize("amount", method="iqr", by="store")
        .groupby("store")["amount"]
        .max()
    )

    assert result["a"] < 40.0
    assert result["b"] < 4000.0


@pytest.mark.functions
@pytest.mark.parametrize("column", ["store", "promo"])
def test_winsorize_non_numeric_column_raises(sales, column):
    """Explicitly selecting text or boolean columns raises TypeError."""
    with pytest.raises(TypeError, match="must be numeric"):
        sales.winsorize(column)


@pytest.mark.functions
def test_winsorize_missing_column_raises(sales):
    """Selecting a column that does not exist raises KeyError."""
    with pytest.raises(KeyError):
        sales.winsorize("not_a_column")


@pytest.mark.functions
def test_winsorize_invalid_method_raises(sales):
    """Unknown methods raise ValueError; non-strings raise TypeError."""
    with pytest.raises(ValueError, match="`method` must be"):
        sales.winsorize(method="zscore")
    with pytest.raises(TypeError):
        sales.winsorize(method=1)


@pytest.mark.functions
@pytest.mark.parametrize(
    "limits",
    [
        (0.9, 0.1),
        (0.5, 0.5),
        (-0.1, 0.9),
        (0.1, 1.5),
        (None, None),
        (0.1,),
        0.5,
        ("low", 0.9),
        (True, 0.9),
    ],
)
def test_winsorize_invalid_limits_raise(sales, limits):
    """Invalid limits raise ValueError."""
    with pytest.raises(ValueError):
        sales.winsorize("amount", limits=limits)


@pytest.mark.functions
def test_winsorize_invalid_iqr_factor_raises(sales):
    """iqr_factor must be a non-negative number."""
    with pytest.raises(ValueError, match="non-negative"):
        sales.winsorize("amount", method="iqr", iqr_factor=-1)
    with pytest.raises(TypeError, match="iqr_factor"):
        sales.winsorize("amount", method="iqr", iqr_factor="1.5")


@pytest.mark.functions
def test_winsorize_invalid_suffix_raises(sales):
    """suffix must be a non-empty string."""
    with pytest.raises(ValueError, match="non-empty"):
        sales.winsorize("amount", suffix="")
    with pytest.raises(TypeError):
        sales.winsorize("amount", suffix=1)


@pytest.mark.functions
@settings(deadline=None, max_examples=10)
@given(
    values=st.lists(
        st.floats(min_value=-1e6, max_value=1e6, allow_nan=False),
        min_size=1,
        max_size=50,
    ),
    lower=st.floats(min_value=0, max_value=0.49),
    upper=st.floats(min_value=0.51, max_value=1),
)
def test_winsorize_properties(values, lower, upper):
    """Results stay within the bounds, keep order and leave inner values alone."""
    df = pd.DataFrame({"x": values})
    low, high = df["x"].quantile(lower), df["x"].quantile(upper)

    result = df.winsorize("x", limits=(lower, upper))["x"]

    assert (result >= low).all()
    assert (result <= high).all()
    inside = df["x"].between(low, high)
    assert_series_equal(result[inside], df["x"][inside])
    order = np.argsort(df["x"].to_numpy(), kind="stable")
    assert (np.diff(result.to_numpy()[order]) >= 0).all()
