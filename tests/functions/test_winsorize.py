"""Tests for the winsorize function."""

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal, assert_series_equal

import janitor  # noqa: F401

finite_floats = st.floats(
    min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False
)
# Mostly finite values, with some missing and infinite ones mixed in.
values_with_gaps = st.lists(
    st.one_of(
        finite_floats,
        finite_floats,
        finite_floats,
        st.sampled_from([np.nan, np.inf, -np.inf]),
    ),
    min_size=1,
    max_size=40,
)
probabilities = st.floats(min_value=0, max_value=1)
valid_limits = st.one_of(
    st.tuples(probabilities, probabilities).filter(lambda pair: pair[0] < pair[1]),
    st.tuples(st.none(), probabilities),
    st.tuples(probabilities, st.none()),
)
thresholds = st.one_of(st.just(0.0), st.floats(min_value=0, max_value=10))


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
def test_winsorize_iqr_ignores_limits():
    """limits has no effect when method='iqr'."""
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0, 100.0]})

    default = df.winsorize("x", method="iqr")
    custom = df.winsorize("x", method="iqr", limits=(0.4, 0.6))

    assert_frame_equal(default, custom)


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
def test_winsorize_caps_infinite_values():
    """inf is left out of the bounds and then capped, for both methods."""
    df = pd.DataFrame({"x": [-np.inf, 1.0, 2.0, 3.0, 4.0, np.inf]})

    by_quantile = df.winsorize("x", limits=(0, 0.75))
    by_iqr = df.winsorize("x", method="iqr")

    # The finite values are 1, 2, 3 and 4: Q1 = 1.75, Q3 = 3.25, IQR = 1.5.
    assert by_quantile["x"].tolist() == [1.0, 1.0, 2.0, 3.0, 3.25, 3.25]
    assert by_iqr["x"].tolist() == [-0.5, 1.0, 2.0, 3.0, 4.0, 5.5]


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
def test_winsorize_complex_column_raises():
    """Complex columns are rejected, and skipped by the default selection."""
    df = pd.DataFrame({"z": [1 + 1j, 2 + 0j, 3 - 1j], "x": [1.0, 2.0, 30.0]})

    with pytest.raises(TypeError, match="not boolean or complex"):
        df.winsorize("z")
    result = df.winsorize(limits=(0, 0.5))
    assert_series_equal(result["z"], df["z"])
    assert result["x"].tolist() == [1.0, 2.0, 2.0]


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
@pytest.mark.parametrize("threshold", [-1, np.inf, np.nan])
def test_winsorize_invalid_threshold_raises(sales, threshold):
    """threshold must be a finite, non-negative number."""
    with pytest.raises(ValueError, match="finite, non-negative"):
        sales.winsorize("amount", method="iqr", threshold=threshold)


@pytest.mark.functions
@pytest.mark.parametrize("threshold", ["1.5", True, None])
def test_winsorize_non_numeric_threshold_raises(sales, threshold):
    """threshold must be a real number."""
    with pytest.raises(TypeError, match="threshold"):
        sales.winsorize("amount", method="iqr", threshold=threshold)


@pytest.mark.functions
def test_winsorize_invalid_suffix_raises(sales):
    """suffix must be a non-empty string."""
    with pytest.raises(ValueError, match="non-empty"):
        sales.winsorize("amount", suffix="")
    with pytest.raises(TypeError):
        sales.winsorize("amount", suffix=1)


@st.composite
def frames_with_missing_values(draw):
    """Frames with missing values, a repeated index and an optional group."""
    dtype = draw(st.sampled_from(["float64", "Float64", "Int64"]))
    size = draw(st.integers(min_value=1, max_value=30))
    numbers = st.integers(-1000, 1000) if dtype == "Int64" else finite_floats
    values = draw(st.lists(st.none() | numbers, min_size=size, max_size=size))
    if dtype == "float64":
        x = np.array([np.nan if v is None else v for v in values], dtype=float)
    else:
        x = pd.array(values, dtype=dtype)
    groups = draw(
        st.lists(st.sampled_from(["a", "b", None]), min_size=size, max_size=size)
    )
    index = draw(st.lists(st.integers(0, 3), min_size=size, max_size=size))
    return pd.DataFrame({"g": groups, "x": x}, index=index)


@st.composite
def grouped_frames(draw):
    """Frames with a value column and a group key that may be missing."""
    values = draw(values_with_gaps)
    groups = draw(
        st.lists(
            st.sampled_from(["a", "b", "c", None]),
            min_size=len(values),
            max_size=len(values),
        )
    )
    return pd.DataFrame({"g": groups, "x": values})


@pytest.mark.functions
@settings(deadline=None, max_examples=100)
@given(values=values_with_gaps, limits=valid_limits)
def test_winsorize_quantile_bounds_property(values, limits):
    """Output stays within the input's quantile bounds and keeps its order."""
    x = pd.Series(values, name="x")
    finite = x[np.isfinite(x)]

    result = pd.DataFrame({"x": x}).winsorize("x", limits=limits)["x"]

    if finite.empty:
        # Without finite values there are no bounds, so nothing changes.
        assert_series_equal(result, x)
        return
    lower, upper = limits
    low = -np.inf if lower is None else finite.quantile(lower)
    high = np.inf if upper is None else finite.quantile(upper)
    capped = result[np.isfinite(result)]
    assert (capped >= low).all()
    assert (capped <= high).all()
    # Infinite values are capped on every side that has a bound.
    if lower is not None:
        assert not (result == -np.inf).any()
    if upper is not None:
        assert not (result == np.inf).any()
    inside = x.between(low, high)
    assert_series_equal(result[inside], x[inside])
    order = np.argsort(x.to_numpy(), kind="stable")
    ranked = result.to_numpy()[order]
    ranked = ranked[~np.isnan(ranked)]
    assert (ranked[1:] >= ranked[:-1]).all()


@pytest.mark.functions
@settings(deadline=None, max_examples=100)
@given(
    df=frames_with_missing_values(),
    method=st.sampled_from(["quantile", "iqr"]),
    grouped=st.booleans(),
)
def test_winsorize_keeps_missing_values_and_index(df, method, grouped):
    """Missing values stay missing, and the index is unchanged."""
    result = df.winsorize("x", method=method, by="g" if grouped else None)

    assert result.index.equals(df.index)
    assert result["x"].isna().tolist() == df["x"].isna().tolist()
    assert_series_equal(result["g"], df["g"])


@pytest.mark.functions
@settings(deadline=None, max_examples=100)
@given(values=values_with_gaps, threshold=thresholds)
def test_winsorize_iqr_matches_tukey_fences(values, threshold):
    """method="iqr" clips at Tukey's fences computed from the finite values."""
    x = pd.Series(values, name="x")
    q1, q3 = x[np.isfinite(x)].quantile([0.25, 0.75])
    spread = threshold * (q3 - q1)
    low, high = q1 - spread, q3 + spread

    result = pd.DataFrame({"x": x}).winsorize("x", method="iqr", threshold=threshold)[
        "x"
    ]

    assert_series_equal(result, x.clip(low, high))
    inside = x.between(low, high)
    assert_series_equal(result[inside], x[inside])


@pytest.mark.functions
@settings(deadline=None, max_examples=100)
@given(
    df=grouped_frames(),
    method=st.sampled_from(["quantile", "iqr"]),
    limits=valid_limits,
    threshold=thresholds,
)
def test_winsorize_by_group_matches_each_group(df, method, limits, threshold):
    """Grouped results equal winsorizing each group on its own."""
    options = {"method": method, "limits": limits, "threshold": threshold}

    result = df.winsorize("x", by="g", **options)

    expected = df["x"].copy()
    for key in df["g"].dropna().unique():
        rows = (df["g"] == key).to_numpy()
        expected[rows] = df[rows].winsorize("x", **options)["x"].to_numpy()
    assert_series_equal(result["x"], expected)
    missing = df["g"].isna()
    assert_series_equal(result["x"][missing], df["x"][missing])
