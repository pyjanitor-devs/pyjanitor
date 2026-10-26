"""Regression cases for the homogeneous NumPy .value fast paths (#1821)."""

import importlib

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from janitor import pivot_longer, pivot_longer_spec

pivot_module = importlib.import_module("janitor.functions.pivot")


@pytest.mark.parametrize("with_other", [False, True])
@pytest.mark.parametrize("arrangement", ["contiguous", "interleaved", "uneven"])
@pytest.mark.parametrize("sort_by_appearance", [False, True])
@pytest.mark.parametrize(
    "dtype", ["float64", "float32", "int64", "uint64", "bool", "object", "Int64"]
)
def test_fast_value_groups_equal_previous(
    monkeypatch, with_other, arrangement, sort_by_appearance, dtype
):
    """Fast positional extraction must preserve the old output exactly."""
    groups = ["a", "a", "b", "b", "c", "c"]
    parts = [0, 1, 0, 1, 0, 1]
    if arrangement == "interleaved":
        order = [0, 2, 4, 1, 3, 5]
    elif arrangement == "uneven":
        order = [0, 1, 2, 3, 4]
    else:
        order = list(range(6))
    names = [f"col_{j}" for j in order]
    values = {}
    for j, name in zip(order, names):
        data = [j + 1, j + 2, None if j % 3 == 0 else j + 3]
        if dtype in {"int64", "uint64", "bool"}:
            data[-1] = j + 3
        if dtype == "bool":
            data = [bool(value % 2) for value in data]
        if dtype == "object":
            values[name] = pd.Series(data, dtype="object")
        elif dtype == "Int64":
            values[name] = pd.array(data, dtype="Int64")
        else:
            values[name] = pd.Series(data, dtype=dtype)
    frame = pd.DataFrame(values)
    spec = pd.DataFrame({".name": names, ".value": [groups[j] for j in order]})
    if with_other:
        spec["part"] = [parts[j] for j in order]
    base = frame.copy(deep=True)
    options = {"sort_by_appearance": sort_by_appearance, "ignore_index": True}
    expected_factory = pivot_module._fast_numpy_value_matrix
    with monkeypatch.context() as patched:
        patched.setattr(pivot_module, "_fast_numpy_value_matrix", lambda df: None)
        expected = pivot_longer_spec(frame.copy(deep=True), spec.copy(), **options)
    assert pivot_module._fast_numpy_value_matrix is expected_factory
    fast_path_calls = []
    eligible_matrices = []
    real_fast_ravel = pivot_module._fast_ravel_group

    def record_fast_matrix(dataframe):
        matrix = expected_factory(dataframe)
        eligible_matrices.append(matrix is not None)
        return matrix

    def record_fast_ravel(*args, **kwargs):
        fast_path_calls.append(1)
        return real_fast_ravel(*args, **kwargs)

    with monkeypatch.context() as patched:
        patched.setattr(pivot_module, "_fast_numpy_value_matrix", record_fast_matrix)
        patched.setattr(pivot_module, "_fast_ravel_group", record_fast_ravel)
        actual = pivot_longer_spec(frame, spec.copy(), **options)

    assert bool(fast_path_calls) == any(eligible_matrices)
    if dtype == "Int64":
        assert not fast_path_calls, "Extension dtypes must use the legacy path."
    assert_frame_equal(actual, expected)
    # Mutate the *actual returned frame*: copy(deep=False) would activate
    # pandas Copy-on-Write and could hide a raw-buffer aliasing regression.
    writes_checked = 0
    for col in actual.columns:
        if col in (".name", ".value", "part") or actual.empty:
            continue
        column_index = actual.columns.get_loc(col)
        returned = pivot_longer_spec(frame, spec.copy(), **options)
        # Assign a value compatible with each dtype. Incompatible assignments
        # are skipped, but the float/int/bool paths must remain exercised.
        old_value = returned.iloc[0, column_index]
        value = (not bool(old_value)) if dtype == "bool" else 999
        try:
            returned.iloc[0, column_index] = value
        except (TypeError, ValueError):
            continue
        assert_frame_equal(frame, base)
        writes_checked += 1
    assert writes_checked > 0, "No output mutation was verified."


@pytest.mark.parametrize("with_other", [False, True])
@pytest.mark.parametrize("sort_by_appearance", [False, True])
def test_homogeneous_buffer_does_not_alias_source(with_other, sort_by_appearance):
    """Mutation of fast-path float results must not change source values."""
    cols = ["a_1", "a_2", "b_1", "b_2"]
    frame = pd.DataFrame(np.arange(16, dtype="float64").reshape((4, 4)), columns=cols)
    spec = pd.DataFrame({".name": cols, ".value": ["a", "a", "b", "b"]})
    if with_other:
        spec["part"] = [1, 2, 1, 2]
    before = frame.copy(deep=True)
    result = pivot_longer_spec(frame, spec, sort_by_appearance=sort_by_appearance)
    for col in ["a", "b"]:
        result.loc[result.index[0], col] = -500.0
    assert_frame_equal(frame, before)


@pytest.mark.parametrize("sort_by_appearance", [False, True])
def test_names_transform_preserves_output_and_input(monkeypatch, sort_by_appearance):
    """The public API names_transform path must match the pre-fast-path result."""
    names = [f"v{group}_{part}" for group in range(3) for part in range(2)]
    source = pd.DataFrame(np.arange(24, dtype="float64").reshape(4, 6), columns=names)
    original = source.copy(deep=True)
    options = {
        "names_to": (".value", "part"),
        "names_pattern": r"(v[0-9]+)_([0-9]+)",
        "names_transform": {"part": "int64"},
        "sort_by_appearance": sort_by_appearance,
    }
    with monkeypatch.context() as patch:
        patch.setattr(pivot_module, "_fast_numpy_value_matrix", lambda df: None)
        expected = pivot_longer(source.copy(deep=True), **options)
    result = pivot_longer(source, **options)
    assert_frame_equal(result, expected)
    assert result["part"].dtype == np.dtype("int64")
    result.loc[result.index[0], "v0"] = -500.0
    assert_frame_equal(source, original)
