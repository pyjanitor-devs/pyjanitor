import numpy as np
import pandas as pd
from janitor.functions._conditional_join._equi_join_rust import _get_indices

from janitor.functions._conditional_join import _equi_join_rust


def test_unique_equi_join_returns_direct_indices():
    left = pd.DataFrame({"key": ["b", "missing", "a"]})
    right = pd.DataFrame({"key": ["a", "b"]})

    result = _get_indices(left, right, [("key", "key", "==")], "all")

    np.testing.assert_array_equal(result["left_index"], [0, 2])
    np.testing.assert_array_equal(result["right_index"], [1, 0])


def test_unique_multi_column_equi_join_returns_direct_indices():
    left = pd.DataFrame({"first": ["a", "a"], "second": [2, 1]})
    right = pd.DataFrame(
        {"first": ["a", "b", "a"], "second": [1, 1, 2]},
        index=[10, 11, 12],
    )

    result = _get_indices(
        left,
        right,
        [
            ("first", "first", "=="),
            ("second", "second", "=="),
        ],
        "all",
    )

    np.testing.assert_array_equal(result["left_index"], [0, 1])
    np.testing.assert_array_equal(result["right_index"], [10, 12])


def test_equi_join_filters_null_rows_before_building_keys():
    left = pd.DataFrame({"key": [1, np.nan, 2]}, index=[10, 11, 12])
    right = pd.DataFrame({"key": [np.nan, 2, 1]}, index=[20, 21, 22])

    result = _get_indices(left, right, [("key", "key", "==")], "all")

    np.testing.assert_array_equal(result["left_index"], [10, 12])
    np.testing.assert_array_equal(result["right_index"], [22, 21])


def test_duplicate_equi_join_returns_rust_building_blocks(monkeypatch):
    left = pd.DataFrame({"key": ["b", "a", "missing"]})
    right = pd.DataFrame({"key": ["a", "b", "a"]})
    expected = {
        "left_index": np.array([0, 1, 2]),
        "right_index": np.array([0, 1, 2]),
        "left_indexer": np.array([1, 0, -1]),
        "offsets": np.array([0, 0, 1, 1, 3]),
        "positions": np.array([1, 0, 2]),
    }
    captured = {}

    def fake_building_blocks(
        left_index,
        right_index,
        left_indexer,
        right_codes,
    ):
        captured["left_index"] = left_index
        captured["right_index"] = right_index
        captured["left_indexer"] = left_indexer
        captured["right_codes"] = right_codes
        return expected

    monkeypatch.setattr(
        _equi_join_rust.janitor_rs,
        "equi_join_building_blocks",
        fake_building_blocks,
    )

    result = _get_indices(
        left,
        right,
        [("key", "key", "==")],
        "all",
        return_building_blocks=True,
    )

    assert result is expected
    np.testing.assert_array_equal(captured["left_index"], [0, 1, 2])
    np.testing.assert_array_equal(captured["right_index"], [0, 1, 2])
    np.testing.assert_array_equal(captured["left_indexer"], [1, 0, -1])
    np.testing.assert_array_equal(captured["right_codes"], [0, 1, 0])


def test_duplicate_equi_join_materialized_path_forwards_keep(monkeypatch):
    left = pd.DataFrame({"key": ["b", "a"]})
    right = pd.DataFrame({"key": ["a", "b", "a"]})
    captured = {}

    def fake_equi_join_indices(
        left_index,
        right_index,
        left_indexer,
        right_codes,
        keep,
    ):
        captured["left_index"] = left_index
        captured["right_index"] = right_index
        captured["left_indexer"] = left_indexer
        captured["right_codes"] = right_codes
        captured["keep"] = keep
        return {"left_index": np.array([0]), "right_index": np.array([1])}

    monkeypatch.setattr(
        _equi_join_rust.janitor_rs,
        "equi_join_indices",
        fake_equi_join_indices,
    )

    result = _get_indices(
        left,
        right,
        [("key", "key", "==")],
        "last",
    )

    np.testing.assert_array_equal(result["left_index"], [0])
    np.testing.assert_array_equal(result["right_index"], [1])
    np.testing.assert_array_equal(captured["left_indexer"], [1, 0])
    np.testing.assert_array_equal(captured["right_codes"], [0, 1, 0])
    assert captured["keep"] == "last"


def test_duplicate_equi_join_returns_empty_indices_when_no_matches(monkeypatch):
    left = pd.DataFrame({"key": ["missing"]})
    right = pd.DataFrame({"key": ["a", "b", "a"]})

    def fail_if_called(*args):
        raise AssertionError("Rust should not be called when there are no equi matches")

    monkeypatch.setattr(
        _equi_join_rust.janitor_rs,
        "equi_join_building_blocks",
        fail_if_called,
    )

    result = _get_indices(
        left,
        right,
        [("key", "key", "==")],
        "all",
        return_building_blocks=True,
    )

    assert set(result) == {"left_index", "right_index"}
    assert result["left_index"].size == 0
    assert result["right_index"].size == 0


def test_shared_range_layout_forwards_two_ranges(monkeypatch):
    left = pd.DataFrame({"key": [1], "lower": [2], "upper": [8]})
    right = pd.DataFrame(
        {"key": [1, 1], "lower": [3, 7], "upper": [7, 9]},
        index=[10, 11],
    )
    captured = {}

    def fake_filtered_indices(*args):
        captured["args"] = args
        return {"left_index": np.array([0]), "right_index": np.array([10])}

    monkeypatch.setattr(
        _equi_join_rust.janitor_rs,
        "equi_join_filtered_indices",
        fake_filtered_indices,
    )

    result = _get_indices(
        left,
        right,
        [
            ("key", "key", "=="),
            ("lower", "lower", "<="),
            ("upper", "upper", ">="),
        ],
        "all",
    )

    np.testing.assert_array_equal(result["left_index"], [0])
    assert len(captured["args"][4]) == 2
    assert captured["args"][5] == ()


def test_range_and_residual_predicates_are_forwarded_to_rust(monkeypatch):
    left = pd.DataFrame({"key": [1], "first": [5], "second": [5]})
    right = pd.DataFrame(
        {"key": [1, 1], "first": [8, 3], "second": [1, 9]},
        index=[10, 11],
    )
    captured = {}

    def fake_filtered_indices(
        left_index,
        right_index,
        left_indexer,
        right_codes,
        range_predicates,
        residual_predicates,
        keep,
    ):
        captured["left_index"] = left_index
        captured["right_index"] = right_index
        captured["left_indexer"] = left_indexer
        captured["right_codes"] = right_codes
        captured["range_predicates"] = range_predicates
        captured["residual_predicates"] = residual_predicates
        captured["keep"] = keep
        return {"left_index": np.array([0]), "right_index": np.array([10])}

    monkeypatch.setattr(
        _equi_join_rust.janitor_rs,
        "equi_join_filtered_indices",
        fake_filtered_indices,
    )

    result = _get_indices(
        left,
        right,
        [
            ("key", "key", "=="),
            ("first", "first", "<="),
            ("second", "second", "<="),
        ],
        "all",
    )

    np.testing.assert_array_equal(result["left_index"], [0])
    np.testing.assert_array_equal(result["right_index"], [10])
    assert len(captured["range_predicates"]) == 1
    assert len(captured["residual_predicates"]) == 1
    assert len(captured["range_predicates"][0]) == 3
    assert len(captured["residual_predicates"][0]) == 3
    assert captured["keep"] == "all"
    assert captured["right_index"].tolist() == [11, 10]


def test_duplicate_equi_range_path_calls_rust_with_real_lists():
    left = pd.DataFrame({"key": [1], "value": [5]})
    right = pd.DataFrame(
        {"key": [1, 1, 1], "value": [3, 5, 7]},
        index=[10, 11, 12],
    )

    result = _get_indices(
        left,
        right,
        [
            ("key", "key", "=="),
            ("value", "value", "<="),
        ],
        "all",
    )

    np.testing.assert_array_equal(result["left_index"], [0, 0])
    np.testing.assert_array_equal(result["right_index"], [11, 12])


def test_not_equal_residual_uses_rust_tuple_representation(monkeypatch):
    left = pd.DataFrame({"key": [1], "value": [5]})
    right = pd.DataFrame(
        {"key": [1, 1], "value": [4, 6]},
        index=[10, 11],
    )
    captured = {}

    def fake_filtered_indices(*args):
        captured["residual_predicates"] = args[5]
        return {"left_index": np.array([0]), "right_index": np.array([10])}

    monkeypatch.setattr(
        _equi_join_rust.janitor_rs,
        "equi_join_filtered_indices",
        fake_filtered_indices,
    )

    _get_indices(
        left,
        right,
        [
            ("key", "key", "=="),
            ("value", "value", "!="),
        ],
        "all",
    )

    residual = captured["residual_predicates"]
    assert len(residual) == 1
    assert len(residual[0]) in {3, 6}
    assert residual[0][-1] == "!="
