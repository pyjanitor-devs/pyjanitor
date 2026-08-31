import numpy as np

from janitor.functions._conditional_join import _range_indices


def _fixture():
    right_index = np.array([24, 58, 2, 13, 91, 7, 40, 5] * 8, dtype=np.int64)
    starts = np.zeros(64, dtype=np.int64)
    ends = np.full(64, 16, dtype=np.int64)
    return right_index, starts, ends


def test_range_rmq_dispatches_first_to_rust(monkeypatch):
    right_index, starts, ends = _fixture()
    called = {}

    def fake_kernel(**kwargs):
        called.update(kwargs)
        return np.full(starts.size, 2, dtype=np.int64)

    monkeypatch.setattr(
        _range_indices.janitor_rs,
        "index_starts_and_ends_keep_first_direct",
        fake_kernel,
        raising=False,
    )

    result = _range_indices._range_rmq(right_index, starts, ends, "first")

    assert np.array_equal(result, np.full(starts.size, 2, dtype=np.int64))
    assert called["index"] is right_index
    assert np.array_equal(called["starts"], starts)
    assert np.array_equal(called["ends"], ends)


def test_range_rmq_dispatches_last_to_rust(monkeypatch):
    right_index, starts, ends = _fixture()

    monkeypatch.setattr(
        _range_indices.janitor_rs,
        "index_starts_and_ends_keep_last_direct",
        lambda **kwargs: np.full(starts.size, 91, dtype=np.int64),
        raising=False,
    )

    result = _range_indices._range_rmq(right_index, starts, ends, "last")

    assert np.array_equal(result, np.full(starts.size, 91, dtype=np.int64))


def test_range_rmq_falls_back_for_narrow_workloads(monkeypatch):
    right_index = np.arange(64, dtype=np.int64)
    starts = np.arange(64, dtype=np.int64)
    ends = starts + 1

    def unexpected_kernel(**kwargs):
        raise AssertionError("narrow ranges must retain the existing path")

    monkeypatch.setattr(
        _range_indices.janitor_rs,
        "index_starts_and_ends_keep_first_direct",
        unexpected_kernel,
        raising=False,
    )

    assert _range_indices._range_rmq(right_index, starts, ends, "first") is None


def test_range_rmq_falls_back_for_non_int64_index(monkeypatch):
    right_index = np.arange(64, dtype=np.int32)
    starts = np.zeros(64, dtype=np.int64)
    ends = np.full(64, 32, dtype=np.int64)

    def unexpected_kernel(**kwargs):
        raise AssertionError("unsupported index dtypes must retain the existing path")

    monkeypatch.setattr(
        _range_indices.janitor_rs,
        "index_starts_and_ends_keep_first_direct",
        unexpected_kernel,
        raising=False,
    )

    assert _range_indices._range_rmq(right_index, starts, ends, "first") is None
