from __future__ import annotations

import numpy as np
import pytest

from research.papers.coding_survey.experiments.benchmark_inference_coder_isolation import (
    _balanced_orders,
    _load_trace,
    _save_trace,
    _trace_sha256,
    _validate_trace,
)


def _trace():
    return {
        "lows": np.array([[0, 2], [1, 0]], dtype=np.int64),
        "highs": np.array([[1, 4], [3, 1]], dtype=np.int64),
        "totals": np.array([[4, 4], [4, 1]], dtype=np.int64),
        "counts": np.array([2, 1], dtype=np.int64),
    }


def test_frozen_trace_round_trip_and_hash(tmp_path):
    trace = _trace()
    path = tmp_path / "trace.npz"
    _save_trace(path, trace)

    loaded = _load_trace(path)

    assert all(np.array_equal(loaded[name], trace[name]) for name in trace)
    assert _trace_sha256(loaded) == _trace_sha256(trace)


def test_frozen_trace_rejects_invalid_active_interval():
    trace = _trace()
    trace["highs"][1, 0] = trace["lows"][1, 0]

    with pytest.raises(ValueError, match="invalid target interval"):
        _validate_trace(trace)


def test_balanced_orders_rotate_and_reverse():
    assert _balanced_orders(["a", "b"], 4) == [
        ["a", "b"],
        ["b", "a"],
        ["b", "a"],
        ["a", "b"],
    ]
