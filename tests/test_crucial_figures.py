from pathlib import Path

import pandas as pd

from evaluation.crucial_figures import (
    add_ratio,
    benchmark_brotli,
    benchmark_fsst,
    deserialize_token_ids,
    parse_predictive_payload,
    plot_surprisal_scatter,
    serialize_predictive_payload,
    serialize_token_ids,
    validate_bar_results,
)


def test_token_id_stream_round_trip():
    token_ids = [0, 151642, 17, 999, 42]
    blob = serialize_token_ids(token_ids, vocab_size=151643, raw_size_bytes=123)
    decoded = deserialize_token_ids(blob)
    assert decoded["token_ids"] == token_ids
    assert decoded["width"] == 18
    assert decoded["raw_size_bytes"] == 123


def test_predictive_stream_round_trip():
    bits = [1, 0, 1, 1, 0, 0, 1]
    blob = serialize_predictive_payload(
        bits,
        seed_token=785,
        model_code=2,
        token_count=8,
        raw_size_bytes=37,
    )
    decoded = parse_predictive_payload(blob)
    assert decoded == {
        "bits": bits,
        "seed_token": 785,
        "model_code": 2,
        "token_count": 8,
        "raw_size_bytes": 37,
    }


def test_brotli_is_measured_as_complete_stream():
    data = (b"the quick brown fox " * 20) + b"end"
    result = add_ratio(benchmark_brotli(data))
    assert result["round_trip"] is True
    assert result["serialized_size_bytes"] < result["raw_size_bytes"]
    assert result["relative_size_percent"] < 100


def test_fsst_round_trip_when_executable_is_configured():
    executable = Path("/tmp/compression-economics-fsst/build/fsst")
    if not executable.exists():
        return
    data = (b"database compression with a static symbol table " * 100) + b"end"
    result = benchmark_fsst(data, executable=executable)
    assert result["round_trip"] is True
    assert result["serialized_size_bytes"] < result["raw_size_bytes"]


def test_bar_validation_keeps_only_measured_rows():
    frame = pd.DataFrame(
        [
            {
                "dataset": "text8",
                "codec": "Brotli q11",
                "raw_size_bytes": 100,
                "serialized_size_bytes": 50,
                "round_trip": True,
                "status": "measured",
            },
            {
                "dataset": "text8",
                "codec": "Qwen + AC",
                "raw_size_bytes": 100,
                "serialized_size_bytes": pd.NA,
                "round_trip": pd.NA,
                "status": "missing",
            },
        ]
    )
    measured = validate_bar_results(frame, ["text8"])
    assert measured["codec"].tolist() == ["Brotli q11"]
    assert measured["relative_size_percent"].tolist() == [50.0]


def test_surprisal_scatter_connects_means_and_fades_individual_runs():
    frame = pd.DataFrame(
        [
            {"block": 0, "model": "Qwen2.5-0.5B", "surprisal_bits_per_token": 2, "relative_size_percent": 20},
            {"block": 1, "model": "Qwen2.5-0.5B", "surprisal_bits_per_token": 4, "relative_size_percent": 30},
            {"block": 0, "model": "Anti-Qwen", "surprisal_bits_per_token": 40, "relative_size_percent": 45},
            {"block": 1, "model": "Anti-Qwen", "surprisal_bits_per_token": 50, "relative_size_percent": 55},
        ]
    )

    fig, ax = plot_surprisal_scatter(frame)
    try:
        mean_connector = ax.lines[0]
        assert mean_connector.get_xdata().tolist() == [3.0, 45.0]
        assert mean_connector.get_ydata().tolist() == [25.0, 50.0]
        assert len(ax.lines) == 2  # mean connector plus raw UTF-8 reference

        run_markers = [
            collection for collection in ax.collections
            if collection.get_alpha() == 0.30
        ]
        assert len(run_markers) == 2
        assert [len(collection.get_offsets()) for collection in run_markers] == [2, 2]

        mean_markers = [
            collection for collection in ax.collections
            if collection.get_label().endswith(" mean")
        ]
        assert len(mean_markers) == 2
        assert all(len(collection.get_offsets()) == 1 for collection in mean_markers)
    finally:
        fig.clear()
