import math

import numpy as np
import pytest

from src.coding.trace_benchmark import (
    CODERS,
    ProbabilityTrace,
    benchmark_coder,
    decode_bitpacked_ranks,
    decode_huffman_ranks,
    encode_bitpacked_ranks,
    encode_huffman_ranks,
    load_probability_trace,
    save_probability_trace,
    synthetic_probability_trace,
)


@pytest.fixture(scope="module")
def trace():
    return synthetic_probability_trace(80, 8, seed=12)


def test_probability_trace_roundtrips_without_dtype_or_value_changes(tmp_path, trace):
    path = tmp_path / "trace.npz"
    save_probability_trace(path, trace)
    restored = load_probability_trace(path)

    assert restored.sha256 == trace.sha256
    assert restored.probabilities.dtype == np.float64
    assert np.array_equal(restored.probabilities, trace.probabilities)
    assert np.array_equal(restored.symbols, trace.symbols)


@pytest.mark.parametrize("coder", CODERS)
def test_every_coder_has_a_matched_end_to_end_decode(trace, coder):
    result = benchmark_coder(
        trace, coder, ans_block_size=17, ans_lanes=3,
        perturbation_scales=[1e-9], seed=4,
    )

    assert result["exact_roundtrip_valid"]
    assert result["archive_bytes"] > 0
    assert result["payload_bits"] > 0
    assert result["encode_seconds"] > 0
    assert result["decode_seconds"] > 0
    assert result["ideal_cross_entropy_bits"] > 0
    assert result["quantized_distribution_cross_entropy_bits"] > 0


def test_probability_and_rank_semantics_are_not_conflated(trace):
    ac = benchmark_coder(trace, "AC")
    rank = benchmark_coder(trace, "BITPACKED_RANK")

    assert ac["codes_original_probability_distribution"]
    assert not rank["codes_original_probability_distribution"]


def test_huffman_archive_charges_the_serialized_codebook(trace):
    stream = encode_huffman_ranks(trace)
    decoded = decode_huffman_ranks(stream, trace.probabilities)

    assert np.array_equal(decoded, trace.symbols)
    assert stream.codebook_bytes > 0
    assert stream.archive_bytes == (
        stream.framing_bytes + stream.codebook_bytes + math.ceil(stream.payload_bits / 8)
    )


def test_bitpacked_archive_has_end_to_end_decoder(trace):
    stream = encode_bitpacked_ranks(trace)

    assert np.array_equal(decode_bitpacked_ranks(stream, trace.probabilities), trace.symbols)
    assert stream.archive_bytes == stream.framing_bytes + math.ceil(stream.payload_bits / 8)


@pytest.mark.parametrize("lanes", [1, 2, 4])
def test_ans_lane_setting_is_executable_and_recorded(trace, lanes):
    result = benchmark_coder(trace, "ANS", ans_block_size=13, ans_lanes=lanes)

    assert result["parameters"]["block_symbols"] == 13
    assert result["parameters"]["lanes"] == lanes
    assert result["exact_roundtrip_valid"]
    assert result["framing_bytes"] > 0


@pytest.mark.parametrize("streams", [1, 4, 7])
def test_multistream_ac_charges_each_stream_and_records_partition(trace, streams):
    result = benchmark_coder(trace, "AC_MULTISTREAM", ac_streams=streams)

    assert result["parameters"]["streams"] == streams
    assert result["exact_roundtrip_valid"]
    assert result["archive_bytes"] == result["framing_bytes"] + result["payload_bytes"]
    assert result["payload_bytes"] * 8 == result["payload_bits"] + result["padding_bits"]
    if streams == 1:
        assert result["payload_bits"] == benchmark_coder(trace, "AC")["payload_bits"]


def test_pmatic_runs_safe_numerical_reproducibility_scenario(trace):
    result = benchmark_coder(trace, "PMATIC", pmatic_delta=0.01)
    safe = next(
        item for item in result["numerical_reproducibility"]
        if item["name"].startswith("pmatic_safe_delta")
    )

    assert safe["roundtrip_valid"]
    assert result["helper_symbols"] > 0


def test_invalid_trace_is_rejected():
    with pytest.raises(ValueError, match="sum to one"):
        ProbabilityTrace(np.asarray([[0.2, 0.2]]), np.asarray([0]))


def test_parallel_benchmark_records_backend_and_roundtrip(trace):
    pytest.importorskip("numba")
    python = benchmark_coder(trace, "AC_MULTISTREAM", ac_streams=4)
    parallel = benchmark_coder(
        trace, "AC_MULTISTREAM", ac_streams=4,
        ac_backend="numba_parallel", ac_threads=2,
    )
    assert parallel["exact_roundtrip_valid"]
    assert parallel["payload_bits"] == python["payload_bits"]
    assert parallel["archive_bytes"] == python["archive_bytes"]
    assert parallel["parameters"]["ac_threads"] == 2
    assert parallel["backend"] == "numba_parallel"
    assert parallel["range_encode_seconds"] > 0

def test_benchmark_can_time_without_memory_or_perturbation_checks(trace):
    result = benchmark_coder(
        trace, "PMATIC", profile_memory=False, pmatic_safe_scenario=False
    )
    assert result["exact_roundtrip_valid"]
    assert result["numerical_reproducibility"] == []
    assert result["encode_peak_traced_bytes"] is None
    assert result["decode_peak_traced_bytes"] is None
    assert result["memory_metric"] == "not measured"


def test_full_vocabulary_sweep_uses_valid_total_and_can_omit_pmatic():
    from research.papers.cidr_2027.experiments.compare_coders_model import configurations

    settings = configurations(50257, include_pmatic=False)
    assert all(row["coder"] != "PMATIC" for _, row in settings)
    assert dict(settings)["AC_total_low"]["total"] == 65536
    assert dict(settings)["ANS_low_block64_lanes1"]["total"] == 65536
