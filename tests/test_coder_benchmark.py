import math

import numpy as np
import pytest

from src.coder_benchmark import (
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
