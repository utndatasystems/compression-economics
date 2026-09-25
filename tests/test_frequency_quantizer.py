"""Bitstream compatibility checks for the switchable frequency quantizer."""

import numpy as np
import pytest

from src.coder_benchmark import (
    benchmark_coder,
    decode_multistream_probability_stream,
    decode_probability_stream,
    encode_multistream_probability_stream,
    encode_probability_stream,
    synthetic_probability_trace,
)
from src.coding.encoding_utils import build_cumul


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("alphabet,total", [(2, 16), (8, 256), (257, 1024)])
def test_exact_cdf_parity_on_random_rows_and_ties(dtype, alphabet, total):
    rng = np.random.default_rng(43)
    rows = [rng.dirichlet(np.ones(alphabet)).astype(dtype) for _ in range(50)]
    rows += [np.full(alphabet, 1 / alphabet, dtype=dtype)]
    zeros = np.zeros(alphabet, dtype=dtype)
    zeros[0] = 1
    rows.append(zeros)
    for row in rows:
        assert np.array_equal(
            build_cumul(row, total, method="reference"),
            build_cumul(row, total, method="vectorized_exact"),
        )


def test_exact_cdf_parity_on_full_passes_and_negative_fallback():
    # The API accepts raw arrays; these deliberately unnormalized rows exercise
    # both adjustment branches, including the reference decrement fallback.
    for row in (np.array([0.1, 0.1, 0.1]), np.array([0.6, 0.6])):
        assert np.array_equal(
            build_cumul(row, 32, method="reference"),
            build_cumul(row, 32, method="vectorized_exact"),
        )


def test_unknown_method_is_rejected():
    with pytest.raises(ValueError, match="frequency_quantizer"):
        build_cumul(np.array([0.5, 0.5]), method="unknown")


@pytest.fixture(scope="module")
def trace():
    return synthetic_probability_trace(28, 8, seed=19)


@pytest.mark.parametrize("coder", ["AC", "ANS", "PMATIC"])
def test_probability_coder_archives_match_and_cross_decode(trace, coder):
    kwargs = dict(total=256, ans_block_size=11, ans_lanes=1, pmatic_delta=0.01)
    reference = encode_probability_stream(
        trace, coder, frequency_quantizer="reference", **kwargs
    )
    optimized = encode_probability_stream(
        trace, coder, frequency_quantizer="vectorized_exact", **kwargs
    )
    assert optimized.archive == reference.archive
    assert optimized.bits == reference.bits
    decode_kwargs = dict(total=256, alphabet_size=trace.alphabet_size,
                         ans_lanes=1, pmatic_delta=0.01)
    for stream, method in ((reference, "vectorized_exact"),
                           (optimized, "reference")):
        decoded = decode_probability_stream(
            stream, trace.probabilities, coder,
            frequency_quantizer=method, **decode_kwargs
        )
        assert np.array_equal(decoded, trace.symbols)


def test_multistream_archive_matches_and_cross_decodes(trace):
    kwargs = dict(total=256, stream_count=4)
    reference = encode_multistream_probability_stream(
        trace, frequency_quantizer="reference", **kwargs
    )
    optimized = encode_multistream_probability_stream(
        trace, frequency_quantizer="vectorized_exact", **kwargs
    )
    assert optimized.archive == reference.archive
    for stream, method in ((reference, "vectorized_exact"),
                           (optimized, "reference")):
        decoded = decode_multistream_probability_stream(
            stream, trace.probabilities, frequency_quantizer=method
        )
        assert np.array_equal(decoded, trace.symbols)


@pytest.mark.parametrize("coder", ["AC", "ANS", "AC_MULTISTREAM"])
def test_benchmark_records_implementation_and_timing_scope(trace, coder):
    result = benchmark_coder(
        trace, coder, total=256, ac_streams=4, profile_memory=False,
        frequency_quantizer="vectorized_exact"
    )
    assert result["parameters"]["frequency_quantizer"] == "vectorized_exact"
    assert result["exact_roundtrip_valid"]
    assert result["quantize_seconds"] > 0
    assert result["quantize_timing_scope"] == (
        "within_encode" if coder == "AC_MULTISTREAM" else "separate_trace_pass"
    )


def test_text_archive_records_nondefault_quantizer_and_legacy_default(tmp_path):
    from types import SimpleNamespace

    from src.utils import load_global_mask_file, make_key, save_global_mask_file

    common = dict(
        input_path=str(tmp_path / "source.txt"), output_path=str(tmp_path / "coded.bin"),
        model_name="fake", context_length=32, first_n_tokens=3,
        retain_tokens=16, use_kv_cache=False, batch_size=1,
        encoding="AC", reduce_tokens=True, engine="transformer", lora_path=None,
    )
    reference = SimpleNamespace(**common, frequency_quantizer="reference")
    optimized = SimpleNamespace(**common, frequency_quantizer="vectorized_exact")
    assert make_key(reference) != make_key(optimized)

    for args in (reference, optimized):
        save_global_mask_file(args, [1], [1, 0, 1], b"mask")
        loaded = SimpleNamespace(**vars(args))
        loaded.input_path = args.output_path
        loaded, seeds, bits, bitmap = load_global_mask_file(loaded)
        assert loaded.frequency_quantizer == args.frequency_quantizer
        assert (seeds, bits, bitmap) == ([1], [1, 0, 1], b"mask")

    # Older files without the optional header field still select the reference.
    save_global_mask_file(reference, [1], [1, 0, 1], b"mask")
    assert b"frequency_quantizer" not in (tmp_path / "coded.bin").read_bytes()
