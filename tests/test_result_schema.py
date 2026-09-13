import hashlib

import pytest

from src.result_schema import (
    CoderSpec,
    DatasetSpec,
    ExecutionSpec,
    PlainTextCompressionResult,
    PredictorSpec,
    SizeBreakdown,
    SymbolCounts,
    TimingBreakdown,
    TokenizerSpec,
)


def _result(*, tokenizer_bytes=20) -> PlainTextCompressionResult:
    source = b"reproducible text"
    return PlainTextCompressionResult(
        dataset=DatasetSpec(
            name="fixture",
            sha256=hashlib.sha256(source).hexdigest(),
            source_bytes=len(source),
        ),
        tokenizer=TokenizerSpec(
            name="byte", kind="byte", vocabulary_size=256,
            state_bytes=tokenizer_bytes,
        ),
        predictor=PredictorSpec(
            name="bigram", family="ngram", context_length=1,
            total_parameters=0, active_parameters=0, model_state_bytes=30,
        ),
        coder=CoderSpec(name="AC", probability_total=262_144),
        execution=ExecutionSpec(
            device="cpu", backend="torch", batch_size=2, seed=7,
        ),
        counts=SymbolCounts(
            input_symbols=17, encoded_symbols=15, model_input_tokens=150,
        ),
        sizes=SizeBreakdown(
            payload_bits=65, payload_bytes=9, framing_bytes=4,
            bitmap_bytes=3, seed_bytes=8, tokenizer_bytes=tokenizer_bytes,
            model_bytes=30, adapter_bytes=0,
        ),
        timings=TimingBreakdown(encode_seconds=1.0, decode_seconds=2.0),
        roundtrip_valid=True,
    )


def test_canonical_result_derives_non_overlapping_accounting_and_throughput():
    record = _result().to_dict()

    assert record["schema_version"] == 1
    assert record["derived"]["stream_bytes"] == 24
    assert record["derived"]["self_contained_bytes"] == 74
    assert record["derived"]["shared_model_compression_factor"] == 17 / 24
    assert record["derived"]["encode_input_symbols_per_second"] == 17
    assert record["derived"]["decode_input_symbols_per_second"] == 8.5
    assert record["derived"]["encode_model_input_tokens_per_second"] == 150
    assert record["derived"]["decode_model_input_tokens_per_second"] == 75
    assert len(record["condition_id"]) == 16


def test_condition_id_excludes_observations():
    original = _result()
    slower = PlainTextCompressionResult(
        dataset=original.dataset,
        tokenizer=original.tokenizer,
        predictor=original.predictor,
        coder=original.coder,
        execution=original.execution,
        counts=original.counts,
        sizes=original.sizes,
        timings=TimingBreakdown(encode_seconds=9.0, decode_seconds=10.0),
        roundtrip_valid=True,
        repetition=9,
    )

    assert original.condition_id == slower.condition_id


def test_unknown_dependency_size_suppresses_self_contained_claim():
    record = _result(tokenizer_bytes=None).to_dict()

    assert record["derived"]["self_contained_bytes"] is None
    assert record["derived"]["self_contained_compression_factor"] is None


def test_missing_model_input_count_suppresses_model_work_rates():
    original = _result()
    result = PlainTextCompressionResult(
        dataset=original.dataset,
        tokenizer=original.tokenizer,
        predictor=original.predictor,
        coder=original.coder,
        execution=original.execution,
        counts=SymbolCounts(input_symbols=17, encoded_symbols=15),
        sizes=original.sizes,
        timings=original.timings,
        roundtrip_valid=True,
    ).to_dict()

    assert result["derived"]["encode_model_input_tokens_per_second"] is None
    assert result["derived"]["decode_model_input_tokens_per_second"] is None


def test_payload_byte_count_is_validated():
    with pytest.raises(ValueError, match="ceil"):
        SizeBreakdown(payload_bits=9, payload_bytes=1)
