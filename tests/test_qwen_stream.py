import pytest

from src.qwen_stream import (
    parse_qwen_stream,
    serialize_qwen_stream,
    verify_decoded_bytes,
)


def test_qwen_stream_preserves_bits_and_decoder_configuration():
    raw = b"exact UTF-8 bytes"
    blob = serialize_qwen_stream(
        [1, 0, 1, 1, 0],
        model_name="Qwen/Qwen2.5-0.5B",
        seed_token_ids=[785],
        token_count=4,
        raw_size_bytes=len(raw),
        context_length=1000,
        retain_tokens=100,
        use_kv_cache=True,
        decoded_bytes=raw,
        candidate_token_ids=[9, 3, 9],
    )
    parsed = parse_qwen_stream(blob)
    assert parsed["bits"] == [1, 0, 1, 1, 0]
    assert parsed["candidate_token_ids"] == [3, 9]
    assert parsed["seed_token_ids"] == [785]
    assert parsed["payload_bit_count"] == 5
    verify_decoded_bytes(parsed, raw)


def test_qwen_stream_rejects_changed_decoded_bytes_and_padding():
    raw = b"abc"
    blob = serialize_qwen_stream(
        [1], model_name="model", seed_token_ids=[1], token_count=2,
        raw_size_bytes=3, context_length=8, retain_tokens=4,
        use_kv_cache=False, decoded_bytes=raw,
    )
    parsed = parse_qwen_stream(blob)
    with pytest.raises(AssertionError):
        verify_decoded_bytes(parsed, b"abd")
    with pytest.raises(ValueError, match="padding"):
        parse_qwen_stream(blob[:-1] + bytes([blob[-1] | 1]))
