from src.utils import *
from src.fast_ac import AC_FAST2_FORMAT, FastACCompressor

from types import SimpleNamespace

import pytest
import torch

def test_check_mismatch(tmp_path):
    # Create two files: original and reconstructed
    # write the original into the project's data/ directory because
    # `check_mismatch` expects input_path to be relative to data/
    from pathlib import Path
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    filename = f"tmp_test_original_{os.getpid()}.txt"
    original = data_dir / filename
    reconstructed = tmp_path / "reconstructed.txt"
    original.write_text("hello world\n")
    reconstructed.write_text("hello world\n")

    # Should match
    assert check_mismatch(filename, output_path=str(reconstructed)) is True

    # Now change reconstructed to differ
    reconstructed.write_text("goodbye\n")
    assert check_mismatch(filename, output_path=str(reconstructed)) is False


def test_save_load_ac_fast_payload(tmp_path):
    compressor = FastACCompressor(stream_count=2)
    compressor.encode_batch(
        row_ids=[0, 1],
        target_token_ids=[1, 0],
        probs=torch.tensor([[0.2, 0.8], [0.7, 0.3]], dtype=torch.float32),
    )
    payload = compressor.finish()

    args = SimpleNamespace(
        output_path=str(tmp_path / "compressed.bin"),
        input_path="data/text8",
        model_name="fake",
        context_length=16,
        first_n_tokens=4,
        retain_tokens=8,
        use_kv_cache=True,
        batch_size=2,
        encoding="AC_MULTISTREAM",
        reduce_tokens=True,
        engine="transformer",
        lora_path=None,
        pmatic_delta=None,
        pmatic_r=None,
        vllm_window_size=1,
        encode_backend="numba",
        encode_threads=3,
        pipeline_encoding=True,
    )

    save_global_mask_file(args, first_token=[3, 4], bit_string=payload, bitmask_data=b"mask")

    load_args = SimpleNamespace(input_path=args.output_path)
    loaded_args, first_token, loaded_payload, bitmask_data = load_global_mask_file(load_args)

    assert loaded_args.encoding == "AC_MULTISTREAM"
    assert loaded_args.encode_backend == "numba"
    assert loaded_args.encode_threads == 3
    assert loaded_args.pipeline_encoding is True
    assert first_token == [3, 4]
    assert loaded_payload["format"] == payload["format"]
    assert loaded_payload["streams"] == payload["streams"]
    assert bitmask_data == b"mask"


def test_save_load_ac_fast2_payload(tmp_path):
    compressor = FastACCompressor(stream_count=1, payload_format=AC_FAST2_FORMAT)
    compressor.encode_batch(
        row_ids=[0],
        target_token_ids=[1],
        probs=torch.tensor([[0.2, 0.8]], dtype=torch.float32),
    )
    payload = compressor.finish()

    args = SimpleNamespace(
        output_path=str(tmp_path / "compressed-fast2.bin"),
        input_path="data/text8",
        model_name="fake",
        context_length=16,
        first_n_tokens=2,
        retain_tokens=8,
        use_kv_cache=True,
        batch_size=1,
        encoding="AC_TARGET_INTERVAL",
        reduce_tokens=True,
        engine="transformer",
        lora_path=None,
        pmatic_delta=None,
        pmatic_r=None,
        vllm_window_size=1,
        encode_backend="numba",
        encode_threads=2,
        pipeline_encoding=False,
    )

    save_global_mask_file(args, first_token=[3], bit_string=payload, bitmask_data=b"mask")
    loaded_args, first_token, loaded_payload, bitmask_data = load_global_mask_file(
        SimpleNamespace(input_path=args.output_path)
    )

    assert loaded_args.encoding == "AC_TARGET_INTERVAL"
    assert first_token == [3]
    assert loaded_payload["format"] == AC_FAST2_FORMAT
    assert loaded_payload["streams"] == payload["streams"]
    assert bitmask_data == b"mask"


def test_save_load_ac_fast_packed_payload(tmp_path):
    compressor = FastACCompressor(stream_count=1, backend="numba_packed")
    compressor.encode_batch(
        row_ids=[0],
        target_token_ids=[1],
        probs=torch.tensor([[0.2, 0.8]], dtype=torch.float32),
    )
    payload = compressor.finish()

    args = SimpleNamespace(
        output_path=str(tmp_path / "compressed-packed.bin"),
        input_path="data/text8",
        model_name="fake",
        context_length=16,
        first_n_tokens=2,
        retain_tokens=8,
        use_kv_cache=True,
        batch_size=1,
        encoding="AC_MULTISTREAM",
        reduce_tokens=True,
        engine="transformer",
        lora_path=None,
        pmatic_delta=None,
        pmatic_r=None,
        vllm_window_size=1,
        encode_backend="numba_packed",
        encode_threads=2,
        pipeline_encoding=True,
    )

    save_global_mask_file(args, first_token=[3], bit_string=payload, bitmask_data=b"mask")
    loaded_args, first_token, loaded_payload, bitmask_data = load_global_mask_file(
        SimpleNamespace(input_path=args.output_path)
    )

    assert loaded_args.encode_backend == "numba_packed"
    assert loaded_args.pipeline_encoding is True
    assert first_token == [3]
    assert loaded_payload["stream_mode"] == "packed_bytes"
    assert loaded_payload["bit_counts"] == payload["bit_counts"]
    assert loaded_payload["streams"] == payload["streams"]
    assert bitmask_data == b"mask"
