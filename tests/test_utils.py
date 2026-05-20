from src.utils import *
from src.fast_ac import FastACCompressor

from types import SimpleNamespace

import pytest
import torch

@pytest.mark.parametrize(
    "input_path, output_path, expected",
    [   (   "/home/hpc/v164be/v164be10/src/compression-economics/text_results_gt.txt",
            "/home/hpc/v164be/v164be10/src/compression-economics/text_results.txt",
            False,),
        (   "/home/hpc/v164be/v164be10/src/compression-economics/text_results.txt",
            "/home/hpc/v164be/v164be10/src/compression-economics/text_results.txt",
            True,),],)
def test_check_mismatch(input_path, output_path, expected):
    assert check_mismatch(input_path, output_path=output_path) is expected


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
        encoding="AC_FAST",
        reduce_tokens=True,
        engine="transformer",
        lora_path=None,
        pmatic_delta=None,
        pmatic_r=None,
        vllm_window_size=1,
    )

    save_global_mask_file(args, first_token=[3, 4], bit_string=payload, bitmask_data=b"mask")

    load_args = SimpleNamespace(input_path=args.output_path)
    loaded_args, first_token, loaded_payload, bitmask_data = load_global_mask_file(load_args)

    assert loaded_args.encoding == "AC_FAST"
    assert first_token == [3, 4]
    assert loaded_payload["format"] == payload["format"]
    assert loaded_payload["streams"] == payload["streams"]
    assert bitmask_data == b"mask"
