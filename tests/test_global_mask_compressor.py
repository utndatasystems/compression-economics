# test for global_mask_compressor.py

from types import SimpleNamespace
import pytest
import numpy as np
import torch

from src.encoding import LLMCompressor, choose_pmatic_r
from src.global_mask_compressor import (
    run_global_mask_decompression,
    run_global_mask_speculative_decompression,
)


def test_global_mask_pmatic_decompression_uses_pmatic_decoder(monkeypatch):
    probs_by_prompt = {
        (0,): np.array([0.1, 0.7, 0.1, 0.1], dtype=np.float64),
        (2,): np.array([0.1, 0.1, 0.1, 0.7], dtype=np.float64),
        (0, 1): np.array([0.1, 0.1, 0.7, 0.1], dtype=np.float64),
        (2, 3): np.array([0.1, 0.7, 0.1, 0.1], dtype=np.float64),
    }

    class FakeTokenPredictor:
        tokens_list = [0, 1, 2, 3]

        def __init__(self, args, bitmap_data=None):
            self.args = args
            self.bitmap_data = bitmap_data

        def run_batched_inference(self, prompts, enable_kv_cache=True):
            probs = [probs_by_prompt[tuple(prompt)] for prompt in prompts]
            return (
                self.tokens_list,
                torch.tensor(np.stack(probs), dtype=torch.float32),
                0.0,
                0.0,
            )

        def get_token_by_id(self, token_idx):
            return self.tokens_list[token_idx]

        def detokenize(self, tokens):
            return " ".join(str(token) for token in tokens)

    monkeypatch.setattr(
        "src.global_mask_compressor.TokenPredictor",
        FakeTokenPredictor,
    )

    delta = 1e-3
    r = choose_pmatic_r(delta)
    compressor = LLMCompressor(
        algorithm="PMATIC",
        alphabet_size=len(FakeTokenPredictor.tokens_list),
        delta=delta,
        r=r,
    )
    for token_idx, probs in [
        (1, probs_by_prompt[(0,)]),
        (3, probs_by_prompt[(2,)]),
        (2, probs_by_prompt[(0, 1)]),
        (1, probs_by_prompt[(2, 3)]),
    ]:
        compressor.next_token(token_idx, probs)

    args = SimpleNamespace(
        model_name="fake",
        engine="transformer",
        reduce_tokens=True,
        encoding="PMATIC",
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
        spec_k=None,
        first_n_tokens=6,
        batch_size=2,
        context_length=128,
        retain_tokens=64,
        use_kv_cache=False,
        pmatic_delta=delta,
        pmatic_r=r,
    )

    reconstructed_tokens, detoken_string, _ = run_global_mask_decompression(
        args=args,
        first_tokens=[0, 2],
        bit_string=compressor.compress(),
        bitmap=None,
    )

    assert reconstructed_tokens == [0, 1, 2, 2, 3, 1]
    assert detoken_string == "0 1 2 2 3 1"

@pytest.mark.skip(reason="Temporarily disabled for debugging")
def test_run_global_mask_speculative_decompression(monkeypatch):
    """Smoke/integration test for speculative decompression.

    This test replaces LLMDecompressor with a dummy verifier that always
    decodes the argmax token from the provided probability vector.
    That avoids needing a real compressed bitstream while still exercising
    the new speculative decompression loop.
    """

    class DummyDecompressor:
        def __init__(self, bit_string):
            self.bit_string = bit_string

        def decompress(self, probs):
            # Return the column index of the highest-probability token.
            return int(probs.argmax())

    # Patch the decompressor used inside global_mask_compressor.py
    monkeypatch.setattr(
        "src.global_mask_compressor.LLMDecompressor",
        DummyDecompressor,
    )

    args = SimpleNamespace(
        model_name="gpt2",
        engine="transformer",
        reduce_tokens=False,
        encoding="AC",
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
        spec_k=3,
        first_n_tokens=8,
        batch_size=2,
        context_length=128,
        retain_tokens=64,
        use_kv_cache=False,
    )

    first_tokens = [1, 4]
    bit_string = "dummy_bitstring"
    bitmap = None

    reconstructed_tokens, detoken_string, stats = run_global_mask_speculative_decompression(
        args=args,
        first_tokens=first_tokens,
        bit_string=bit_string,
        bitmap=bitmap,
    )

    print("First tokens:", first_tokens)
    print("Reconstructed tokens:", reconstructed_tokens)
    print("Detokenized string:", detoken_string)
    print("Stats:", stats)

    assert isinstance(reconstructed_tokens, list)
    assert all(isinstance(tok, int) for tok in reconstructed_tokens)

    assert isinstance(detoken_string, str)

    assert isinstance(stats, dict)
    for key in [
        "args",
        "decompression_time_sec",
        "input_tokens_cnt",
        "total_decompression_time",
        "detokenize_time",
        "inference_time",
        "ac_time",
        "data_copy_time",
        "softmax_time",
        "throughput_kibibytes_per_sec",
        "inference_throughput_kibibytes_per_sec",
    ]:
        assert key in stats, f"Missing stats key: {key}"

    # We expect at least the seed tokens to be present
    assert len(reconstructed_tokens) >= len(first_tokens)

    # If first_n_tokens is distributed across the batch, total output should
    # typically match first_n_tokens exactly.
    assert len(reconstructed_tokens) == args.first_n_tokens, (
        f"Expected {args.first_n_tokens} reconstructed tokens, "
        f"got {len(reconstructed_tokens)}"
    )

    print("test_run_global_mask_speculative_decompression passed.\n")
