# test for global_mask_compressor.py

from types import SimpleNamespace
import pytest
import numpy as np
import torch

from src.encoding import LLMCompressor, choose_pmatic_r
from src.fast_ac import AC_FAST2_FORMAT, FastACCompressor, target_intervals_from_probs_tensor
from src.global_mask_compressor import (
    run_global_mask_compression,
    run_global_mask_decompression,
    run_global_mask_speculative_decompression,
)


@pytest.mark.parametrize("engine", ["transformer", "vllm"])
def test_global_mask_pmatic_decompression_uses_pmatic_decoder(monkeypatch, engine):
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

    def fake_get_token_predictor(args, bitmap_data=None):
        return FakeTokenPredictor(args, bitmap_data)

    monkeypatch.setattr(
        "src.global_mask_compressor.get_token_predictor",
        fake_get_token_predictor,
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
        engine=engine,
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


def test_global_mask_ac_fast2_decompression_uses_row_streams(monkeypatch):
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
        "src.global_mask_compressor.get_token_predictor",
        lambda args, bitmap_data=None: FakeTokenPredictor(args, bitmap_data),
    )

    compressor = FastACCompressor(stream_count=2, payload_format=AC_FAST2_FORMAT)
    compressor.encode_batch(
        [0, 1],
        [1, 3],
        torch.tensor(np.stack([probs_by_prompt[(0,)], probs_by_prompt[(2,)]]), dtype=torch.float32),
    )
    compressor.encode_batch(
        [0, 1],
        [2, 1],
        torch.tensor(np.stack([probs_by_prompt[(0, 1)], probs_by_prompt[(2, 3)]]), dtype=torch.float32),
    )

    args = SimpleNamespace(
        model_name="fake",
        engine="transformer",
        reduce_tokens=True,
        encoding="AC_TARGET_INTERVAL",
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
        spec_k=None,
        first_n_tokens=6,
        batch_size=2,
        context_length=128,
        retain_tokens=64,
        use_kv_cache=False,
    )

    reconstructed_tokens, detoken_string, _ = run_global_mask_decompression(
        args=args,
        first_tokens=[0, 2],
        bit_string=compressor.finish(),
        bitmap=None,
    )

    assert reconstructed_tokens == [0, 1, 2, 2, 3, 1]
    assert detoken_string == "0 1 2 2 3 1"


def test_global_mask_ac_fast2_compression_uses_interval_inference(monkeypatch):
    interval_calls = []

    class FakeTokenDataPreparer:
        def __init__(self, args):
            self.args = args

        def get_data_tokens(self):
            return [0, 1, 2, 3]

        def get_args(self):
            return self.args

        def get_bitmap(self):
            return b"bitmap"

    class FakeTokenPredictor:
        tokens_list = [0, 1, 2, 3]

        def __init__(self, args, bitmap_data=None):
            self.args = args
            self.bitmap_data = bitmap_data

        def run_batched_interval_inference(self, prompts, target_tokens):
            interval_calls.append(([p[:] for p in prompts], target_tokens[:]))
            targets = torch.tensor(target_tokens, dtype=torch.long)
            probs = torch.full((len(prompts), 4), 0.25, dtype=torch.float32)
            lows, highs, totals, target_probs = target_intervals_from_probs_tensor(
                probs,
                targets,
            )
            return (
                self.tokens_list,
                {
                    "lows": lows,
                    "highs": highs,
                    "totals": totals,
                    "target_probs": target_probs,
                },
                0.0,
                0.0,
            )

        def detokenize(self, tokens):
            return "".join(str(token) for token in tokens)

        def cleanup(self):
            return None

    monkeypatch.setattr("src.global_mask_compressor.TokenDataPreparer", FakeTokenDataPreparer)
    monkeypatch.setattr(
        "src.global_mask_compressor.get_token_predictor",
        lambda args, bitmap_data=None: FakeTokenPredictor(args, bitmap_data),
    )

    args = SimpleNamespace(
        input_path="data/text8",
        model_name="fake",
        context_length=128,
        first_n_tokens=4,
        batch_size=2,
        use_kv_cache=True,
        retain_tokens=64,
        engine="transformer",
        encoding="AC_TARGET_INTERVAL",
        reduce_tokens=True,
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
        encode_backend="python",
    )

    _, payload, _, stats, _, _ = run_global_mask_compression(args)

    assert payload["format"] == AC_FAST2_FORMAT
    assert interval_calls == [
        ([[0], [2]], [1, 3]),
        ([[0, 1], [2, 3]], [0, 0]),
    ]
    assert stats["encode_backend"] == "python"


def test_global_mask_ac_fast_decompression_uses_row_streams(monkeypatch):
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
        "src.global_mask_compressor.get_token_predictor",
        lambda args, bitmap_data=None: FakeTokenPredictor(args, bitmap_data),
    )

    compressor = FastACCompressor(stream_count=2)
    compressor.encode_batch(
        [0, 1],
        [1, 3],
        torch.tensor(np.stack([probs_by_prompt[(0,)], probs_by_prompt[(2,)]]), dtype=torch.float32),
    )
    compressor.encode_batch(
        [0, 1],
        [2, 1],
        torch.tensor(np.stack([probs_by_prompt[(0, 1)], probs_by_prompt[(2, 3)]]), dtype=torch.float32),
    )

    args = SimpleNamespace(
        model_name="fake",
        engine="transformer",
        reduce_tokens=True,
        encoding="AC_MULTISTREAM",
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
        spec_k=None,
        first_n_tokens=6,
        batch_size=2,
        context_length=128,
        retain_tokens=64,
        use_kv_cache=False,
    )

    reconstructed_tokens, detoken_string, _ = run_global_mask_decompression(
        args=args,
        first_tokens=[0, 2],
        bit_string=compressor.finish(),
        bitmap=None,
    )

    assert reconstructed_tokens == [0, 1, 2, 2, 3, 1]
    assert detoken_string == "0 1 2 2 3 1"


@pytest.mark.parametrize("engine", ["transformer", "vllm"])
def test_global_mask_compression_uses_selected_engine(monkeypatch, engine):
    created_engines = []

    class FakeTokenDataPreparer:
        def __init__(self, args):
            self.args = args

        def get_data_tokens(self):
            return [0, 1]

        def get_args(self):
            return self.args

        def get_bitmap(self):
            return b"bitmap"

    class FakeTokenPredictor:
        tokens_list = [0, 1]

        def __init__(self, args, bitmap_data=None):
            created_engines.append(args.engine)
            self.args = args
            self.bitmap_data = bitmap_data

        def run_batched_inference(self, prompts, enable_kv_cache=True):
            del prompts, enable_kv_cache
            return self.tokens_list, torch.tensor([[0.2, 0.8]], dtype=torch.float32), 0.0, 0.0

        def detokenize(self, tokens):
            return "".join(str(token) for token in tokens)

        def cleanup(self):
            return None

    class FakeLLMCompressor:
        def __init__(self, *args, **kwargs):
            self.encoded = []

        def next_token(self, token_idx, probs):
            self.encoded.append((token_idx, probs[token_idx]))

        def compress(self, encoding="AC", rank_list=None):
            del encoding, rank_list
            return "101"

    monkeypatch.setattr(
        "src.global_mask_compressor.TokenDataPreparer",
        FakeTokenDataPreparer,
    )
    monkeypatch.setattr(
        "src.global_mask_compressor.get_token_predictor",
        lambda args, bitmap_data=None: FakeTokenPredictor(args, bitmap_data),
    )
    monkeypatch.setattr(
        "src.global_mask_compressor.LLMCompressor",
        FakeLLMCompressor,
    )

    args = SimpleNamespace(
        input_path="data/text8",
        model_name="fake",
        context_length=128,
        first_n_tokens=2,
        batch_size=1,
        use_kv_cache=True,
        retain_tokens=64,
        engine=engine,
        encoding="AC",
        reduce_tokens=True,
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
    )

    first_tokens, bit_string, bitmask_data, stats, returned_args, _ = run_global_mask_compression(args)

    assert created_engines == [engine]
    assert first_tokens == [0]
    assert bit_string == "101"
    assert bitmask_data == b"bitmap"
    assert stats["args"]["engine"] == engine
    assert returned_args.engine == engine


def test_global_mask_vllm_windowed_compression_preserves_token_order(monkeypatch):
    encoded_tokens = []
    forced_calls = []

    class FakeTokenDataPreparer:
        def __init__(self, args):
            self.args = args

        def get_data_tokens(self):
            return [0, 1, 2, 3, 0, 1]

        def get_args(self):
            return self.args

        def get_bitmap(self):
            return b"bitmap"

    class FakeTokenPredictor:
        tokens_list = [0, 1, 2, 3]

        def __init__(self, args, bitmap_data=None):
            self.args = args
            self.bitmap_data = bitmap_data

        def run_batched_forced_inference(self, prompts, target_token_windows):
            forced_calls.append((
                [prompt[:] for prompt in prompts],
                [targets[:] for targets in target_token_windows],
            ))
            steps = max(len(row) for row in target_token_windows)
            rows = len(prompts)
            probs = torch.full((steps, rows, 4), 0.25, dtype=torch.float32)
            return self.tokens_list, probs, 0.0, 0.0

        def detokenize(self, tokens):
            return "".join(str(token) for token in tokens)

        def cleanup(self):
            return None

    class FakeLLMCompressor:
        def __init__(self, *args, **kwargs):
            pass

        def next_token(self, token_idx, probs):
            del probs
            encoded_tokens.append(token_idx)

        def compress(self, encoding="AC", rank_list=None):
            del encoding, rank_list
            return "101"

    monkeypatch.setattr(
        "src.global_mask_compressor.TokenDataPreparer",
        FakeTokenDataPreparer,
    )
    monkeypatch.setattr(
        "src.global_mask_compressor.get_token_predictor",
        lambda args, bitmap_data=None: FakeTokenPredictor(args, bitmap_data),
    )
    monkeypatch.setattr(
        "src.global_mask_compressor.LLMCompressor",
        FakeLLMCompressor,
    )

    args = SimpleNamespace(
        input_path="data/text8",
        model_name="fake",
        context_length=128,
        first_n_tokens=6,
        batch_size=2,
        use_kv_cache=True,
        retain_tokens=64,
        engine="vllm",
        encoding="AC",
        reduce_tokens=True,
        is_seq2seq=False,
        is_mamba=False,
        lora_path=None,
        vllm_window_size=2,
    )

    run_global_mask_compression(args)

    assert forced_calls[0] == (
        [[0], [3]],
        [[1, 2], [0, 1]],
    )
    assert encoded_tokens == [1, 0, 2, 1]


def test_run_global_mask_speculative_decompression_rejects_vllm():
    with pytest.raises(NotImplementedError, match="Speculative decompression"):
        run_global_mask_speculative_decompression(
            args=SimpleNamespace(engine="vllm"),
            first_tokens=[0],
            bit_string="dummy",
            bitmap=None,
        )

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
