"""Quantizer, archive, and actual local Transformer coverage for O3."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from src.coding.multistream_ac import MultistreamACEncoder, MultistreamACDecoder
from src.coding.target_interval import target_intervals_from_probs_tensor
from src.prediction import TokenPredictor


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_intervals_match_independent_floor_cdf_and_decode(device, dtype):
    rng = np.random.default_rng(91)
    values = rng.dirichlet(np.ones(17), size=100)
    values[0] = 0
    values[0, 0] = 1
    values[1] = 1 / 17
    probabilities = torch.tensor(values, dtype=dtype, device=device)
    targets = torch.arange(100, device=device) % 17
    lows, highs, totals, _ = target_intervals_from_probs_tensor(probabilities, targets)
    encoder = MultistreamACEncoder(3, target_interval=True)
    for index, row in enumerate(probabilities.cpu().numpy()):
        widened = row.astype(np.float64)
        frequencies = np.floor(widened / widened.sum() * (262144 - 17)).astype(np.int64) + 1
        cdf = np.concatenate(([0], frequencies.cumsum()))
        symbol = index % 17
        assert (int(lows[index]), int(highs[index]), int(totals[index])) == (
            int(cdf[symbol]), int(cdf[symbol + 1]), int(cdf[-1]))
        encoder.encode_interval(index % 3, int(lows[index]), int(highs[index]), int(totals[index]))
    archive = encoder.finish()
    assert archive[4] == 2
    decoder = MultistreamACDecoder(archive)
    assert decoder.target_interval
    for index, row in enumerate(probabilities.cpu().numpy()):
        assert decoder.decode(index % 3, row) == index % 17
    decoder.assert_complete()


@pytest.mark.parametrize("backend", ["python", "numba_parallel"])
def test_variable_totals_and_empty_streams_match_python(backend):
    if backend == "numba_parallel":
        pytest.importorskip("numba")
    reference = MultistreamACEncoder(3, target_interval=True)
    encoder = MultistreamACEncoder(3, target_interval=True, backend=backend)
    for index in range(100):
        total = 100 + index
        for instance in (reference, encoder):
            instance.encode_interval(index % 2, index % 31, index % 31 + 3, total)
    assert encoder.finish() == reference.finish()


@pytest.mark.parametrize("values,targets", [
    ([[float("nan"), 1]], [0]), ([[0, 0]], [0]), ([[-1, 2]], [0]),
    ([[0.5, 0.5]], [2]), ([[0.5, 0.5]], []),
])
def test_invalid_distributions_or_targets_rejected(values, targets):
    with pytest.raises(ValueError):
        target_intervals_from_probs_tensor(torch.tensor(values), targets)


def test_version_flags_and_interval_bounds_rejected():
    encoder = MultistreamACEncoder(1, target_interval=True)
    with pytest.raises(ValueError):
        encoder.encode_interval(0, 3, 2, 100)
    with pytest.raises(ValueError):
        encoder.encode_interval(0, 0, 3, 262145)
    with pytest.raises(ValueError):
        encoder.encode(0, 0, np.array([0.5, 0.5]))
    archive = bytearray(encoder.finish())
    archive[4] = 1  # The v2 flag is not valid in v1.
    with pytest.raises(ValueError):
        MultistreamACDecoder(bytes(archive))


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("mask", [False, True])
def test_tiny_transformer_capture_decodes_with_full_distribution(cache, mask):
    torch.manual_seed(23)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=32, n_positions=32, n_embd=16, n_layer=1, n_head=2)).eval()
    def make():
        predictor = TokenPredictor.__new__(TokenPredictor)
        predictor.model = model
        predictor.device = torch.device("cpu")
        predictor.args = SimpleNamespace(engine="transformer", encoding="AC_TARGET_INTERVAL")
        predictor.tokens_list = [1, 3, 7, 9] if mask else list(range(32))
        predictor.reduce_tokens = mask
        predictor.index_tensor = torch.tensor(predictor.tokens_list)
        predictor.reset_kv_cache()
        return predictor
    encoder_predictor, decoder_predictor = make(), make()
    encoder = MultistreamACEncoder(2, target_interval=True)
    decoded_probabilities = []
    for prompts, targets in [([[1], [3]], [7, 9]), ([[1, 7], [3, 9]], [3, 1])]:
        _, intervals, _, _ = encoder_predictor.run_batched_interval_inference(prompts, targets, cache)
        assert encoder_predictor.last_interval_transfer_bytes == 64
        assert all(values.shape == (2,) for values in intervals.values())
        probabilities = decoder_predictor.run_batched_inference(prompts, cache)[1]
        decoded_probabilities.append((targets, probabilities))
        for row in range(2):
            encoder.encode_interval(row, int(intervals["lows"][row]),
                                    int(intervals["highs"][row]), int(intervals["totals"][row]))
    decoder = MultistreamACDecoder(encoder.finish())
    for targets, probabilities in decoded_probabilities:
        assert [decoder_predictor.tokens_list[decoder.decode(row, probabilities[row].numpy())]
                for row in range(2)] == targets
    decoder.assert_complete()


@pytest.mark.parametrize("backend", [
    "python",
    "numba_parallel",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(
            not torch.cuda.is_available(), reason="CUDA unavailable"
        ),
    ),
])
@pytest.mark.parametrize("decode_backend", ["host", "device"])
def test_local_transformer_text_archive_roundtrip(tmp_path, monkeypatch, backend, decode_backend):
    if backend == "numba_parallel":
        pytest.importorskip("numba")
    from src.global_mask_compressor import run_global_mask_compression, run_global_mask_decompression
    from src.utils import save_global_mask_file, load_global_mask_file
    from tests.test_global_mask_compressor import _CharacterTokenizer
    import src.prediction as prediction_module
    torch.manual_seed(8)
    model = GPT2LMHeadModel(GPT2Config(
        vocab_size=256, n_positions=16, n_embd=16, n_layer=1, n_head=2)).eval()
    monkeypatch.setattr(prediction_module.AutoTokenizer, "from_pretrained",
                        lambda *args, **kwargs: _CharacterTokenizer())
    monkeypatch.setattr(prediction_module.AutoModelForCausalLM, "from_pretrained",
                        lambda *args, **kwargs: model)
    source = tmp_path / "source.txt"
    source.write_text("ab\r\nabba\nab", newline="")
    args = SimpleNamespace(
        mode="compress", input_path=str(source), text_input=None,
        output_path=str(tmp_path / "archive.bin"), model_name="local-test",
        engine="transformer", encoding="AC_TARGET_INTERVAL", lora_path=None,
        reduce_tokens=True, is_mamba=False, is_seq2seq=False, first_n_tokens=None,
        batch_size=3, context_length=4, retain_tokens=2, use_kv_cache=True,
        ac_backend=backend, ac_threads=2, spec_k=None)
    seeds, payload, bitmap, stats, args = run_global_mask_compression(args)
    if backend == "cuda":
        assert stats["interval_transfer_bytes"] == 0
        assert stats["cuda_ac_metrics"]["kernel_seconds"] > 0
        assert stats["cuda_ac_metrics"]["device_to_host_bytes"] > 0
    else:
        assert stats["interval_transfer_bytes"] > 0
    save_global_mask_file(args, seeds, payload, bitmap)
    args.mode = "decompress"
    args.ac_decode_backend = decode_backend
    args.input_path = args.output_path
    args, seeds, payload, bitmap = load_global_mask_file(args)
    tokens, text, decode_stats = run_global_mask_decompression(args, seeds, payload, bitmap)
    assert decode_stats["ac_decode_backend"] == decode_backend
    assert bytes(tokens) == source.read_bytes()
    assert text.encode() == source.read_bytes()
