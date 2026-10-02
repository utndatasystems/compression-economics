"""Device-side arithmetic state and inverse CDF must match the host decoder."""

import numpy as np
import pytest
import torch

from src.coding.cuda_ac import (
    encode_intervals_cuda,
    target_intervals_from_probs_cuda,
)
from src.coding.device_ac import (
    CudaCDFMultistreamACDecoder,
    DeviceMultistreamACDecoder,
)
from src.coding.multistream_ac import MultistreamACDecoder, MultistreamACEncoder
from src.coding.paired_ac import pack_standard_archive
from src.coding.target_interval import target_intervals_from_probs_tensor


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
@pytest.mark.parametrize("paired", [False, True])
def test_device_decoder_matches_host_with_inactive_rows(device, paired):
    rng = np.random.default_rng(77)
    rows = rng.dirichlet(np.ones(13), size=(17, 3)).astype(np.float32)
    targets = rng.integers(0, 13, size=(17, 3))
    lengths = [17, 15, 9]
    encoder = MultistreamACEncoder(3, target_interval=True)
    for step in range(17):
        probabilities = torch.tensor(rows[step], device=device)
        target = torch.tensor(targets[step], device=device)
        low, high, total, _ = target_intervals_from_probs_tensor(probabilities, target)
        for stream, length in enumerate(lengths):
            if step < length:
                encoder.encode_interval(stream, int(low[stream]), int(high[stream]), int(total[stream]))
    standard = encoder.finish()
    archive = pack_standard_archive(standard) if paired else standard
    host = MultistreamACDecoder(standard)
    coder = DeviceMultistreamACDecoder(archive, device, paired=paired)
    for step in range(17):
        probabilities = torch.tensor(rows[step], device=device)
        active = [step < length for length in lengths]
        actual = coder.decode(probabilities, active).cpu().tolist()
        for stream in range(3):
            if active[stream]:
                expected = host.decode(stream, rows[step, stream])
                assert actual[stream] == expected == targets[step, stream]
    host.assert_complete()
    coder.assert_complete()
    with pytest.raises(ValueError, match="no more symbols"):
        coder.decode(torch.tensor(rows[-1], device=device), [True, False, False])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("paired", [False, True])
def test_cuda_cdf_decoder_matches_host_with_inactive_rows(paired):
    rng = np.random.default_rng(78)
    rows = rng.dirichlet(np.ones(29), size=(31, 5)).astype(np.float32)
    targets = rng.integers(0, 29, size=(31, 5))
    lengths = [31, 25, 17, 3, 0]
    encoder = MultistreamACEncoder(5, target_interval=True)
    rows_device = torch.tensor(rows, device="cuda")
    for step in range(31):
        low, high, total, _ = target_intervals_from_probs_tensor(
            rows_device[step], torch.tensor(targets[step], device="cuda")
        )
        for stream, length in enumerate(lengths):
            if step < length:
                encoder.encode_interval(
                    stream, int(low[stream]), int(high[stream]), int(total[stream])
                )
    standard = encoder.finish()
    archive = pack_standard_archive(standard) if paired else standard
    host = MultistreamACDecoder(standard)
    coder = CudaCDFMultistreamACDecoder(archive, "cuda", paired=paired)
    for step in range(31):
        active = [step < length for length in lengths]
        actual = coder.decode(rows_device[step], active).cpu().tolist()
        for stream, is_active in enumerate(active):
            if is_active:
                assert host.decode(stream, rows[step, stream]) == actual[stream]
                assert actual[stream] == targets[step, stream]
    host.assert_complete()
    coder.assert_complete()
    assert coder.metrics["kernel_seconds"] > 0
    assert coder.metrics["quantization_seconds"] > 0
    coder.decode(rows_device[-1], [True, False, False, False, False])
    with pytest.raises(ValueError, match="no more symbols"):
        coder.assert_complete()


def _cuda_interval_trace(steps=19, streams=4, alphabet=17):
    rng = np.random.default_rng(2027)
    probabilities = rng.dirichlet(
        np.ones(alphabet), size=(steps, streams)
    ).astype(np.float32)
    targets = rng.integers(0, alphabet, size=(steps, streams), dtype=np.int64)
    rows = torch.tensor(probabilities, device="cuda")
    target_tensor = torch.tensor(targets, device="cuda")
    lows = torch.empty((steps, streams), dtype=torch.int64, device="cuda")
    highs = torch.empty_like(lows)
    totals = torch.empty_like(lows)
    for step in range(steps):
        lows[step], highs[step], totals[step], _ = target_intervals_from_probs_tensor(
            rows[step], target_tensor[step]
        )
    return probabilities, lows, highs, totals, targets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.bfloat16])
@pytest.mark.parametrize("alphabet", [2, 17, 257])
def test_fused_cuda_target_intervals_match_torch(dtype, alphabet):
    generator = torch.Generator(device="cuda").manual_seed(2028 + alphabet)
    probabilities = torch.rand(
        (73, alphabet), dtype=dtype, device="cuda", generator=generator
    )
    probabilities[0].zero_()
    probabilities[0, 0] = 1
    targets = torch.arange(73, dtype=torch.int64, device="cuda") % alphabet
    expected = target_intervals_from_probs_tensor(probabilities, targets)
    actual = target_intervals_from_probs_cuda(probabilities, targets)
    for expected_tensor, actual_tensor in zip(expected, actual):
        assert torch.equal(actual_tensor, expected_tensor)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("workspace_bytes", [None, 1])
def test_cuda_interval_encoder_matches_python_archive(workspace_bytes):
    probabilities, lows, highs, totals, targets = _cuda_interval_trace()
    counts = torch.tensor([19, 13, 1, 0], dtype=torch.int64, device="cuda")
    metrics = {}
    actual = encode_intervals_cuda(
        lows, highs, totals, counts,
        workspace_bytes=workspace_bytes, metrics=metrics,
    )
    assert metrics["kernel_seconds"] > 0
    assert metrics["device_to_host_bytes"] > 0
    assert metrics["workspace_retries"] == (workspace_bytes == 1)

    reference = MultistreamACEncoder(4, target_interval=True)
    host_lows = lows.cpu().numpy()
    host_highs = highs.cpu().numpy()
    host_totals = totals.cpu().numpy()
    host_counts = counts.cpu().tolist()
    for step in range(lows.shape[0]):
        for stream, count in enumerate(host_counts):
            if step < count:
                reference.encode_interval(
                    stream,
                    int(host_lows[step, stream]),
                    int(host_highs[step, stream]),
                    int(host_totals[step, stream]),
                )
    expected = reference.finish()
    assert actual == expected

    decoder = MultistreamACDecoder(actual)
    assert decoder.symbol_counts == host_counts
    for step in range(lows.shape[0]):
        for stream, count in enumerate(host_counts):
            if step < count:
                assert (
                    decoder.decode(stream, probabilities[step, stream])
                    == targets[step, stream]
                )
    decoder.assert_complete()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_interval_encoder_rejects_invalid_inputs():
    _, lows, highs, totals, _ = _cuda_interval_trace(steps=2, streams=1)
    counts = torch.tensor([2], dtype=torch.int64, device="cuda")
    highs[0, 0] = lows[0, 0]
    with pytest.raises(ValueError, match="invalid target interval"):
        encode_intervals_cuda(lows, highs, totals, counts)
    highs[0, 0] += 1
    counts[0] = 3
    with pytest.raises(ValueError, match="symbol count"):
        encode_intervals_cuda(lows, highs, totals, counts)
