"""Focused CUDA-kernel safety, boundary, and numerical-contract coverage."""

import numpy as np
import pytest
import torch

from src.coding.cuda_ac import (
    CudaPipelinedACEncoder,
    encode_intervals_cuda,
    target_intervals_from_probs_cuda,
    target_intervals_from_probs_cuda_v3,
    target_intervals_from_probs_cuda_v4,
)
from src.coding.multistream_ac import MultistreamACEncoder


requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"
)


def test_pipelined_encoder_rejects_int64_product_overflow_before_cuda_use():
    with pytest.raises(ValueError, match="signed 64-bit"):
        CudaPipelinedACEncoder(
            1, 1, "cuda", nominal_total=1 << 30, state_bits=40
        )


@requires_cuda
@pytest.mark.parametrize(
    "quantizer",
    [
        target_intervals_from_probs_cuda,
        target_intervals_from_probs_cuda_v3,
        target_intervals_from_probs_cuda_v4,
    ],
)
def test_cuda_quantizers_support_empty_batches_without_launching(quantizer):
    probabilities = torch.empty((0, 2), dtype=torch.float32, device="cuda")
    targets = torch.empty(0, dtype=torch.int64, device="cuda")
    outputs = quantizer(probabilities, targets)
    torch.cuda.synchronize()
    assert [tuple(output.shape) for output in outputs] == [(0,)] * 4


@requires_cuda
@pytest.mark.parametrize("alphabet", [2, 255, 256, 257, 1023, 1024, 1025])
def test_cuda_v4_tile_boundaries_are_exact_deterministic_and_handle_signed_zero(
    alphabet,
):
    rng = np.random.default_rng(9081 + alphabet)
    rows = rng.random((7, alphabet), dtype=np.float32)
    rows[0, ::2] = 0.0
    rows[0, 1::4] = -0.0
    rows[0, 0] = 1.0
    rows[1] = 1.0
    rows[2, 0] = np.nextafter(np.float32(0), np.float32(1))
    targets_host = np.array(
        [0, 1, alphabet // 4, alphabet // 2, alphabet - 2, alphabet - 1, 0],
        dtype=np.int64,
    )

    # The one-element prefix gives a non-default but naturally aligned storage
    # offset while remaining contiguous, exercising pointer-offset handling.
    storage = torch.empty(rows.size + 1, dtype=torch.float32, device="cuda")
    probabilities = storage[1:].view_as(torch.from_numpy(rows))
    probabilities.copy_(torch.from_numpy(rows).to("cuda"))
    targets = torch.from_numpy(targets_host).to("cuda")

    first = target_intervals_from_probs_cuda_v4(probabilities, targets)
    second = target_intervals_from_probs_cuda_v4(probabilities, targets)
    torch.cuda.synchronize()
    for first_tensor, second_tensor in zip(first, second):
        assert torch.equal(first_tensor, second_tensor)

    expected_lows = []
    expected_highs = []
    expected_probabilities = []
    for row32, target in zip(rows, targets_host):
        row64 = row32.astype(np.float64)
        normalized = row64 / row64.sum(dtype=np.float64)
        frequencies = (
            np.floor(normalized * (262144 - alphabet)).astype(np.int64) + 1
        )
        residual = int(262144 - frequencies.sum())
        order = np.argsort(-normalized, kind="stable")
        frequencies[order[:residual]] += 1
        cumulative = np.concatenate(([0], frequencies.cumsum()))
        expected_lows.append(int(cumulative[target]))
        expected_highs.append(int(cumulative[target + 1]))
        expected_probabilities.append(float(normalized[target]))

    assert first[0].cpu().tolist() == expected_lows
    assert first[1].cpu().tolist() == expected_highs
    assert first[2].cpu().tolist() == [262144] * len(rows)
    np.testing.assert_array_equal(first[3].cpu().numpy(), expected_probabilities)


@requires_cuda
def test_cuda_v4_rejects_finite_float64_that_overflows_float32_contract():
    probabilities = torch.tensor(
        [[torch.finfo(torch.float64).max, 1.0]],
        dtype=torch.float64,
        device="cuda",
    )
    with pytest.raises(ValueError, match="converted to float32"):
        target_intervals_from_probs_cuda_v4(probabilities, [0])


@requires_cuda
@pytest.mark.parametrize("streams", [1, 127, 128, 129])
@pytest.mark.parametrize("steps", [0, 1, 17])
def test_cuda_encoder_partial_blocks_and_zero_steps(streams, steps):
    lows = torch.zeros((steps, streams), dtype=torch.int64, device="cuda")
    highs = torch.ones_like(lows)
    totals = torch.full_like(lows, 2)
    counts = torch.full((streams,), steps, dtype=torch.int64, device="cuda")
    actual = encode_intervals_cuda(lows, highs, totals, counts)
    torch.cuda.synchronize()

    reference = MultistreamACEncoder(streams, target_interval=True)
    for _ in range(steps):
        for stream in range(streams):
            reference.encode_interval(stream, 0, 1, 2)
    assert actual == reference.finish()


@requires_cuda
def test_cuda_encoder_rejects_noncontiguous_interval_storage():
    base = torch.zeros((2, 4), dtype=torch.int64, device="cuda")
    lows = base[:, ::2]
    highs = torch.ones_like(lows)
    totals = torch.full_like(lows, 2)
    counts = torch.ones(2, dtype=torch.int64, device="cuda")
    with pytest.raises(ValueError, match="lows must be contiguous"):
        encode_intervals_cuda(lows, highs, totals, counts)
