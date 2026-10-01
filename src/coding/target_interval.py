"""Device-side floor-count quantization for MSAC v2 target intervals."""

import torch

DEFAULT_TOTAL = 262144


def floor_frequencies(probabilities, total=DEFAULT_TOTAL, *, validate=True):
    """Canonical float64 normalization; optionally defer checks to a device coder."""
    if probabilities.device.type == "mps":
        raise ValueError("target-interval float64 quantization requires CPU or CUDA")
    if probabilities.ndim != 2 or not 1 <= probabilities.shape[1] < total:
        raise ValueError("expected a 2D alphabet smaller than the frequency total")
    probabilities = probabilities.to(torch.float64)
    if validate and (
        not torch.isfinite(probabilities).all() or (probabilities < 0).any()
    ):
        raise ValueError("probabilities must be finite and nonnegative")
    mass = probabilities.sum(dim=1, keepdim=True)
    if validate and (mass <= 0).any():
        raise ValueError("probability rows must have positive mass")
    normalized = probabilities / mass
    frequencies = torch.floor(normalized * (total - probabilities.shape[1])).to(torch.int64) + 1
    return frequencies, normalized


def target_intervals_from_probs_tensor(
    probabilities, targets, total=DEFAULT_TOTAL, *, validate=True
):
    frequencies, normalized = floor_frequencies(
        probabilities, total, validate=validate
    )
    targets = torch.as_tensor(targets, device=probabilities.device, dtype=torch.long)
    if targets.ndim != 1 or targets.numel() != probabilities.shape[0]:
        raise ValueError("one target index is required per probability row")
    if validate and (
        (targets < 0).any() or (targets >= probabilities.shape[1]).any()
    ):
        raise ValueError("target outside alphabet")
    rows = torch.arange(probabilities.shape[0], device=probabilities.device)
    cumulative = frequencies.cumsum(dim=1)
    highs = cumulative[rows, targets]
    lows = highs - frequencies[rows, targets]
    return lows, highs, cumulative[:, -1], normalized[rows, targets]
