"""Device-side floor-count and exact fixed-total target-interval quantization."""

import torch

DEFAULT_TOTAL = 262144
FLOOR_INTERVAL_QUANTIZER = "floor_float64_v1"
EXACT_INTERVAL_QUANTIZER = "fixed_total_stable_v2"


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


def exact_fixed_frequencies(probabilities, total=DEFAULT_TOTAL, *, validate=True):
    """Apply the CUDA-v4 deterministic fixed-total rule on a torch device.

    Model probabilities are widened to float32 because the full-logits path
    explicitly performs that conversion before NumPy quantization. Equal
    probabilities retain token-column order through a stable descending sort;
    this explicit tie contract is versioned separately from legacy MSAC v1.
    """
    if probabilities.ndim != 2 or not 1 < probabilities.shape[1] < total:
        raise ValueError("expected a 2D alphabet smaller than the frequency total")
    probabilities = probabilities.to(torch.float32)
    if validate and (
        not torch.isfinite(probabilities).all() or (probabilities < 0).any()
    ):
        raise ValueError("probabilities must be finite and nonnegative")
    # CUDA v4 defines normalization as float64 accumulation over float32 model
    # probabilities. This avoids float32 row-sum boundaries while retaining a
    # source-dtype-independent archive contract.
    probabilities64 = probabilities.to(torch.float64)
    mass = probabilities64.sum(dim=1, keepdim=True)
    if validate and (mass <= 0).any():
        raise ValueError("probability rows must have positive mass")

    # The archive contract starts by normalizing float32 values. In particular,
    # BF16 softmax rows copied through the host path need not retain a unit sum,
    # and the CUDA v4 kernel accepts the same arbitrary positive row mass.
    probabilities = probabilities64 / mass

    alphabet = probabilities.shape[1]
    frequencies = torch.floor(probabilities * (total - alphabet)).to(torch.int64) + 1
    difference = total - frequencies.sum(dim=1)
    if validate and (difference < 0).any():
        raise ValueError("fixed-total normalization produced an overfull row")
    order = torch.argsort(probabilities, dim=1, descending=True, stable=True)

    positive = difference.clamp_min(0)
    passes = torch.div(positive, alphabet, rounding_mode="floor")
    remainder = positive.remainder(alphabet)
    frequencies = frequencies + passes[:, None]
    ordered_bonus = (
        torch.arange(alphabet, device=probabilities.device)[None, :]
        < remainder[:, None]
    ).to(torch.int64)
    bonus = torch.zeros_like(frequencies)
    bonus.scatter_(1, order, ordered_bonus)
    frequencies = frequencies + bonus

    return frequencies, probabilities


def exact_fixed_cdfs(probabilities, total=DEFAULT_TOTAL, *, validate=True):
    """Return exact fixed-total CDF rows, including their leading zero."""
    frequencies, normalized = exact_fixed_frequencies(
        probabilities, total, validate=validate
    )
    zeros = torch.zeros(
        (frequencies.shape[0], 1),
        dtype=torch.int64,
        device=frequencies.device,
    )
    return torch.cat((zeros, frequencies.cumsum(dim=1)), dim=1), normalized


def target_intervals_from_probs_exact(
    probabilities, targets, total=DEFAULT_TOTAL, *, validate=True
):
    """Construct CUDA-v4 fixed-total target intervals."""
    frequencies, normalized = exact_fixed_frequencies(
        probabilities, total, validate=validate
    )
    targets = torch.as_tensor(
        targets, device=probabilities.device, dtype=torch.long
    )
    if targets.ndim != 1 or targets.numel() != probabilities.shape[0]:
        raise ValueError("one target index is required per probability row")
    if validate and (
        (targets < 0).any() or (targets >= probabilities.shape[1]).any()
    ):
        raise ValueError("target outside alphabet")
    cumulative = frequencies.cumsum(dim=1)
    rows = torch.arange(probabilities.shape[0], device=probabilities.device)
    highs = cumulative[rows, targets]
    lows = highs - frequencies[rows, targets]
    totals = torch.full_like(lows, total)
    return lows, highs, totals, normalized[rows, targets]
