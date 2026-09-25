import numpy as np
import math
from typing import List

def build_cumul(prob_vec: np.ndarray, total: int = 262144) -> np.ndarray:
    """
    Turn a probability vector into a cumulative-frequency array for arithmetic coding.
    Ensures every symbol ≥1 and freq.sum() == total.
    """
    alphabet_size = prob_vec.size

    # Step 1: allocate counts proportional to prob_vec
    freq = (prob_vec * (total - alphabet_size)).astype(np.int64)

    # Step 2: ensure every symbol ≥1
    freq += 1

    # Step 3: compute diff and adjust
    diff = total - freq.sum()
    if diff != 0:
        # adjust the symbol with largest probability first
        # np.argsort(-prob_vec) gives descending probabilities
        indices = np.argsort(-prob_vec)
        idx = 0
        while diff != 0:
            i = indices[idx % alphabet_size]  # wrap around if needed
            if diff > 0:
                freq[i] += 1
                diff -= 1
            else:  # diff < 0
                if freq[i] > 1:
                    freq[i] -= 1
                    diff += 1
            idx += 1

    # Safety check
    assert freq.sum() == total, f"freq.sum={freq.sum()} != total={total}"
    assert np.all(freq >= 1)

    # Build cumulative array
    cumul = np.empty(alphabet_size + 1, dtype=np.int64)
    cumul[0] = 0
    cumul[1:] = np.cumsum(freq)
    return cumul


# ------------------------------------------------------------------
# PMATIC utilities
# ------------------------------------------------------------------

def binary_cumul(p1: float, total: int = 262144) -> np.ndarray:
    """
    Cumulative table for Bernoulli bit:
      symbol 0 has prob 1-p1
      symbol 1 has prob p1
    """
    p1 = float(np.clip(p1, 1e-12, 1.0 - 1e-12))
    return build_cumul(np.array([1.0 - p1, p1], dtype=np.float64), total=total)

def token_to_bits(token_idx: int, bit_width: int) -> List[int]:
    return [(token_idx >> shift) & 1 for shift in range(bit_width - 1, -1, -1)]

def bits_to_token(bits: List[int]) -> int:
    x = 0
    for b in bits:
        x = (x << 1) | int(b)
    return x

def next_power_of_two(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()

def pad_probs_to_power_of_two(probs: np.ndarray, alphabet_size: int) -> np.ndarray:
    """
    Pads probs to a power-of-two PMATIC alphabet.

    Real symbols keep their original probabilities.
    Dummy symbols get probability 0.
    """
    probs = np.asarray(probs, dtype=np.float64)

    if probs.size != alphabet_size:
        raise ValueError(f"Expected probs of size {alphabet_size}, got {probs.size}")

    padded_size = next_power_of_two(alphabet_size)

    padded = np.zeros(padded_size, dtype=np.float64)
    padded[:alphabet_size] = probs

    s = padded.sum()
    if s <= 0:
        raise ValueError("Probability vector must have positive total mass")

    return padded / s

def conditional_bit_prob(
    probs: np.ndarray,
    prefix_bits: List[int],
    bit_width: int,
) -> float:
    """
    Computes P[next bit = 1 | prefix].

    Works safely with padded zero-probability dummy symbols.
    """
    probs = np.asarray(probs, dtype=np.float64)

    j = len(prefix_bits)
    total_mass = 0.0
    one_mass = 0.0

    for tok, p in enumerate(probs):
        if p == 0:
            continue

        bits = token_to_bits(tok, bit_width)

        if bits[:j] == prefix_bits:
            total_mass += p
            if bits[j] == 1:
                one_mass += p

    if total_mass <= 0:
        # This path should be unreachable if the arithmetic stream is valid.
        # Return neutral probability to avoid crashing before the true issue appears.
        return 0.5

    return one_mass / total_mass

def pmatic_quantize_encoder(p: float, delta: float, r: float):
    """
    Returns:
      helper_bit, quantized_probability

    Bins have width 2r.
    If p is safely inside a bin, helper=0 and use bin center.
    If p is near a bin boundary, helper=1 and use the nearest boundary.
    """
    if not (r > 2 * delta):
        raise ValueError("PMATIC requires r > 2 * delta")

    p = float(np.clip(p, 0.0, 1.0))
    width = 2.0 * r
    m = round(1.0 / width)

    if abs(m * width - 1.0) > 1e-9:
        raise ValueError("PMATIC expects r = 1/(2m) for integer m")

    # Find bin k in 0-based indexing.
    k = min(int(p / width), m - 1)
    left = k * width
    right = (k + 1) * width

    # PMATIC delta-interior.
    if k == 0:
        in_delta_interior = p <= right - delta
    elif k == m - 1:
        in_delta_interior = p >= left + delta
    else:
        in_delta_interior = (p >= left + delta) and (p <= right - delta)

    if in_delta_interior:
        center = left + r
        return 0, center

    # Near boundary: use nearest internal boundary.
    boundaries = [width * kk for kk in range(1, m)]
    boundary = min(boundaries, key=lambda b: abs(b - p))
    return 1, boundary


def pmatic_quantize_decoder(q: float, helper_bit: int, delta: float, r: float):
    """
    Decoder-side quantization.
    If helper=0, use center of q's bin.
    If helper=1, use nearest internal boundary.
    """
    if not (r > 2 * delta):
        raise ValueError("PMATIC requires r > 2 * delta")

    q = float(np.clip(q, 0.0, 1.0))
    width = 2.0 * r
    m = round(1.0 / width)

    if helper_bit == 0:
        k = min(int(q / width), m - 1)
        return k * width + r

    boundaries = [width * kk for kk in range(1, m)]
    return min(boundaries, key=lambda b: abs(b - q))


def max_conditional_mismatch(p, q, alphabet_size):
    padded_p = pad_probs_to_power_of_two(p, alphabet_size)
    padded_q = pad_probs_to_power_of_two(q, alphabet_size)

    padded_size = next_power_of_two(alphabet_size)
    bit_width = math.ceil(math.log2(padded_size))

    max_diff = 0.0

    prefixes = [[]]
    for depth in range(bit_width):
        new_prefixes = []
        for prefix in prefixes:
            pp = conditional_bit_prob(padded_p, prefix, bit_width)
            qq = conditional_bit_prob(padded_q, prefix, bit_width)
            max_diff = max(max_diff, abs(pp - qq))

            new_prefixes.append(prefix + [0])
            new_prefixes.append(prefix + [1])

        prefixes = new_prefixes

    return max_diff


def make_safe_decoder_probs(p, alphabet_size, delta, initial_scale=1e-6):
    """
    Creates perturbed decoder probabilities whose induced conditional bit
    probabilities stay within delta.
    """
    scale = initial_scale

    for _ in range(30):
        noise = np.random.normal(scale=scale, size=alphabet_size)
        q = p + noise
        q = np.clip(q, 1e-12, None)
        q /= q.sum()

        if max_conditional_mismatch(p, q, alphabet_size) < delta:
            return q

        scale *= 0.5

    # Fallback: exact probabilities
    return p.copy()


def choose_pmatic_r(delta: float, safety: float = 1.01) -> float:
    """
    Choose r adaptively for PMATIC.

    Requirements:
      r > 2*delta
      r = 1/(2m), m integer

    Uses the smallest valid r up to the discrete 1/(2m) grid.
    """
    if not (0 < delta < 0.5):
        raise ValueError("delta must be in (0, 0.5)")

    target = 2.0 * delta * safety

    # Need 1/(2m) > target  =>  m < 1/(2*target)
    m_max = math.floor((1.0 / (2.0 * target)) - 1e-12)

    if m_max < 1:
        raise ValueError(
            f"delta={delta} is too large; cannot choose r=1/(2m) with r > 2*delta"
        )

    r = 1.0 / (2.0 * m_max)

    if not (r > 2.0 * delta):
        raise AssertionError(f"Internal error: r={r} does not satisfy r > 2*delta")

    return r