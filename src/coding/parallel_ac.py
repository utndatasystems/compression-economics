"""Compiled, GIL-free range encoding for independent MSAC streams."""

from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:  # The Python MSAC backend has no Numba dependency.
    njit = None


if njit is not None:
    @njit(cache=True, nogil=True)
    def encode_intervals_packed(
        lows: np.ndarray, highs: np.ndarray, total: int, state_bits: int,
        totals: np.ndarray = np.empty(0, dtype=np.int64),
    ) -> tuple[np.ndarray, int]:
        """Encode one stream, returning packed bytes and its unpadded bit count."""
        max_range = 1 << state_bits
        mask = max_range - 1
        top_mask = max_range >> 1
        second_mask = top_mask >> 1
        low = 0
        high = mask
        underflow = 0
        output = []
        current_byte = 0
        bit_position = 0
        bit_count = 0

        for i in range(lows.size):
            symbol_total = totals[i] if totals.size else total
            current_range = high - low + 1
            base_low = low
            low = base_low + lows[i] * current_range // symbol_total
            high = base_low + highs[i] * current_range // symbol_total - 1

            while ((low ^ high) & top_mask) == 0:
                bit = low >> (state_bits - 1)
                current_byte = (current_byte << 1) | bit
                bit_position += 1
                bit_count += 1
                if bit_position == 8:
                    output.append(current_byte)
                    current_byte = 0
                    bit_position = 0
                inverse = bit ^ 1
                for _ in range(underflow):
                    current_byte = (current_byte << 1) | inverse
                    bit_position += 1
                    bit_count += 1
                    if bit_position == 8:
                        output.append(current_byte)
                        current_byte = 0
                        bit_position = 0
                underflow = 0
                low = (low << 1) & mask
                high = ((high << 1) & mask) | 1

            while (low & ~high & second_mask) != 0:
                underflow += 1
                low = (low << 1) & (mask >> 1)
                high = ((high << 1) & (mask >> 1)) | top_mask | 1

        current_byte = (current_byte << 1) | 1
        bit_position += 1
        bit_count += 1
        if bit_position == 8:
            output.append(current_byte)
        else:
            output.append(current_byte << (8 - bit_position))
        return np.asarray(output, dtype=np.uint8), bit_count
else:
    encode_intervals_packed = None


def prepare_parallel_encoder(state_bits: int, total: int) -> None:
    """Reject unsupported arithmetic sizes and compile before timing a run."""
    if encode_intervals_packed is None:
        raise RuntimeError("parallel MSAC encoding requires numba>=0.64")
    if (1 << state_bits) * total > (1 << 63) - 1:
        raise ValueError("parallel MSAC interval product exceeds signed 64-bit arithmetic")
    encode_intervals_packed(
        np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64), total, state_bits
    )
