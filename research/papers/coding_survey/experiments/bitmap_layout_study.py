"""Reproduce the manuscript's controlled global-mask size comparison.

The universe and cardinality are shared across cases.  Counts for the proposed
layouts are exact under their stated coding rules; Roaring uses the installed
pyroaring serializer, as the text archive does.
"""

from pyroaring import BitMap


UNIVERSE = 65_536
CARDINALITY = 462


def gamma_bits(value: int) -> int:
    if value < 1:
        raise ValueError("Elias gamma codes require a positive integer")
    return 2 * value.bit_length() - 1


def gap_bits(ids: list[int]) -> int:
    previous = -1
    total = 0
    for token_id in ids:
        total += gamma_bits(token_id - previous)
        previous = token_id
    return total


def run_bits(ids: list[int]) -> int:
    """Gamma(zero run + 1), gamma(one run); terminal zeros are implied."""
    total = 0
    cursor = 0
    index = 0
    while index < len(ids):
        start = ids[index]
        end = start + 1
        index += 1
        while index < len(ids) and ids[index] == end:
            end += 1
            index += 1
        total += gamma_bits(start - cursor + 1) + gamma_bits(end - start)
        cursor = end
    return total


def rows() -> list[tuple[str, int, int, int, int, int]]:
    spread = [(2 * index + 1) * UNIVERSE // (2 * CARDINALITY)
              for index in range(CARDINALITY)]
    clustered = list(range(1_000, 1_000 + CARDINALITY))
    return [
        (name, UNIVERSE, CARDINALITY * 16, gap_bits(ids), run_bits(ids),
         8 * len(BitMap(ids).serialize()))
        for name, ids in (("spread", spread), ("clustered", clustered))
    ]


if __name__ == "__main__":
    print("case dense_bits fixed16_bits gamma_gap_bits gamma_run_bits roaring_bits")
    for row in rows():
        print(*row)
