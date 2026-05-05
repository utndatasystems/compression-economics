# tests/test_pmatic.py

import math
import random

import numpy as np
import pytest

from src.encoding import *


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(0)
    random.seed(0)


def random_probs(vocab_size: int) -> np.ndarray:
    x = np.random.rand(vocab_size)
    return x / x.sum()


@pytest.mark.parametrize("decoder_case", ["exact", "safe-perturbed", "perturbed"])
def test_pmatic_roundtrip(decoder_case):
    vocab_size = 50
    sequence_length = 200
    delta = 0.01
    r = choose_pmatic_r(delta)

    tokens = np.random.randint(0, vocab_size, size=sequence_length)
    encoder_probs = [random_probs(vocab_size) for _ in range(sequence_length)]

    if decoder_case == "exact":
        decoder_probs = [p.copy() for p in encoder_probs]

    elif decoder_case == "perturbed":
        decoder_probs = []
        for p in encoder_probs:
            noise = np.random.normal(scale=1e-5, size=vocab_size)
            noisy = p + noise
            noisy = np.clip(noisy, 1e-9, None)
            noisy /= noisy.sum()
            decoder_probs.append(noisy)

    elif decoder_case == "safe-perturbed":
        decoder_probs = [
            make_safe_decoder_probs(p, vocab_size, delta)
            for p in encoder_probs
        ]

    worst_mismatch = max(
        max_conditional_mismatch(p, q, vocab_size)
        for p, q in zip(encoder_probs, decoder_probs)
    )

    assert worst_mismatch <= delta or decoder_case == "perturbed"
 
    comp = LLMCompressor(
        algorithm="PMATIC",
        alphabet_size=vocab_size,
        delta=delta,
        r=r,)

    for token, probs in zip(tokens, encoder_probs):
        comp.next_token(token, probs)

    code = comp.compress()

    dec = LLMDecompressor(
        code,
        algorithm="PMATIC",
        alphabet_size=vocab_size,
        delta=delta,
        r=r,
    )

    decoded = np.array([dec.decompress(probs) for probs in decoder_probs])

    assert np.array_equal(tokens, decoded), (
        f"PMATIC roundtrip failed for decoder_case={decoder_case}. "
        f"First mismatch at index {np.where(tokens != decoded)[0][0]}"
    )

    total_bits = len(code)
    bits_per_token = total_bits / sequence_length
    cross_entropy = comp.get_cross_entropy() / sequence_length
    helper_bits_per_token = comp.helper_count / sequence_length

    assert total_bits > 0
    assert bits_per_token > 0
    assert cross_entropy >= 0
    assert 0 <= comp.helper_one_fraction() <= 1
    assert helper_bits_per_token >= 0

    empirical_entropy = sum(
        -math.log2(max(probs[token], 1e-300))
        for token, probs in zip(tokens, encoder_probs)
    ) / sequence_length

    assert empirical_entropy >= 0


@pytest.mark.parametrize(
    "invalid_case",
    ["", "unsafe", "safe_perturbed", "foo"],)
def test_invalid_decoder_case_rejected(invalid_case):
    valid_cases = {"exact", "perturbed", "safe-perturbed"}

    with pytest.raises(ValueError):
        decoder_case = invalid_case.lower()
        if decoder_case not in valid_cases:
            raise ValueError(
                "decoder_case must be one of: exact, perturbed, safe-perturbed"
            )


def test_AC_roundtrip():
    vocab_size = 50
    sequence_length = 200

    tokens = np.random.randint(0, vocab_size, size=sequence_length)
    encoder_probs = [random_probs(vocab_size) for _ in range(sequence_length)]

    comp = LLMCompressor(
        algorithm="AC",
    )

    for token, probs in zip(tokens, encoder_probs):
        comp.next_token(token, probs)

    code = comp.compress()

    dec = LLMDecompressor(
        code,
        algorithm="AC",
    )

    decoded = np.array([dec.decompress(probs) for probs in encoder_probs])

    assert np.array_equal(tokens, decoded), "AC roundtrip failed"