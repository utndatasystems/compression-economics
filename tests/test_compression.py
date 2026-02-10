import random
import numpy as np
import pytest
from string import printable

from src.encoding import LLMCompressor, LLMDecompressor


@pytest.fixture(scope="session")
def alphabet_size():
    return len(printable)


@pytest.fixture
def rng():
    """
    Seed both Python and NumPy random number generators.
    Ensures deterministic test behaviour / reproductbility.
    """
    random.seed(42)
    np.random.seed(42)


@pytest.fixture
def random_input(rng, alphabet_size):
    """
    Generate a random token sequence and corresponding probability tables.

    Each token is an integer index into the shared alphabet, and each
    probability table represents a valid categorical distribution over
    that alphabet for a single timestep.

    :param rng: Ensures deterministic randomness
    :param alphabet_size: Size of the token vocabulary
    :return: Tuple of (text, prob_tables)
    """
    
    text_len = 1_000
    text = [random.randint(0, alphabet_size - 1) for _ in range(text_len)]

    prob_tables = [(probs := np.random.rand(alphabet_size)) / probs.sum() for _ in range(text_len)]

    return text, prob_tables


@pytest.mark.parametrize("compressor_cls,decompressor_cls", [
    (LLMCompressor, LLMDecompressor),])

def test_compress_decompress_roundtrip(
    random_input,
    compressor_cls,
    decompressor_cls,):
    """
    Verify lossless roundtrip compression and decompression.

    Given a sequence of tokens and corresponding probability tables,
    decompression must exactly reproduce the original token sequence
    after compression.

    :param random_input: Randomly generated token sequence and probabilities
    :param compressor_cls: Compressor implementation under test
    :param decompressor_cls: Matching decompressor implementation
    """
    text, prob_tables = random_input

    compressor = compressor_cls()
    for token, probs in zip(text, prob_tables):
        compressor.next_token(token, probs)

    code = compressor.compress()

    decompressor = decompressor_cls(code)
    for i, probs in enumerate(prob_tables):
        decoded = decompressor.decompress(probs)
        assert decoded == text[i]
