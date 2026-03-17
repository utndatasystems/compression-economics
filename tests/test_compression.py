import random
import numpy as np
import pytest
from string import printable
from tqdm import tqdm
from argparse import Namespace

from src.encoding import LLMCompressor, LLMDecompressor
from src.utils import save_global_mask_file, load_global_mask_file


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


@pytest.fixture
def zipf_input(rng, alphabet_size):
    """
    Generate a random token sequence and corresponding probability tables following a Zipf distribution.

    Each token is an integer index into the shared alphabet, and each
    probability table represents a valid categorical distribution over
    that alphabet for a single timestep.

    :param rng: Ensures deterministic randomness
    :param alphabet_size: Size of the token vocabulary
    :return: Tuple of (text, prob_tables)
    """

    text_len = 1_000
    ranks = np.arange(1, alphabet_size + 1)
    zipf_probs = 1 / ranks
    zipf_probs /= zipf_probs.sum()

    text = [np.random.choice(alphabet_size, p=zipf_probs) for _ in range(text_len)]
    prob_tables = [zipf_probs for _ in range(text_len)]

    return text, prob_tables


@pytest.fixture
def input_fixture(request):
    """
    Dispatch fixture that selects an input distribution
    based on the parametrized fixture name.
    """
    return request.getfixturevalue(request.param)


def distort_prob_table(prob_table, noise_level=0.001):
    """
    Distort a probability table by adding random noise and re-normalizing. 
    Used to test robustness of decompression against small perturbations in the probability tables.

    :param prob_table: Original probability table (numpy array)
    :param noise_level: Magnitude of noise to add
    :return: Distorted probability table
    """
    noise = np.random.rand(*prob_table.shape) * noise_level
    distorted = prob_table + noise
    return distorted / distorted.sum()


@pytest.fixture
def random_code(rng, random_input, compressor_cls):
    """
    Generate only compressed code from random input using the specified compressor class.
    """
    text, prob_tables = random_input

    compressor = compressor_cls()
    for token, probs in zip(text, prob_tables):
        compressor.next_token(token, probs)

    code = compressor.compress()
    return code


@pytest.mark.parametrize("compressor_cls,decompressor_cls", [(LLMCompressor, LLMDecompressor),])
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


def test_global_mask_header_roundtrip(tmp_path):
    args = Namespace(
        input_path="data/text8",
        output_path=str(tmp_path / "roundtrip.bin"),
        model_name="Qwen/Qwen3-0.6B",
        context_length=1000,
        first_n_tokens=1025,
        retain_tokens=100,
        use_kv_cache=True,
        batch_size=4,
        reduce_tokens=True,
        engine="vllm",
        encoding="AC",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.75,
        lora_path=None,
    )

    save_global_mask_file(
        args=args,
        first_token=[1, 2, 3, 4],
        bit_string=[1, 0, 1, 1],
        bitmask_data=b"bitmap",
    )

    loaded_args = Namespace(
        input_path=str(tmp_path / "roundtrip.bin"),
        reduce_tokens=False,
        engine="transformer",
        encoding="bitpacked",
        tensor_parallel_size=8,
        gpu_memory_utilization=None,
        lora_path="adapter",
    )
    loaded_args, first_token, bit_string, bitmask_data = load_global_mask_file(loaded_args)

    assert first_token == [1, 2, 3, 4]
    assert bit_string == [1, 0, 1, 1]
    assert bitmask_data == b"bitmap"
    assert loaded_args.engine == "vllm"
    assert loaded_args.encoding == "AC"
    assert loaded_args.reduce_tokens is True
    assert loaded_args.tensor_parallel_size == 1
    assert loaded_args.gpu_memory_utilization == 0.75


@pytest.mark.skip(reason="Temporarily disabled for debugging")
@pytest.mark.parametrize("input_fixture",["random_input", "zipf_input"], indirect=True,)
@pytest.mark.parametrize("compressor_cls,decompressor_cls",[(LLMCompressor, LLMDecompressor)],)
def test_compress_decompress_brute_force(input_fixture, compressor_cls,decompressor_cls,):
    """
    Verify that decompression remains correct even when probability
    tables are slightly distorted, across different input distributions.
    """
    text, prob_tables = input_fixture

    compressor = compressor_cls()
    for token, probs in zip(text, prob_tables):
        compressor.next_token(token, probs)

    code = compressor.compress()

    decompressor = decompressor_cls(code)

    probs_distorted = [
        distort_prob_table(probs) for probs in prob_tables
    ]

    for i, probs in enumerate(tqdm(probs_distorted)):
        decoded = decompressor.decompress(probs)
        assert decoded == text[i], (
            f"Decoded token {decoded} does not match "
            f"original {text[i]} at index {i}"
        )


# TODO: Implement brute-force decompressor that tries all possible token sequences and probability tables to find a match for the given code.

"""
def generate_sequences(length, alphabet_size):
            if length == 0:
                yield []
            else:
                for token in range(alphabet_size):
                    for seq in generate_sequences(length - 1, alphabet_size):
                        yield [token] + seq

                        
       for candidate_text in generate_sequences(text_len, alphabet_size):
            compressor = LLMCompressor()
            for token, probs in zip(candidate_text, prob_tables):
                compressor.next_token(token, probs)
            candidate_code = compressor.compress()
            if candidate_code == self.code:
                return candidate_text

        raise ValueError("No matching token sequence found for the given code.") """