import random
import numpy as np
from string import printable

#from llm_compression.compression import (
#    LLMCompressor,
#    LLMDecompressor,
#)

from .src.encoding import LLMCompressor, LLMDecompressor

def test_compress_decompress_roundtrip():
    alphabet_size = len(printable)
    text_len = 1_000

    text = [
        random.randint(0, alphabet_size - 1)
        for _ in range(text_len)
    ]

    prob_tables = []
    for _ in range(text_len):
        probs = np.random.rand(alphabet_size)
        prob_tables.append(probs / probs.sum())

    compressor = LLMCompressor()
    for token, probs in zip(text, prob_tables):
        compressor.next_token(token, probs)

    code = compressor.compress()

    decompressor = LLMDecompressor(code)
    for i, probs in enumerate(prob_tables):
        decoded = decompressor.decompress(probs)
        assert decoded == text[i]

if __name__ == "__main__":
    test_compress_decompress_roundtrip()
    print("Test passed!")