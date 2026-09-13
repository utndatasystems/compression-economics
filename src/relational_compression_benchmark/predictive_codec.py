"""Decoder-verified arithmetic coding for fixed next-symbol predictors."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np
import torch

from src.models import NextSymbolPredictor
from src.encoding import LLMCompressor, LLMDecompressor


@dataclass(frozen=True)
class PredictiveCompressionResult:
    """A shared-model, payload-only predictive coding measurement."""

    payload_bits: int
    source_symbols: int
    encode_seconds: float
    decode_seconds: float
    roundtrip_valid: bool

    @property
    def payload_bytes(self) -> int:
        return math.ceil(self.payload_bits / 8)


def _probabilities(predictor: NextSymbolPredictor, context: Sequence[int]) -> np.ndarray:
    with torch.inference_mode():
        logits = predictor.logits([context])[0]
        return torch.softmax(logits.float(), dim=-1).cpu().numpy().astype(np.float64)


def compress_and_verify(
    predictor: NextSymbolPredictor, symbols: Sequence[int], *, total: int = 262_144
) -> PredictiveCompressionResult:
    """Arithmetic-code and decode one sequence using reconstructed history only.

    The predictor is a shared decoder dependency. Its weights/counts and the
    symbol alphabet are excluded from payload_bits; callers must report those
    separately for self-contained archive accounting.
    """
    if not symbols:
        raise ValueError("cannot compress an empty sequence")
    if predictor.vocabulary_size > total:
        raise ValueError("CDF total must be at least the vocabulary size")

    encoder = LLMCompressor(algorithm="AC", alphabet_size=predictor.vocabulary_size, total=total)
    started = time.perf_counter()
    for index, symbol in enumerate(symbols):
        context = symbols[max(0, index - predictor.context_length):index]
        encoder.next_token(symbol, _probabilities(predictor, context))
    bits = encoder.compress()
    encode_seconds = time.perf_counter() - started

    decoder = LLMDecompressor(
        bits, algorithm="AC", alphabet_size=predictor.vocabulary_size, total=total
    )
    decoded: list[int] = []
    started = time.perf_counter()
    for _ in symbols:
        context = decoded[-predictor.context_length:]
        decoded.append(decoder.decompress(_probabilities(predictor, context)))
    decode_seconds = time.perf_counter() - started
    return PredictiveCompressionResult(
        payload_bits=len(bits), source_symbols=len(symbols),
        encode_seconds=encode_seconds, decode_seconds=decode_seconds,
        roundtrip_valid=decoded == list(symbols),
    )
