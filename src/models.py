"""Small local next-symbol model definitions for the model survey.

This module contains model interfaces and implementations only. Training and
scoring helpers live in :mod:`src.predictors`.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import math
from typing import Protocol, Sequence

import torch
from torch import Tensor, nn


class NextSymbolPredictor(Protocol):
    """Causal predictor interface shared by training and coding helpers."""

    vocabulary_size: int
    context_length: int

    def logits(self, contexts: Sequence[Sequence[int]]) -> Tensor:
        """Return one unnormalized score vector per (possibly short) context."""


def validate_symbols(symbols: Sequence[int], vocabulary_size: int) -> None:
    """Require integer symbols within the model's fixed vocabulary."""
    if vocabulary_size < 1:
        raise ValueError("vocabulary_size must be positive")
    if any(not isinstance(symbol, int) or not 0 <= symbol < vocabulary_size for symbol in symbols):
        raise ValueError("symbols must be integer IDs inside the vocabulary")


class NGramPredictor:
    """Add-one-smoothed byte or token bigram/trigram predictor."""

    def __init__(self, vocabulary_size: int, *, order: int) -> None:
        if order not in {2, 3}:
            raise ValueError("order must be 2 or 3")
        if vocabulary_size < 1:
            raise ValueError("vocabulary_size must be positive")
        self.vocabulary_size = vocabulary_size
        self.order = order
        self.context_length = order - 1
        self._counts: dict[tuple[int, ...], Counter[int]] = defaultdict(Counter)

    def logits(self, contexts: Sequence[Sequence[int]]) -> Tensor:
        result = torch.empty((len(contexts), self.vocabulary_size), dtype=torch.float32)
        for row, context in enumerate(contexts):
            validate_symbols(context, self.vocabulary_size)
            usable = tuple(context[-self.context_length:])
            while usable not in self._counts and usable:
                usable = usable[1:]
            counts = self._counts.get(usable, Counter())
            result[row] = torch.tensor(
                [math.log(counts.get(symbol, 0) + 1) for symbol in range(self.vocabulary_size)]
            )
        return result


class WindowModel(nn.Module):
    """Shared validation and left-padding for fixed-window neural models."""

    def __init__(self, vocabulary_size: int, context_length: int) -> None:
        super().__init__()
        if vocabulary_size < 1 or context_length < 1:
            raise ValueError("vocabulary_size and context_length must be positive")
        self.vocabulary_size = vocabulary_size
        self.context_length = context_length
        self.padding_idx = vocabulary_size

    def contexts_tensor(self, contexts: Sequence[Sequence[int]], device: torch.device) -> Tensor:
        """Convert variable-length causal contexts to padded model input."""
        rows: list[list[int]] = []
        for context in contexts:
            validate_symbols(context, self.vocabulary_size)
            tail = list(context[-self.context_length:])
            rows.append([self.padding_idx] * (self.context_length - len(tail)) + tail)
        return torch.tensor(rows, dtype=torch.long, device=device)

    def logits(self, contexts: Sequence[Sequence[int]]) -> Tensor:
        device = next(self.parameters()).device
        return self.forward(self.contexts_tensor(contexts, device))


class WindowNPLM(WindowModel):
    """Bengio-style fixed-window neural probabilistic language model."""

    def __init__(self, vocabulary_size: int, *, context_length: int, embedding_dim: int = 64, hidden_dim: int = 128) -> None:
        super().__init__(vocabulary_size, context_length)
        self.embedding = nn.Embedding(vocabulary_size + 1, embedding_dim, padding_idx=self.padding_idx)
        self.network = nn.Sequential(
            nn.Linear(context_length * embedding_dim, hidden_dim), nn.GELU(),
            nn.Linear(hidden_dim, vocabulary_size),
        )

    def forward(self, contexts: Tensor) -> Tensor:
        return self.network(self.embedding(contexts).flatten(1))


class TinyTransformerLM(WindowModel):
    """A dense causal Transformer encoder over one fixed prediction window."""

    def __init__(self, vocabulary_size: int, *, context_length: int, embedding_dim: int = 64, hidden_dim: int = 128, layers: int = 2, heads: int = 4) -> None:
        super().__init__(vocabulary_size, context_length)
        if embedding_dim % heads:
            raise ValueError("embedding_dim must be divisible by heads")
        self.embedding = nn.Embedding(vocabulary_size + 1, embedding_dim, padding_idx=self.padding_idx)
        self.position = nn.Embedding(context_length, embedding_dim)
        layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=heads, dim_feedforward=hidden_dim, dropout=0.0, batch_first=True, activation="gelu")
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.output = nn.Linear(embedding_dim, vocabulary_size)

    def forward(self, contexts: Tensor) -> Tensor:
        length = contexts.shape[1]
        positions = torch.arange(length, device=contexts.device)
        hidden = self.embedding(contexts) + self.position(positions).unsqueeze(0)
        causal_mask = torch.triu(torch.ones(length, length, device=contexts.device, dtype=torch.bool), diagonal=1)
        return self.output(self.encoder(hidden, mask=causal_mask)[:, -1])


class TinyRecurrentLM(WindowModel):
    """A compact GRU language model, providing the recurrent survey point."""

    def __init__(self, vocabulary_size: int, *, context_length: int, embedding_dim: int = 64, hidden_dim: int = 128, layers: int = 2) -> None:
        super().__init__(vocabulary_size, context_length)
        self.embedding = nn.Embedding(vocabulary_size + 1, embedding_dim, padding_idx=self.padding_idx)
        self.recurrent = nn.GRU(embedding_dim, hidden_dim, num_layers=layers, batch_first=True)
        self.output = nn.Linear(hidden_dim, vocabulary_size)

    def forward(self, contexts: Tensor) -> Tensor:
        states, _ = self.recurrent(self.embedding(contexts))
        return self.output(states[:, -1])
