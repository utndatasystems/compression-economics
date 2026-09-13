"""CIDR model-survey specifications, construction, training, and scoring."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Sequence

import torch

from src.models import (
    NextSymbolPredictor,
    NGramPredictor,
    TinyRecurrentLM,
    TinyTransformerLM,
    WindowModel,
    WindowNPLM,
    validate_symbols,
)
from torch.nn import functional as F


SymbolKind = Literal["byte", "token"]


@dataclass(frozen=True)
class ModelSpec:
    """A reproducible model choice in the CIDR survey."""

    name: str
    family: Literal["ngram", "nplm", "transformer", "recurrent"]
    symbol_kind: SymbolKind
    context_length: int
    order: int | None = None
    embedding_dim: int = 64
    hidden_dim: int = 128
    layers: int = 2
    heads: int = 4

    def __post_init__(self) -> None:
        if self.context_length < 1:
            raise ValueError("context_length must be positive")
        if self.family == "ngram" and self.order not in {2, 3}:
            raise ValueError("n-gram models require order 2 or 3")
        if self.family != "ngram" and self.order is not None:
            raise ValueError("only n-gram models accept order")
        if self.symbol_kind == "byte" and self.family != "ngram":
            # Byte neural models can be added later, but keeping this survey
            # small avoids silently comparing unlike vocabularies.
            raise ValueError("neural survey models currently use token symbols")


def survey_model_specs() -> tuple[ModelSpec, ...]:
    """Return the requested compact model-survey matrix.

    N-grams are included for both byte and token streams.  The neural models
    use token streams, and their names encode every architecture choice that
    affects a trained checkpoint.
    """
    return (
        ModelSpec("byte-bigram", "ngram", "byte", 1, order=2),
        ModelSpec("byte-trigram", "ngram", "byte", 2, order=3),
        ModelSpec("token-bigram", "ngram", "token", 1, order=2),
        ModelSpec("token-trigram", "ngram", "token", 2, order=3),
        ModelSpec("token-nplm-w8", "nplm", "token", 8),
        ModelSpec("token-nplm-w32", "nplm", "token", 32),
        ModelSpec("token-nplm-w128", "nplm", "token", 128),
        ModelSpec("token-tiny-transformer-w128", "transformer", "token", 128),
        ModelSpec("token-tiny-gru-w128", "recurrent", "token", 128),
    )


def build_predictor(spec: ModelSpec, vocabulary_size: int) -> NextSymbolPredictor:
    """Construct an untrained predictor from a fully recorded survey spec."""
    if spec.symbol_kind == "byte" and vocabulary_size != 256:
        raise ValueError("byte predictors require vocabulary_size=256")
    if spec.family == "ngram":
        return NGramPredictor(vocabulary_size, order=spec.order or 2)
    kwargs = dict(context_length=spec.context_length, embedding_dim=spec.embedding_dim, hidden_dim=spec.hidden_dim)
    if spec.family == "nplm":
        return WindowNPLM(vocabulary_size, **kwargs)
    if spec.family == "transformer":
        return TinyTransformerLM(vocabulary_size, layers=spec.layers, heads=spec.heads, **kwargs)
    return TinyRecurrentLM(vocabulary_size, layers=spec.layers, **kwargs)


def train_ngram_predictor(model: NGramPredictor, symbols: Sequence[int]) -> NGramPredictor:
    """Fit count tables for an n-gram model from a symbol sequence."""
    validate_symbols(symbols, model.vocabulary_size)
    model._counts.clear()
    for index, target in enumerate(symbols):
        for width in range(0, min(index, model.context_length) + 1):
            context = tuple(symbols[index - width:index]) if width else ()
            model._counts[context][target] += 1
    return model

def train_neural_predictor(model: WindowModel, symbols: Sequence[int], *, epochs: int = 1, batch_size: int = 128, learning_rate: float = 3e-4, seed: int = 0) -> list[float]:
    """Teacher-force a neural predictor and return mean loss for each epoch.

    Training uses only preceding symbols; left padding makes predictions for the
    first symbols well-defined.  Callers must keep tuning and evaluation data
    separate themselves.
    """
    validate_symbols(symbols, model.vocabulary_size)
    if len(symbols) < 2:
        raise ValueError("at least two symbols are required for training")
    if epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("epochs, batch_size, and learning_rate must be positive")
    torch.manual_seed(seed)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    targets = torch.tensor(symbols[1:], dtype=torch.long)
    contexts = [symbols[max(0, index - model.context_length):index] for index in range(1, len(symbols))]
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        total, count = 0.0, 0
        for start in range(0, len(targets), batch_size):
            stop = min(start + batch_size, len(targets))
            inputs = model.contexts_tensor(contexts[start:stop], device)
            loss = F.cross_entropy(model(inputs), targets[start:stop].to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item() * (stop - start)
            count += stop - start
        losses.append(total / count)
    model.eval()
    return losses


def bits_per_symbol(predictor: NextSymbolPredictor, symbols: Sequence[int]) -> float:
    """Return causal cross-entropy in bits/symbol for an evaluation sequence."""
    validate_symbols(symbols, predictor.vocabulary_size)
    if not symbols:
        raise ValueError("cannot score an empty sequence")
    contexts = [symbols[max(0, index - predictor.context_length):index] for index in range(len(symbols))]
    with torch.inference_mode():
        log_probs = F.log_softmax(predictor.logits(contexts), dim=-1)
        target = torch.tensor(symbols, dtype=torch.long, device=log_probs.device)
        return float((-log_probs.gather(1, target[:, None]).mean() / math.log(2)).cpu())
