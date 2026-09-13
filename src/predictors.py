"""CIDR model-survey specifications, construction, training, and scoring."""

from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import math
from pathlib import Path
import pickle
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


def expand_context_specs(
    specs: Sequence[ModelSpec], context_lengths: Sequence[int] | None
) -> tuple[ModelSpec, ...]:
    """Expand each distinct neural architecture over requested contexts.

    N-gram context is fixed by its statistical order and is therefore retained
    exactly once. Existing neural specs that differ only in context are treated
    as one architecture template, preventing duplicate NPLM conditions.
    """
    if context_lengths is None:
        return tuple(specs)
    if not context_lengths or any(length < 1 for length in context_lengths):
        raise ValueError("context lengths must be a non-empty sequence of positive integers")
    if len(set(context_lengths)) != len(context_lengths):
        raise ValueError("context lengths must not contain duplicates")

    result: list[ModelSpec] = []
    seen_neural: set[tuple[object, ...]] = set()
    for spec in specs:
        if spec.family == "ngram":
            result.append(spec)
            continue
        architecture = (
            spec.family,
            spec.symbol_kind,
            spec.embedding_dim,
            spec.hidden_dim,
            spec.layers,
            spec.heads,
        )
        if architecture in seen_neural:
            continue
        seen_neural.add(architecture)
        base_name = spec.name.rsplit("-w", 1)[0]
        result.extend(
            replace(spec, name=f"{base_name}-w{length}", context_length=length)
            for length in context_lengths
        )
    return tuple(result)


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


def save_ngram_predictor(
    path: Path, model: NGramPredictor, *, tokenizer_name: str, token_ids: Sequence[int]
) -> str:
    """Persist a token n-gram model and return its SHA-256 digest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": 1,
        "kind": "ngram",
        "order": model.order,
        "vocabulary_size": model.vocabulary_size,
        "tokenizer_name": tokenizer_name,
        "token_ids": list(token_ids),
        "counts": dict(model._counts),
    }
    path.write_bytes(pickle.dumps(payload, protocol=5))
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ngram_predictor(
    path: Path, *, tokenizer_name: str, token_ids: Sequence[int], expected_sha256: str | None = None
) -> NGramPredictor:
    """Load a checkpoint only when its tokenizer, alphabet, and digest match."""
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("n-gram checkpoint SHA-256 does not match stream metadata")
    payload = pickle.loads(raw)
    if payload.get("format_version") != 1 or payload.get("kind") != "ngram":
        raise ValueError("unsupported n-gram checkpoint format")
    if payload.get("tokenizer_name") != tokenizer_name:
        raise ValueError("n-gram checkpoint tokenizer does not match this stream")
    if payload.get("token_ids") != list(token_ids):
        raise ValueError("n-gram checkpoint alphabet does not match this stream")
    model = NGramPredictor(payload["vocabulary_size"], order=payload["order"])
    if model.vocabulary_size != len(token_ids):
        raise ValueError("n-gram checkpoint has an invalid vocabulary size")
    model._counts.update(payload["counts"])
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
