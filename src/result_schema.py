"""Canonical result records for plain-text compression experiments.

The schema deliberately separates an experiment's immutable condition from its
observations.  This keeps model, tokenizer, and coder choices available to every
future sweep without forcing analysis code to reverse-engineer run names.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import platform
from typing import Any, Mapping


SCHEMA_NAME = "plain-text-compression-result"
SCHEMA_VERSION = 1


def _require_nonnegative(name: str, value: int | float | None) -> None:
    if value is not None and value < 0:
        raise ValueError(f"{name} must be nonnegative")


@dataclass(frozen=True)
class DatasetSpec:
    """Exact byte region used as compressor input."""

    name: str
    sha256: str
    source_bytes: int
    path: str | None = None
    split: str | None = None
    byte_offset: int = 0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("dataset name must not be empty")
        if len(self.sha256) != 64 or any(c not in "0123456789abcdef" for c in self.sha256):
            raise ValueError("dataset sha256 must be a lowercase hexadecimal digest")
        if self.source_bytes <= 0:
            raise ValueError("dataset source_bytes must be positive")
        _require_nonnegative("dataset byte_offset", self.byte_offset)


@dataclass(frozen=True)
class TokenizerSpec:
    """Tokenizer identity and storage dependency."""

    name: str
    kind: str
    vocabulary_size: int
    revision: str | None = None
    state_bytes: int | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.kind:
            raise ValueError("tokenizer name and kind must not be empty")
        if self.vocabulary_size <= 0:
            raise ValueError("tokenizer vocabulary_size must be positive")
        _require_nonnegative("tokenizer state_bytes", self.state_bytes)


@dataclass(frozen=True)
class PredictorSpec:
    """Predictor architecture, context, adaptation, and stored state."""

    name: str
    family: str
    context_length: int
    revision: str | None = None
    dtype: str | None = None
    total_parameters: int | None = None
    active_parameters: int | None = None
    model_state_bytes: int | None = None
    adapter: str | None = None
    adapter_parameters: int = 0
    adapter_state_bytes: int = 0
    training_mode: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.family:
            raise ValueError("predictor name and family must not be empty")
        if self.context_length < 0:
            raise ValueError("predictor context_length must be nonnegative")
        for name in (
            "total_parameters", "active_parameters", "model_state_bytes",
            "adapter_parameters", "adapter_state_bytes",
        ):
            _require_nonnegative(f"predictor {name}", getattr(self, name))
        if (
            self.total_parameters is not None
            and self.active_parameters is not None
            and self.active_parameters > self.total_parameters
        ):
            raise ValueError("active_parameters cannot exceed total_parameters")


@dataclass(frozen=True)
class CoderSpec:
    """Entropy coder and every parameter needed to reproduce its bitstream."""

    name: str
    probability_total: int | None = None
    block_symbols: int | None = None
    lanes: int | None = None
    settings: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("coder name must not be empty")
        for name in ("probability_total", "block_symbols", "lanes"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"coder {name} must be positive")


@dataclass(frozen=True)
class ExecutionSpec:
    """Runtime condition that may affect performance measurements."""

    device: str
    backend: str
    batch_size: int
    seed: int
    precision: str | None = None
    cpu_threads: int | None = None
    hardware: Mapping[str, Any] = field(default_factory=dict)
    software: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.device or not self.backend:
            raise ValueError("execution device and backend must not be empty")
        if self.batch_size <= 0:
            raise ValueError("execution batch_size must be positive")
        if self.cpu_threads is not None and self.cpu_threads <= 0:
            raise ValueError("execution cpu_threads must be positive")


@dataclass(frozen=True)
class SymbolCounts:
    """Unambiguous work counters; rates are derived from these and timings."""

    input_symbols: int
    encoded_symbols: int
    model_input_tokens: int | None = None

    def __post_init__(self) -> None:
        if self.input_symbols <= 0:
            raise ValueError("input_symbols must be positive")
        if not 0 <= self.encoded_symbols <= self.input_symbols:
            raise ValueError("encoded_symbols must be between zero and input_symbols")
        _require_nonnegative("model_input_tokens", self.model_input_tokens)


@dataclass(frozen=True)
class SizeBreakdown:
    """Non-overlapping archive and shared-dependency byte counts."""

    payload_bits: int
    payload_bytes: int
    framing_bytes: int = 0
    bitmap_bytes: int = 0
    seed_bytes: int = 0
    tokenizer_bytes: int | None = None
    model_bytes: int | None = None
    adapter_bytes: int | None = None

    def __post_init__(self) -> None:
        for name in (
            "payload_bits", "payload_bytes", "framing_bytes", "bitmap_bytes",
            "seed_bytes", "tokenizer_bytes", "model_bytes", "adapter_bytes",
        ):
            _require_nonnegative(f"sizes {name}", getattr(self, name))
        if self.payload_bytes != math.ceil(self.payload_bits / 8):
            raise ValueError("payload_bytes must equal ceil(payload_bits / 8)")

    @property
    def stream_bytes(self) -> int:
        return (
            self.payload_bytes + self.framing_bytes + self.bitmap_bytes
            + self.seed_bytes
        )

    @property
    def self_contained_bytes(self) -> int | None:
        dependencies = (self.tokenizer_bytes, self.model_bytes, self.adapter_bytes)
        if any(value is None for value in dependencies):
            return None
        return self.stream_bytes + sum(value for value in dependencies if value is not None)


@dataclass(frozen=True)
class TimingBreakdown:
    """Wall-clock durations for disjoint compression phases."""

    encode_seconds: float
    decode_seconds: float
    training_seconds: float = 0.0
    tokenization_seconds: float | None = None
    detokenization_seconds: float | None = None
    predictor_encode_seconds: float | None = None
    predictor_decode_seconds: float | None = None
    coder_encode_seconds: float | None = None
    coder_decode_seconds: float | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            _require_nonnegative(f"timings {name}", value)
        if self.encode_seconds <= 0 or self.decode_seconds <= 0:
            raise ValueError("encode_seconds and decode_seconds must be positive")


@dataclass(frozen=True)
class PlainTextCompressionResult:
    """One decoder-verified observation under one fully specified condition."""

    dataset: DatasetSpec
    tokenizer: TokenizerSpec
    predictor: PredictorSpec
    coder: CoderSpec
    execution: ExecutionSpec
    counts: SymbolCounts
    sizes: SizeBreakdown
    timings: TimingBreakdown
    roundtrip_valid: bool
    artifacts: Mapping[str, str] = field(default_factory=dict)
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def condition_id(self) -> str:
        condition = {
            "dataset": asdict(self.dataset),
            "tokenizer": asdict(self.tokenizer),
            "predictor": asdict(self.predictor),
            "coder": asdict(self.coder),
            "execution": asdict(self.execution),
        }
        encoded = json.dumps(condition, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(encoded.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible record with canonical derived metrics."""
        record = asdict(self)
        self_contained = self.sizes.self_contained_bytes
        record.update(
            schema_name=SCHEMA_NAME,
            schema_version=SCHEMA_VERSION,
            condition_id=self.condition_id,
            derived={
                "stream_bytes": self.sizes.stream_bytes,
                "shared_model_compression_factor": (
                    self.dataset.source_bytes / self.sizes.stream_bytes
                ),
                "shared_model_bits_per_source_byte": (
                    8 * self.sizes.stream_bytes / self.dataset.source_bytes
                ),
                "self_contained_bytes": self_contained,
                "self_contained_compression_factor": (
                    self.dataset.source_bytes / self_contained
                    if self_contained is not None else None
                ),
                "self_contained_bits_per_source_byte": (
                    8 * self_contained / self.dataset.source_bytes
                    if self_contained is not None else None
                ),
                "encode_input_symbols_per_second": (
                    self.counts.input_symbols / self.timings.encode_seconds
                ),
                "decode_input_symbols_per_second": (
                    self.counts.input_symbols / self.timings.decode_seconds
                ),
                "encode_model_input_tokens_per_second": (
                    self.counts.model_input_tokens / self.timings.encode_seconds
                    if self.counts.model_input_tokens is not None else None
                ),
                "decode_model_input_tokens_per_second": (
                    self.counts.model_input_tokens / self.timings.decode_seconds
                    if self.counts.model_input_tokens is not None else None
                ),
            },
        )
        return record


def local_execution_spec(*, batch_size: int, seed: int, backend: str = "torch") -> ExecutionSpec:
    """Capture a small, dependency-free execution description for local runs."""
    return ExecutionSpec(
        device="cuda" if __import__("torch").cuda.is_available() else "cpu",
        backend=backend,
        batch_size=batch_size,
        seed=seed,
        cpu_threads=__import__("torch").get_num_threads(),
        hardware={
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor() or None,
        },
        software={
            "python": platform.python_version(),
            "torch": __import__("torch").__version__,
        },
    )
