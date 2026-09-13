"""Validated, versioned catalog for pretrained predictive-compression models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable


CATALOG_VERSION = 1
DEFAULT_CATALOG = Path(__file__).resolve().parents[1] / "experiments" / "model_ladder.json"
_FAMILIES = {"transformer", "ssm", "hybrid_moe"}
_DTYPES = {"float32", "bfloat16", "float16"}


@dataclass(frozen=True)
class ModelLadderEntry:
    """One reproducible rung in the pretrained causal-model ladder."""

    name: str
    model_id: str
    revision: str
    tokenizer_id: str
    tokenizer_revision: str
    family: str
    architecture: str
    total_parameters: int
    active_parameters: int
    checkpoint_dtype: str
    enabled_by_default: bool
    trust_remote_code: bool = False
    notes: str | None = None

    def __post_init__(self) -> None:
        for field_name in ("name", "model_id", "tokenizer_id", "architecture"):
            if not getattr(self, field_name):
                raise ValueError(f"{field_name} must not be empty")
        for field_name in ("revision", "tokenizer_revision"):
            revision = getattr(self, field_name)
            if len(revision) != 40 or any(char not in "0123456789abcdef" for char in revision):
                raise ValueError(f"{field_name} must be a pinned 40-character commit SHA")
        if self.family not in _FAMILIES:
            raise ValueError(f"unsupported model family: {self.family}")
        if self.checkpoint_dtype not in _DTYPES:
            raise ValueError(f"unsupported checkpoint dtype: {self.checkpoint_dtype}")
        if self.total_parameters <= 0 or self.active_parameters <= 0:
            raise ValueError("parameter counts must be positive")
        if self.active_parameters > self.total_parameters:
            raise ValueError("active_parameters cannot exceed total_parameters")


@dataclass(frozen=True)
class ModelLadder:
    catalog_version: int
    parameter_count_policy: str
    models: tuple[ModelLadderEntry, ...]

    def __post_init__(self) -> None:
        if self.catalog_version != CATALOG_VERSION:
            raise ValueError(
                f"unsupported model-ladder version {self.catalog_version}; expected {CATALOG_VERSION}"
            )
        if not self.parameter_count_policy:
            raise ValueError("parameter_count_policy must not be empty")
        names = [model.name for model in self.models]
        if len(names) != len(set(names)):
            raise ValueError("model names must be unique")
        identities = [(model.model_id, model.revision) for model in self.models]
        if len(identities) != len(set(identities)):
            raise ValueError("model ID/revision pairs must be unique")

    def select(
        self, names: Iterable[str] | None = None, *, include_optional: bool = False
    ) -> tuple[ModelLadderEntry, ...]:
        requested = tuple(names or ())
        by_name = {model.name: model for model in self.models}
        unknown = sorted(set(requested) - by_name.keys())
        if unknown:
            raise ValueError(f"unknown model ladder entries: {', '.join(unknown)}")
        if requested:
            requested_set = set(requested)
            return tuple(model for model in self.models if model.name in requested_set)
        if include_optional:
            return self.models
        return tuple(model for model in self.models if model.enabled_by_default)


def load_model_ladder(path: str | Path = DEFAULT_CATALOG) -> ModelLadder:
    """Load the catalog, rejecting misspelled or silently ignored fields."""
    raw: dict[str, Any] = json.loads(Path(path).read_text(encoding="utf-8"))
    expected_root = {"catalog_version", "parameter_count_policy", "models"}
    extra_root = set(raw) - expected_root
    missing_root = expected_root - set(raw)
    if extra_root or missing_root:
        raise ValueError(
            f"invalid model-ladder root fields; missing={sorted(missing_root)}, extra={sorted(extra_root)}"
        )
    entry_fields = set(ModelLadderEntry.__dataclass_fields__)
    models = []
    for index, value in enumerate(raw["models"]):
        extra = set(value) - entry_fields
        required = entry_fields - {"trust_remote_code", "notes"}
        missing = required - set(value)
        if extra or missing:
            raise ValueError(
                f"invalid model entry {index}; missing={sorted(missing)}, extra={sorted(extra)}"
            )
        models.append(ModelLadderEntry(**value))
    return ModelLadder(
        catalog_version=raw["catalog_version"],
        parameter_count_policy=raw["parameter_count_policy"],
        models=tuple(models),
    )
