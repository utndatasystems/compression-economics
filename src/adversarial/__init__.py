"""NeurIPS adversarial attack generation, scoring, and stream framing."""

from .generation import (
    AdversarialGeneration,
    generate_worst_case_sequences,
    normalize_candidate_ids,
    rescore_sequences,
    score_target_tokens,
    select_worst_tokens,
)

__all__ = [
    "AdversarialGeneration",
    "generate_worst_case_sequences",
    "normalize_candidate_ids",
    "rescore_sequences",
    "score_target_tokens",
    "select_worst_tokens",
]
