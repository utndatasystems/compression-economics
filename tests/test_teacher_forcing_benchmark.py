from __future__ import annotations

import pytest

from research.papers.coding_survey.experiments.benchmark_teacher_forcing import (
    ScoringWindow,
    teacher_forcing_windows,
)


def test_teacher_forcing_windows_preserve_periodic_context_policy():
    assert teacher_forcing_windows(14, context_length=8, retain_tokens=4) == [
        ScoringWindow(0, 7, 0, 7),
        ScoringWindow(4, 11, 7, 11),
        ScoringWindow(8, 13, 11, 13),
    ]


def test_teacher_forcing_windows_cover_every_target_once():
    windows = teacher_forcing_windows(37, context_length=9, retain_tokens=3)
    scored = [
        position
        for window in windows
        for position in range(window.score_start, window.score_stop)
    ]
    assert scored == list(range(36))
    assert all(
        window.input_start <= window.score_start < window.score_stop <= window.input_stop
        for window in windows
    )


def test_teacher_forcing_windows_can_bound_each_cached_prefill():
    assert teacher_forcing_windows(
        14, context_length=8, retain_tokens=4, max_score_count=3
    ) == [
        ScoringWindow(0, 3, 0, 3),
        ScoringWindow(0, 6, 3, 6),
        ScoringWindow(0, 7, 6, 7),
        ScoringWindow(4, 10, 7, 10),
        ScoringWindow(4, 11, 10, 11),
        ScoringWindow(8, 13, 11, 13),
    ]


@pytest.mark.parametrize(
    "token_count,context_length,retain_tokens",
    [(1, 8, 4), (8, 8, 0), (8, 8, 8), (8, 4, 5)],
)
def test_teacher_forcing_windows_reject_invalid_inputs(
    token_count, context_length, retain_tokens
):
    with pytest.raises(ValueError):
        teacher_forcing_windows(token_count, context_length, retain_tokens)
