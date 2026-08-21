from pathlib import Path

import pytest

from experiments.tokenizer_fertility import (
    iter_ascii_chunks,
    measure_ascii,
    measure_token_ids,
    read_ascii_region,
)


def test_ascii_fertility_counts_spaces_as_tokens() -> None:
    result = measure_ascii("one two three")
    assert result.token_count == 13
    assert result.word_count == 3
    assert result.fertility_tokens_per_word == pytest.approx(13 / 3)
    assert result.characters_per_token == 1
    assert result.round_trip_exact


def test_measure_token_ids_reports_non_lossless_decode() -> None:
    result = measure_token_ids(
        name="fake", kind="test", vocabulary_size=10, token_ids=[1, 2],
        text="a b", encoding_seconds=0.0, decoded_text="ab",
    )
    assert result.fertility_tokens_per_word == 1
    assert not result.round_trip_exact


def test_regions_are_exact_and_chunked(tmp_path: Path) -> None:
    path = tmp_path / "text8"
    path.write_bytes(b"abcdefghij")
    assert read_ascii_region(path, 3, 4) == "defg"
    assert list(iter_ascii_chunks(path, 2, 7, 3)) == ["cde", "fgh", "i"]


def test_ascii_reader_rejects_non_ascii(tmp_path: Path) -> None:
    path = tmp_path / "not-text8"
    path.write_bytes("café".encode())
    with pytest.raises(ValueError, match="ASCII"):
        read_ascii_region(path, 0, path.stat().st_size)
