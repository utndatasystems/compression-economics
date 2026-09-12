import csv
import gzip
from pathlib import Path

import pytest

from src.benchmark.config import ImdbDatasetConfig
from src.benchmark.datasets import file_sha256, load_imdb_title_basics


HEADER = (
    "tconst",
    "titleType",
    "primaryTitle",
    "originalTitle",
    "isAdult",
    "startYear",
    "endYear",
    "runtimeMinutes",
    "genres",
)


def _write_fixture(path: Path, rows: int = 200) -> str:
    """Create a small title.basics-compatible gzip and return its checksum."""
    with gzip.open(path, "wt", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t", lineterminator="\n")
        writer.writerow(HEADER)
        for index in range(rows):
            writer.writerow(
                (
                    f"tt{index:07d}",
                    "movie" if index % 2 else "short",
                    f"Primary title {index}",
                    f"Original title {index}",
                    str(index % 2),
                    str(1980 + index % 40),
                    r"\N",
                    r"\N" if index % 3 == 0 else str(60 + index % 90),
                    r"\N" if index % 5 == 0 else "Drama,Comedy",
                )
            )
    return file_sha256(path)


def _config(path: Path, sha256: str, split: str, rows: int = 10):
    """Build a concise IMDb configuration for fixture-based tests."""
    return ImdbDatasetConfig(
        kind="imdb_title_basics",
        split=split,
        rows=rows,
        path=path,
        sha256=sha256,
        split_seed=20270303,
        evaluation_fraction=0.2,
    )


def test_imdb_loader_parses_types_and_nulls(tmp_path):
    path = tmp_path / "title.basics.tsv.gz"
    sha256 = _write_fixture(path)

    table = load_imdb_title_basics(_config(path, sha256, "tuning"))

    assert len(table.rows) == 10
    assert tuple(column.name for column in table.columns) == HEADER
    assert isinstance(table.rows[0][4], bool)
    assert isinstance(table.rows[0][5], int)
    assert any(row[6] is None for row in table.rows)
    assert any(row[7] is None for row in table.rows)
    assert any(row[8] is None for row in table.rows)


def test_imdb_splits_are_deterministic_and_disjoint(tmp_path):
    path = tmp_path / "title.basics.tsv.gz"
    sha256 = _write_fixture(path)
    tuning = load_imdb_title_basics(_config(path, sha256, "tuning"))
    evaluation = load_imdb_title_basics(
        _config(path, sha256, "evaluation")
    )

    assert tuning == load_imdb_title_basics(_config(path, sha256, "tuning"))
    assert {row[0] for row in tuning.rows}.isdisjoint(
        row[0] for row in evaluation.rows
    )


def test_imdb_loader_rejects_unpinned_source(tmp_path):
    path = tmp_path / "title.basics.tsv.gz"
    _write_fixture(path)

    with pytest.raises(ValueError, match="checksum mismatch"):
        load_imdb_title_basics(_config(path, "0" * 64, "evaluation"))
