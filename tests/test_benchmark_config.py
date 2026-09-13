from pathlib import Path

import pytest

from src.relational_compression_benchmark.config import ImdbDatasetConfig, load_sweep_config


SMOKE_CONFIG = Path(
    "papers/cidr_2027/experiments/configs/row_column_smoke.toml"
)


def test_smoke_config_is_explicit_and_runnable():
    config = load_sweep_config(SMOKE_CONFIG)

    assert config.layouts == ("row_major", "column_major")
    assert config.blocks_bytes == (65_536,)
    assert [pipeline.name for pipeline in config.pipelines] == [
        "raw-identity",
        "raw-zstd-3",
        "qwen-token-ids",
        "qwen-token-ids-zstd-3",
    ]
    token_pipelines = [
        pipeline
        for pipeline in config.pipelines
        if pipeline.representation == "token_ids"
    ]
    assert all(
        pipeline.tokenizer_name == "Qwen/Qwen2.5-0.5B"
        and pipeline.tokenizer_revision
        for pipeline in token_pipelines
    )


def test_unknown_config_fields_are_rejected(tmp_path):
    text = SMOKE_CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(text + "\naccidental_field = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown artifacts fields"):
        load_sweep_config(path)


def test_imdb_dataset_config_is_typed_and_pinned(tmp_path):
    text = SMOKE_CONFIG.read_text(encoding="utf-8")
    start = text.index("[dataset]")
    end = text.index("[matrix]")
    imdb = """[dataset]
kind = "imdb_title_basics"
split = "evaluation"
rows = 8192
path = "data/cidr_2027/imdb/title.basics.tsv.gz"
sha256 = "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
split_seed = 20270303
evaluation_fraction = 0.2

"""
    path = tmp_path / "imdb.toml"
    path.write_text(text[:start] + imdb + text[end:], encoding="utf-8")

    config = load_sweep_config(path)

    assert isinstance(config.dataset, ImdbDatasetConfig)
    assert config.dataset.path == Path(
        "data/cidr_2027/imdb/title.basics.tsv.gz"
    )
