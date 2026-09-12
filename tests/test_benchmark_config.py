from pathlib import Path

import pytest

from src.benchmark.config import load_sweep_config


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
    ]
    assert all(
        pipeline.representation == "raw_bytes" for pipeline in config.pipelines
    )


def test_unknown_config_fields_are_rejected(tmp_path):
    text = SMOKE_CONFIG.read_text(encoding="utf-8")
    path = tmp_path / "invalid.toml"
    path.write_text(text + "\naccidental_field = true\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Unknown artifacts fields"):
        load_sweep_config(path)
