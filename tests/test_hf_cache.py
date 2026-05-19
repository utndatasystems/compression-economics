from pathlib import Path

from src import hf_cache


def _write_local_snapshot(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")
    (path / "tokenizer.json").write_text("{}", encoding="utf-8")


def test_resolve_pretrained_model_source_prefers_repo_local_snapshot(monkeypatch, tmp_path):
    local_models_dir = tmp_path / "models"
    local_snapshot = local_models_dir / "Qwen2.5-0.5B"
    _write_local_snapshot(local_snapshot)
    monkeypatch.setattr(hf_cache, "LOCAL_MODELS_DIR", local_models_dir)

    resolved = hf_cache.resolve_pretrained_model_source("Qwen/Qwen2.5-0.5B")

    assert resolved == str(local_snapshot.resolve())


def test_resolve_pretrained_model_source_ignores_incomplete_snapshot(monkeypatch, tmp_path):
    local_models_dir = tmp_path / "models"
    incomplete_snapshot = local_models_dir / "Qwen2.5-0.5B"
    incomplete_snapshot.mkdir(parents=True, exist_ok=True)
    (incomplete_snapshot / "config.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(hf_cache, "LOCAL_MODELS_DIR", local_models_dir)

    resolved = hf_cache.resolve_pretrained_model_source("Qwen/Qwen2.5-0.5B")

    assert resolved == "Qwen/Qwen2.5-0.5B"


def test_resolve_pretrained_model_source_preserves_explicit_local_dir(tmp_path):
    explicit_local_dir = tmp_path / "explicit-model"
    explicit_local_dir.mkdir(parents=True, exist_ok=True)

    resolved = hf_cache.resolve_pretrained_model_source(str(explicit_local_dir))

    assert resolved == str(explicit_local_dir.resolve())