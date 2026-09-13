import json

import pytest

from src.model_ladder import CATALOG_VERSION, DEFAULT_CATALOG, load_model_ladder


def test_repository_model_ladder_is_valid_and_ordered_by_family_size():
    ladder = load_model_ladder()
    assert ladder.catalog_version == CATALOG_VERSION
    assert {model.family for model in ladder.models} == {"transformer", "ssm", "hybrid_moe"}
    for prefix in ("gpt2", "mamba", "qwen2.5", "nemotron"):
        assert any(model.name.startswith(prefix) for model in ladder.models)
    for architecture in {model.architecture for model in ladder.models}:
        sizes = [model.total_parameters for model in ladder.models if model.architecture == architecture]
        assert sizes == sorted(sizes)


def test_default_ladder_excludes_accelerator_scale_optional_models():
    ladder = load_model_ladder()
    default = ladder.select()
    assert default and all(model.enabled_by_default for model in default)
    assert "nemotron3-nano-30b-a3b" not in {model.name for model in default}
    assert ladder.select(["nemotron3-nano-30b-a3b"])[0].active_parameters == 3_000_000_000


def test_catalog_rejects_unpinned_revision(tmp_path):
    source = json.loads(DEFAULT_CATALOG.read_text())
    source["models"][0]["revision"] = "main"
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="pinned 40-character commit SHA"):
        load_model_ladder(path)


def test_catalog_rejects_unknown_entry_field(tmp_path):
    source = json.loads(DEFAULT_CATALOG.read_text())
    source["models"][0]["parameterz"] = 1
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(source))
    with pytest.raises(ValueError, match="parameterz"):
        load_model_ladder(path)


def test_selection_rejects_unknown_name():
    with pytest.raises(ValueError, match="unknown model ladder entries"):
        load_model_ladder().select(["not-a-model"])
