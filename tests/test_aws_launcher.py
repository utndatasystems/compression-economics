from pathlib import Path

import pytest

from src.aws_launcher import LaunchPlan, build_remote_command, normalize_spec, render_user_data, validate_spec


def make_base_spec() -> dict:
    return {
        "name": "transformer-smoke",
        "region": "us-east-1",
        "s3_bucket": "example-bucket",
        "ec2": {"launch_template_name": "compression-economics-transformer"},
        "execution": {
            "entrypoint": "main.py",
            "args": ["--mode", "compress", "--engine", "transformer"],
        },
    }


def test_normalize_spec_sets_backend_and_defaults() -> None:
    spec = normalize_spec(make_base_spec())

    validate_spec(spec)

    assert spec["backend_profile"] == "transformer"
    assert spec["execution"]["setup_mode"] == "skip-model-downloads"
    assert spec["execution"]["upload_paths"]


def test_validate_spec_rejects_unsupported_entrypoint() -> None:
    spec = normalize_spec(make_base_spec())
    spec["execution"]["entrypoint"] = "run_grid_search_backends.sh"

    with pytest.raises(ValueError, match="Unsupported entrypoint"):
        validate_spec(spec)


def test_render_user_data_includes_shutdown_and_setup_flags(tmp_path: Path) -> None:
    spec = normalize_spec(make_base_spec())
    validate_spec(spec)

    plan = LaunchPlan(
        spec_path=tmp_path / "spec.json",
        resolved_spec=spec,
        run_id="20260430T000000Z-transformer-smoke-deadbeef",
        created_at="2026-04-30T00:00:00Z",
        run_prefix="compression-economics/runs/20260430T000000Z-transformer-smoke-deadbeef",
        command=build_remote_command(spec["execution"]["entrypoint"], spec["execution"]["args"]),
        backend_profile="transformer",
        staging_dir=tmp_path,
        bundle_path=tmp_path / "workspace.tar.gz",
        bundle_sha256="1234",
        bundle_s3_key="compression-economics/runs/example/inputs/workspace.tar.gz",
        resolved_spec_path=tmp_path / "resolved-spec.json",
        resolved_spec_s3_key="compression-economics/runs/example/inputs/resolved-spec.json",
        user_data_path=tmp_path / "user-data.sh",
        user_data_s3_key="compression-economics/runs/example/inputs/user-data.sh",
        user_data="",
        prepared_assets=[],
        git_sha="abc123",
    )

    user_data = render_user_data(plan)

    assert "./setup.sh --model-cache-dir /opt/compression-cache --backend-profile transformer --skip-model-downloads" in user_data
    assert "shutdown -h now" in user_data
    assert "timeout --preserve-status" not in user_data