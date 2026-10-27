import json
from pathlib import Path

import pytest

from research.papers.coding_survey.experiments.baseline_adapter import (
    build_baseline_result,
    combine_baseline_results,
    load_text_prefix,
    prepare_work_archive,
    read_framed_archive,
    sha256_bytes,
    sha256_file,
    write_framed_archive,
    write_json,
)


def test_load_text_prefix_and_hashes(tmp_path: Path) -> None:
    input_path = tmp_path / "input.txt"
    input_path.write_bytes("héllo".encode())

    source, text = load_text_prefix(input_path, len(input_path.read_bytes()))

    assert text == "héllo"
    assert sha256_file(input_path) == sha256_bytes(source)


def test_load_text_prefix_rejects_short_and_invalid_inputs(tmp_path: Path) -> None:
    short_path = tmp_path / "short.txt"
    short_path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="smaller than the requested prefix"):
        load_text_prefix(short_path, 4)

    invalid_path = tmp_path / "invalid.txt"
    invalid_path.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="not valid UTF-8"):
        load_text_prefix(invalid_path, 1)


def test_prepare_work_archive_rejects_existing_archive(tmp_path: Path) -> None:
    output = tmp_path / "result.json"
    work_dir, archive = prepare_work_archive(output, None, "codec.bin")
    assert work_dir == output.with_suffix("")
    assert archive == work_dir / "codec.bin"
    archive.write_bytes(b"already present")

    with pytest.raises(FileExistsError, match="archive already exists"):
        prepare_work_archive(output, None, "codec.bin")


def test_framed_archive_round_trip_and_payload_validation(tmp_path: Path) -> None:
    archive = tmp_path / "archive.bin"
    metadata = {"payload_bytes": 3, "value": "kept"}
    write_framed_archive(archive, b"TEST", metadata, b"abc")

    actual_metadata, payload = read_framed_archive(
        archive, b"TEST", codec_label="Test"
    )
    assert actual_metadata == metadata
    assert payload == b"abc"

    damaged = bytearray(archive.read_bytes())
    damaged.pop()
    archive.write_bytes(damaged)
    with pytest.raises(ValueError, match="payload length disagrees"):
        read_framed_archive(archive, b"TEST", codec_label="Test")


def _write_result(
    tmp_path: Path, codec: str, *, source: bytes = b"source", recovered: bool = True
) -> Path:
    archive = tmp_path / f"{codec}.bin"
    archive.write_bytes(codec.encode())
    result = build_baseline_result(
        codec=codec,
        implementation={"revision": "abc123"},
        input_path=tmp_path / "input.txt",
        source=source,
        model={"name": "model"},
        configuration={"setting": 1},
        archive_path=archive,
        model_load_seconds=1.0,
        compression_seconds=2.0,
        decompression_seconds=4.0,
        environment={"python": "test"},
        exact_byte_recovery=recovered,
    )
    path = tmp_path / f"{codec}.json"
    write_json(path, result)
    return path


def test_build_and_combine_baseline_results(tmp_path: Path) -> None:
    paths = [_write_result(tmp_path, codec) for codec in ("finezip", "llmzip", "nacrith")]

    summary = combine_baseline_results(paths)

    assert summary["schema_version"] == 1
    assert summary["input"] == {
        "bytes": len(b"source"),
        "sha256": sha256_bytes(b"source"),
    }
    assert [row["codec"] for row in summary["codecs"]] == [
        "finezip",
        "llmzip",
        "nacrith",
    ]
    finezip = json.loads(paths[0].read_text())
    assert finezip["archive"]["compression_ratio"] == len(b"source") / len(b"finezip")
    assert finezip["timing"]["compression_mib_per_second"] == len(b"source") / 2.0 / 2**20


def test_combine_baseline_results_rejects_mismatches(tmp_path: Path) -> None:
    paths = [
        _write_result(tmp_path, "finezip"),
        _write_result(tmp_path, "llmzip", source=b"different"),
        _write_result(tmp_path, "nacrith"),
    ]
    with pytest.raises(ValueError, match="inputs do not match"):
        combine_baseline_results(paths)

    paths[1] = _write_result(tmp_path, "llmzip", recovered=False)
    with pytest.raises(ValueError, match="failed exact byte recovery"):
        combine_baseline_results(paths)
