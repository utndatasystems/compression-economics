#!/usr/bin/env python3
"""Download and checksum IMDb title.basics for reproducible CIDR sweeps."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from urllib.request import Request, urlopen


SOURCE_URL = "https://datasets.imdbws.com/title.basics.tsv.gz"
TERMS_URL = "https://developer.imdb.com/non-commercial-datasets/"
DEFAULT_DESTINATION = Path("data/cidr_2027/imdb/title.basics.tsv.gz")
CHUNK_BYTES = 1024 * 1024


def _repository_root() -> Path:
    """Return the repository containing this script."""
    return Path(__file__).resolve().parents[4]


def _resolve_from_repository(path: Path) -> Path:
    """Resolve command-line paths relative to the repository root."""
    return path if path.is_absolute() else _repository_root() / path


def _hash_file(path: Path) -> tuple[str, int]:
    """Compute a file digest and byte count with bounded memory."""
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _download(
    url: str, destination: Path, expected_sha256: str | None
) -> tuple[str, int, dict[str, str]]:
    """Download a source atomically and return its digest and provenance."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = Request(url, headers={"User-Agent": "CIDR-2027-benchmark/1"})
    digest = hashlib.sha256()
    size = 0
    try:
        with urlopen(request) as response, temporary.open("wb") as output:
            headers = {
                name: value
                for name in ("ETag", "Last-Modified", "x-amz-meta-run-date")
                if (value := response.headers.get(name)) is not None
            }
            while chunk := response.read(CHUNK_BYTES):
                output.write(chunk)
                digest.update(chunk)
                size += len(chunk)
        sha256 = digest.hexdigest()
        if expected_sha256 is not None and sha256 != expected_sha256.lower():
            raise ValueError(
                f"SHA-256 mismatch: expected {expected_sha256.lower()}, "
                f"found {sha256}"
            )
        temporary.replace(destination)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise
    return sha256, size, headers


def _write_manifest(path: Path, manifest: dict[str, object]) -> None:
    """Write download provenance atomically as readable JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def parse_args() -> argparse.Namespace:
    """Parse the intentionally small dataset preparation interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=SOURCE_URL)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument(
        "--expected-sha256",
        help="fail if the downloaded or existing source has a different SHA-256",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing snapshot with the source currently served by IMDb",
    )
    return parser.parse_args()


def main() -> int:
    """Prepare a pinned local snapshot and print its configuration values."""
    args = parse_args()
    destination = _resolve_from_repository(args.destination)
    if destination.exists() and not args.overwrite:
        sha256, size = _hash_file(destination)
        headers: dict[str, str] = {}
        reused = True
    else:
        try:
            sha256, size, headers = _download(
                args.url, destination, args.expected_sha256
            )
        except ValueError as error:
            print(error, file=sys.stderr)
            return 1
        reused = False

    expected = args.expected_sha256
    if expected is not None and sha256.lower() != expected.lower():
        print(
            f"SHA-256 mismatch: expected {expected.lower()}, found {sha256}",
            file=sys.stderr,
        )
        return 1

    try:
        relative_destination = destination.relative_to(_repository_root())
    except ValueError:
        relative_destination = destination
    manifest_path = destination.with_suffix(destination.suffix + ".manifest.json")
    _write_manifest(
        manifest_path,
        {
            "bytes": size,
            "downloaded_at_utc": datetime.now(timezone.utc).isoformat(),
            "http_headers": headers,
            "path": str(relative_destination),
            "reused_existing_file": reused,
            "sha256": sha256,
            "source_url": args.url,
            "terms_url": TERMS_URL,
        },
    )
    print(f"IMDb snapshot: {relative_destination}")
    print(f"Bytes: {size}")
    print(f"SHA-256: {sha256}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
