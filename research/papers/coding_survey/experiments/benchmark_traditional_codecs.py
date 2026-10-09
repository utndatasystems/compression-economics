#!/usr/bin/env python3
"""Benchmark six traditional codecs on the matched 10 MiB text8 prefix."""

from __future__ import annotations

import argparse
import ctypes
import gc
import gzip
import hashlib
import json
import os
import platform
import statistics
import sys
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import brotli
import lz4
import lz4.frame
import zstandard


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "artifacts/papers/coding-survey/traditional-codecs/"
    "text8-first-10MiB.json"
)


@dataclass(frozen=True)
class Codec:
    codec_id: str
    label: str
    compress: Callable[[bytes], bytes]
    decompress: Callable[[bytes], bytes]
    implementation: str


class Fsst8Codec:
    def __init__(self, library: Path) -> None:
        self.library_path = library.resolve()
        self.library = ctypes.CDLL(str(self.library_path))
        byte_pointer = ctypes.POINTER(ctypes.c_ubyte)
        self.library.baseline_fsst8_compress_bound.argtypes = [ctypes.c_size_t]
        self.library.baseline_fsst8_compress_bound.restype = ctypes.c_size_t
        self.library.baseline_fsst8_compress.argtypes = [
            byte_pointer,
            ctypes.c_size_t,
            byte_pointer,
            ctypes.c_size_t,
        ]
        self.library.baseline_fsst8_compress.restype = ctypes.c_size_t
        self.library.baseline_fsst8_decompressed_size.argtypes = [
            byte_pointer,
            ctypes.c_size_t,
        ]
        self.library.baseline_fsst8_decompressed_size.restype = ctypes.c_size_t
        self.library.baseline_fsst8_decompress.argtypes = [
            byte_pointer,
            ctypes.c_size_t,
            byte_pointer,
            ctypes.c_size_t,
        ]
        self.library.baseline_fsst8_decompress.restype = ctypes.c_size_t

    @staticmethod
    def _input(data: bytes) -> ctypes.Array:
        return (ctypes.c_ubyte * len(data)).from_buffer_copy(data)

    def compress(self, source: bytes) -> bytes:
        source_buffer = self._input(source)
        capacity = self.library.baseline_fsst8_compress_bound(len(source))
        output = (ctypes.c_ubyte * capacity)()
        size = self.library.baseline_fsst8_compress(
            source_buffer, len(source), output, capacity
        )
        if size == 0:
            raise RuntimeError("FSST8 compression failed")
        return bytes(output[:size])

    def decompress(self, archive: bytes) -> bytes:
        archive_buffer = self._input(archive)
        capacity = self.library.baseline_fsst8_decompressed_size(
            archive_buffer, len(archive)
        )
        if capacity == 0:
            raise RuntimeError("invalid FSST8 archive")
        output = (ctypes.c_ubyte * capacity)()
        size = self.library.baseline_fsst8_decompress(
            archive_buffer, len(archive), output, capacity
        )
        if size != capacity:
            raise RuntimeError("FSST8 decompression failed")
        return bytes(output)


def _balanced_orders(codec_ids: tuple[str, ...], repeats: int) -> list[list[str]]:
    orders: list[list[str]] = []
    for repeat in range(repeats):
        offset = repeat % len(codec_ids)
        order = list(codec_ids[offset:] + codec_ids[:offset])
        if (repeat // len(codec_ids)) % 2:
            order.reverse()
        orders.append(order)
    return orders


def _time_call(operation: Callable[[bytes], bytes], data: bytes) -> tuple[bytes, int]:
    start = time.perf_counter_ns()
    result = operation(data)
    elapsed_ns = time.perf_counter_ns() - start
    if elapsed_ns <= 0:
        raise RuntimeError("non-positive elapsed time")
    return result, elapsed_ns


def _sample(elapsed_ns: int, source_bytes: int) -> dict:
    seconds = elapsed_ns / 1_000_000_000
    return {
        "elapsed_ns": elapsed_ns,
        "seconds": seconds,
        "source_mb_per_second": source_bytes / seconds / 1_000_000,
        "source_mib_per_second": source_bytes / seconds / (1024 * 1024),
    }


def _summarize(samples: list[dict]) -> dict:
    keys = ("seconds", "source_mb_per_second", "source_mib_per_second")
    return {
        "median": {key: statistics.median(row[key] for row in samples) for key in keys},
        "minimum": {key: min(row[key] for row in samples) for key in keys},
        "maximum": {key: max(row[key] for row in samples) for key in keys},
    }


def _cpu_model() -> str | None:
    try:
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=REPO_ROOT / "data/text8")
    parser.add_argument("--input-bytes", type=int, default=10 * 1024 * 1024)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument(
        "--cpu",
        type=int,
        help="pin the benchmark process to this logical CPU",
    )
    parser.add_argument("--fsst-library", type=Path, required=True)
    parser.add_argument("--fsst-revision", required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.input.is_file() or args.input_bytes < 1 or args.repeats < 1:
        parser.error("input must exist; input-bytes and repeats must be positive")
    if not args.fsst_library.is_file():
        parser.error("--fsst-library must name the compiled wrapper library")
    if args.cpu is not None:
        if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
            parser.error("--cpu requires sched_getaffinity/sched_setaffinity support")
        if args.cpu not in os.sched_getaffinity(0):
            parser.error("--cpu is outside the process's allowed CPU set")
        os.sched_setaffinity(0, {args.cpu})

    source = args.input.read_bytes()[: args.input_bytes]
    if len(source) != args.input_bytes:
        parser.error("input is smaller than the requested prefix")

    fsst = Fsst8Codec(args.fsst_library)
    codecs = (
        Codec(
            "fsst8",
            "FSST8",
            fsst.compress,
            fsst.decompress,
            f"cwida/fsst {args.fsst_revision}",
        ),
        Codec(
            "lz4_default",
            "LZ4 default",
            lambda data: lz4.frame.compress(
                data,
                compression_level=0,
                block_size=lz4.frame.BLOCKSIZE_MAX4MB,
                block_linked=True,
                content_checksum=False,
                block_checksum=False,
                store_size=True,
            ),
            lz4.frame.decompress,
            f"python-lz4 {lz4.__version__}",
        ),
        Codec(
            "gzip_6",
            "gzip level 6",
            lambda data: gzip.compress(data, compresslevel=6, mtime=0),
            gzip.decompress,
            f"Python {platform.python_version()} gzip; zlib {zlib.ZLIB_RUNTIME_VERSION}",
        ),
        Codec(
            "zstd_3",
            "Zstandard level 3",
            zstandard.ZstdCompressor(level=3, threads=0).compress,
            zstandard.ZstdDecompressor().decompress,
            f"python-zstandard {zstandard.__version__}",
        ),
        Codec(
            "zstd_19",
            "Zstandard level 19",
            zstandard.ZstdCompressor(level=19, threads=0).compress,
            zstandard.ZstdDecompressor().decompress,
            f"python-zstandard {zstandard.__version__}",
        ),
        Codec(
            "brotli_11",
            "Brotli quality 11",
            lambda data: brotli.compress(data, quality=11),
            brotli.decompress,
            f"Python Brotli {brotli.__version__}",
        ),
    )
    by_id = {codec.codec_id: codec for codec in codecs}
    codec_ids = tuple(by_id)

    gc.collect()
    gc.disable()
    try:
        archives: dict[str, bytes] = {}
        warmups: dict[str, dict] = {}
        for codec in codecs:
            archive, compression_ns = _time_call(codec.compress, source)
            decoded, decompression_ns = _time_call(codec.decompress, archive)
            if decoded != source:
                raise AssertionError(f"{codec.label} warm-up failed exact recovery")
            archives[codec.codec_id] = archive
            warmups[codec.codec_id] = {
                "compression": _sample(compression_ns, len(source)),
                "decompression": _sample(decompression_ns, len(source)),
            }

        compression_samples = {codec_id: [] for codec_id in codec_ids}
        compression_orders = _balanced_orders(codec_ids, args.repeats)
        for repeat, order in enumerate(compression_orders, start=1):
            for codec_id in order:
                codec = by_id[codec_id]
                archive, elapsed_ns = _time_call(codec.compress, source)
                if archive != archives[codec_id]:
                    raise AssertionError(f"{codec.label} emitted a non-deterministic archive")
                compression_samples[codec_id].append(_sample(elapsed_ns, len(source)))
                print(
                    f"compress {codec_id} repeat {repeat}: "
                    f"{compression_samples[codec_id][-1]['source_mib_per_second']:.3f} MiB/s",
                    flush=True,
                )

        decompression_samples = {codec_id: [] for codec_id in codec_ids}
        decompression_orders = _balanced_orders(tuple(reversed(codec_ids)), args.repeats)
        for repeat, order in enumerate(decompression_orders, start=1):
            for codec_id in order:
                codec = by_id[codec_id]
                decoded, elapsed_ns = _time_call(codec.decompress, archives[codec_id])
                if decoded != source:
                    raise AssertionError(
                        f"{codec.label} repeat {repeat} failed exact recovery"
                    )
                decompression_samples[codec_id].append(
                    _sample(elapsed_ns, len(source))
                )
                print(
                    f"decompress {codec_id} repeat {repeat}: "
                    f"{decompression_samples[codec_id][-1]['source_mib_per_second']:.3f} MiB/s",
                    flush=True,
                )
    finally:
        gc.enable()

    rows = []
    for codec in codecs:
        archive = archives[codec.codec_id]
        rows.append(
            {
                "id": codec.codec_id,
                "label": codec.label,
                "implementation": codec.implementation,
                "archive_bytes": len(archive),
                "archive_sha256": hashlib.sha256(archive).hexdigest(),
                "compression_ratio": len(source) / len(archive),
                "exact_recovery": True,
                "warmup": warmups[codec.codec_id],
                "compression_samples": compression_samples[codec.codec_id],
                "compression_summary": _summarize(compression_samples[codec.codec_id]),
                "decompression_samples": decompression_samples[codec.codec_id],
                "decompression_summary": _summarize(
                    decompression_samples[codec.codec_id]
                ),
            }
        )
    rate_winner = max(rows, key=lambda row: row["compression_ratio"])
    result = {
        "schema_version": 1,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "experiment": "coding-survey traditional-codec rate/throughput frontier",
        "source": {
            "path": str(args.input),
            "definition": f"first {len(source)} bytes",
            "bytes": len(source),
            "sha256": hashlib.sha256(source).hexdigest(),
        },
        "protocol": {
            "warmups_per_codec": 1,
            "timed_repetitions": args.repeats,
            "ordering": "balanced rotations; reverse codec order for decompression",
            "timing_scope": "in-memory one-shot codec calls; excludes file I/O and process startup",
            "throughput_numerator": "uncompressed source bytes",
            "threading": "single-threaded codec configurations",
            "cpu_pinning": args.cpu,
            "correctness": "every timed decompression compared byte-for-byte with source",
            "fsst_accounting": "compression includes table training/export; archive includes table and 12-byte frame",
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "cpu_model": _cpu_model(),
            "cpu_affinity": sorted(os.sched_getaffinity(0))
            if hasattr(os, "sched_getaffinity")
            else None,
        },
        "rate_constraint": {
            "r_star": rate_winner["compression_ratio"],
            "codec_id": rate_winner["id"],
            "definition": "maximum charged compression ratio among registered baselines",
        },
        "codecs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"saved {args.output}")
    print(
        f"R* = {result['rate_constraint']['r_star']:.6f}x "
        f"({result['rate_constraint']['codec_id']})"
    )


if __name__ == "__main__":
    main()
