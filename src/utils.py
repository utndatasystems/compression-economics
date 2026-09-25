"""
Utility helpers for bitstream packing and experiment metadata persistence.

This module provides:
- bit/byte conversion helpers for compact storage of bit strings
- save/load routines for global-mask compression artifacts
- JSON utilities for aggregating experiment results
"""

import struct
import zlib
import json
import os
import tarfile
from datetime import datetime
import torch
from pathlib import Path
from typing import List

def count_parameters(model):
    total, trainable = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return total, trainable

def estimate_model_size_mb(model):
    total_bytes, trainable_bytes = 0, 0
    for p in model.parameters():
        total_bytes += p.numel() * p.element_size()
        if p.requires_grad:
            trainable_bytes += p.numel() * p.element_size()
    return total_bytes / (1024 ** 2), trainable_bytes / (1024 ** 2)

def bits_to_bytes(bits):
    """
    Convert a list of bits (0/1 ints) into a bytes object.

    Returns:
        tuple: (byte_data, padding_bits)
            - byte_data (bytes): Packed bytes, MSB-first.
            - padding_bits (int): Number of zero bits appended to fill the last byte.
    """
    # Build a contiguous bit string and pad to full-byte boundary.
    bit_str = ''.join(str(b) for b in bits)
    padding = (8 - len(bit_str) % 8) % 8  # Pad to full byte
    bit_str = bit_str + '0' * padding
    return int(bit_str, 2).to_bytes(len(bit_str) // 8, 'big'), padding

def bytes_to_bits(byte_data, padding):
    """
    Convert a bytes object back to a list of bits (0/1 ints).

    Args:
        byte_data (bytes): Packed bytes, MSB-first.
        padding (int): Number of zero bits appended during packing.

    Returns:
        list[int]: Bit list with padding removed.
    """
    # Recover the full bit string and drop any padding zeros.
    bit_str = bin(int.from_bytes(byte_data, 'big'))[2:].zfill(len(byte_data) * 8)
    bit_str = bit_str[:-padding] if padding > 0 else bit_str
    return [int(b) for b in bit_str]

_MULTISTREAM_MAGIC = b"GMMS"
_MULTISTREAM_VERSION = 1
_MULTISTREAM_HEADER = struct.Struct(">4sBIIIQ")
_MULTISTREAM_CHECKSUM = struct.Struct(">I")


def _save_multistream_global_mask_file(path, header, seeds, payload, bitmap):
    """Store the text context and MSAC payload with portable field widths."""
    if not isinstance(payload, bytes) or not payload.startswith(b"MSAC"):
        raise ValueError("AC_MULTISTREAM requires an MSAC byte payload")
    if len(seeds) != header["batch_size"]:
        raise ValueError("seed count does not match batch size")
    header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
    seed_bytes = b"".join(struct.pack(">I", int(seed)) for seed in seeds)
    body = header_bytes + seed_bytes + bitmap + payload
    prefix = _MULTISTREAM_HEADER.pack(
        _MULTISTREAM_MAGIC, _MULTISTREAM_VERSION, len(header_bytes),
        len(seeds), len(bitmap), len(payload),
    )
    with open(path, "wb") as handle:
        handle.write(prefix)
        handle.write(body)
        handle.write(_MULTISTREAM_CHECKSUM.pack(zlib.crc32(body)))


def _load_multistream_global_mask_file(path):
    """Reject truncated or changed files before exposing any decoded fields."""
    data = Path(path).read_bytes()
    minimum = _MULTISTREAM_HEADER.size + _MULTISTREAM_CHECKSUM.size
    if len(data) < minimum:
        raise ValueError("truncated multistream text archive")
    magic, version, header_size, seed_count, bitmap_size, payload_size = (
        _MULTISTREAM_HEADER.unpack_from(data)
    )
    if magic != _MULTISTREAM_MAGIC or version != _MULTISTREAM_VERSION:
        raise ValueError("unsupported multistream text archive")
    expected = minimum + header_size + seed_count * 4 + bitmap_size + payload_size
    if len(data) != expected:
        raise ValueError("multistream text archive length mismatch")
    body = data[_MULTISTREAM_HEADER.size:-_MULTISTREAM_CHECKSUM.size]
    (checksum,) = _MULTISTREAM_CHECKSUM.unpack_from(data, len(data) - 4)
    if zlib.crc32(body) != checksum:
        raise ValueError("multistream text archive checksum mismatch")
    offset = _MULTISTREAM_HEADER.size
    header = json.loads(data[offset:offset + header_size].decode("utf-8"))
    offset += header_size
    if header.get("encoding") != "AC_MULTISTREAM" or header.get("batch_size") != seed_count:
        raise ValueError("multistream text archive settings do not match its directory")
    seeds = list(struct.unpack_from(f">{seed_count}I", data, offset))
    offset += seed_count * 4
    bitmap = data[offset:offset + bitmap_size]
    offset += bitmap_size
    payload = data[offset:offset + payload_size]
    from src.multistream_ac import MultistreamACDecoder
    if MultistreamACDecoder(payload).stream_count != seed_count:
        raise ValueError("MSAC stream count does not match seed count")
    return header, seeds, payload, bitmap


def save_global_mask_file(
    args,
    first_token,
    bit_string,    # e.g. [1,1,0,0,...]
    bitmask_data
):
    """
    Save global-mask compression artifacts to a binary file.

    Legacy AC/ANS files use a JSON line, native-endian seed IDs, packed bits,
    and the bitmap. AC_MULTISTREAM files use the versioned GMMS layout with
    big-endian lengths and seeds, the bitmap, an MSAC payload, and a checksum.

    Args:
        args (argparse.Namespace): Experiment configuration with output_path.
        first_token (list[int]): First token for each batch.
        bit_string (list[int] | bytes): Bits for legacy coders or MSAC bytes.
        bitmask_data (bytes): Serialized bitmap describing allowed vocabulary.
    """
    file_path = args.output_path
    parent_dir = os.path.dirname(file_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    header = {
        "input_path": os.path.basename(args.input_path),
        "model_name": args.model_name,
        "model_revision": getattr(args, "model_revision", None),
        "tokenizer_name": getattr(args, "tokenizer_name", None),
        "tokenizer_revision": getattr(args, "tokenizer_revision", None),
        "trust_remote_code": getattr(args, "trust_remote_code", False),
        "context_length": args.context_length,
        "first_n_tokens": args.first_n_tokens,
        "retain_tokens": args.retain_tokens,
        "use_kv_cache": args.use_kv_cache,
        "batch_size": args.batch_size,
        "encoding": getattr(args, "encoding", None),
        "reduce_tokens": getattr(args, "reduce_tokens", None),
        "engine": getattr(args, "engine", None),
        "lora_path": getattr(args, "lora_path", None),
        "pmatic_delta": getattr(args, "pmatic_delta", None),
        "pmatic_r": getattr(args, "pmatic_r", None),
        "ngram_model_path": getattr(args, "ngram_model_path", None),
        "ngram_model_sha256": getattr(args, "ngram_model_sha256", None),
        "ngram_order": getattr(args, "ngram_order", None),
    }
    if header["encoding"] == "AC_MULTISTREAM":
        header["ac_backend"] = getattr(args, "ac_backend", "python")
        header["ac_threads"] = getattr(args, "ac_threads", None)
        _save_multistream_global_mask_file(
            file_path, header, first_token, bit_string, bitmask_data
        )
        return
    with open(file_path, "wb") as f:
        # Write header as JSON
        header_str = json.dumps(header) + "\n"
        f.write(header_str.encode("utf-8"))

        # Write first_token values (batch_size tokens) as uint32.
        for tok in first_token:
            f.write(struct.pack("I", tok))

        # Convert bit_string list to bytes with padding metadata.
        bit_bytes, padding = bits_to_bytes(bit_string)
        f.write(struct.pack("I", len(bit_bytes)))
        f.write(struct.pack("B", padding))  # store padding
        f.write(bit_bytes)

        # Write the serialized bitmap blob.
        f.write(struct.pack("I", len(bitmask_data)))
        f.write(bitmask_data)

def load_global_mask_file(args):
    """
    Load global-mask compression artifacts from a binary file.

    Args: 
        args (argparse.Namespace): Experiment configuration with input_path.

    Returns:
        Updated args, seed tokens, coded bits or MSAC bytes, and bitmap.
    """
    file_path = args.input_path
    with open(file_path, "rb") as probe:
        is_multistream = probe.read(4) == _MULTISTREAM_MAGIC
    if is_multistream:
        header, first_token, bit_string, bitmask_data = (
            _load_multistream_global_mask_file(file_path)
        )
    else:
        with open(file_path, "rb") as f:
            # Header line precedes the binary payload.
            header_line = f.readline().decode("utf-8").strip()
            header = json.loads(header_line)

            # Read batch start tokens (uint32).
            first_token = [struct.unpack("I", f.read(4))[0] for _ in range(header["batch_size"])]

            # Read bit_string
            bit_len = struct.unpack("I", f.read(4))[0]
            padding = struct.unpack("B", f.read(1))[0]
            bit_bytes = f.read(bit_len)
            bit_string = bytes_to_bits(bit_bytes, padding)

            # Read bitmask_data
            bitmask_len = struct.unpack("I", f.read(4))[0]
            bitmask_data = f.read(bitmask_len)


    # Update args with loaded header values (ensures decompression settings match).
    args.model_name = header["model_name"]
    args.model_revision = header.get("model_revision")
    args.tokenizer_name = header.get("tokenizer_name")
    args.tokenizer_revision = header.get("tokenizer_revision")
    args.trust_remote_code = header.get("trust_remote_code", False)
    args.context_length = header["context_length"]
    args.first_n_tokens = header["first_n_tokens"]
    args.retain_tokens = header["retain_tokens"]
    args.use_kv_cache = header["use_kv_cache"]
    args.batch_size = header["batch_size"]
    args.encoding = header.get("encoding", args.encoding)
    args.ac_backend = header.get("ac_backend", "python")
    args.ac_threads = header.get("ac_threads")
    args.reduce_tokens = header.get("reduce_tokens", args.reduce_tokens)
    args.engine = header.get("engine", args.engine)
    args.lora_path = header.get("lora_path", args.lora_path)
    args.pmatic_delta = header.get("pmatic_delta", getattr(args, "pmatic_delta", None))
    args.pmatic_r = header.get("pmatic_r", getattr(args, "pmatic_r", None))
    args.ngram_model_path = header.get("ngram_model_path", getattr(args, "ngram_model_path", None))
    args.ngram_model_sha256 = header.get("ngram_model_sha256", getattr(args, "ngram_model_sha256", None))
    args.ngram_order = header.get("ngram_order", getattr(args, "ngram_order", 2))
    args.input_path = header["input_path"]

    return args, first_token, bit_string, bitmask_data

def load_results(RESULTS_FILE):
    """
    Load previous experiment results from JSON (if present).

    Returns:
        dict: Parsed JSON contents, or empty dict when missing.
    """
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            return json.load(f)
    return {}

def save_results(results, RESULTS_FILE):
    """
    Save experiment results to JSON.

    Args:
        results (dict): Results payload to persist.
        RESULTS_FILE (str): Destination path.
    """
    with open(RESULTS_FILE, "w") as f:
        json.dump(results, f, indent=4)

def make_key(args):
    """
    Generate a stable key string describing an experiment configuration.

    The key captures dataset name and core settings so runs can be indexed in a dict.
    """
    filename = os.path.basename(args.input_path)
    return f"{filename}:{args.model_name}|ctx={args.context_length}|ret={args.retain_tokens}|n={args.first_n_tokens}|kv={args.use_kv_cache}|batch={args.batch_size}|reduce={args.reduce_tokens}|engine={args.engine}|enc={args.encoding}|lora={args.lora_path}"

def create_run_dir(base_dir="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir

def save_params(args, run_dir):
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

def get_device():
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision('high')
        device, use_fp16, use_bf16 = "cuda", True, False
    elif torch.backends.mps.is_available():
        device, use_fp16, use_bf16 = "mps", False, False
        torch.set_float32_matmul_precision("high")
    else:
        device, use_fp16, use_bf16 = "cpu", False, False
    return device, use_fp16, use_bf16


def folder_to_tar(folder_path, tar_path):
    """
    Package a folder into a tar archive.

    Args:
        folder_path (str): Path to the folder to be archived.
        tar_path (str): Output tar file path.

    Returns:
        str: Path to the created tar file.
    """
    if not os.path.isdir(folder_path):
        raise NotADirectoryError(f"{folder_path} is not a valid directory")

    # Create tar archive
    with tarfile.open(tar_path, "w") as tar:
        # Add folder contents recursively
        tar.add(folder_path, arcname=os.path.basename(folder_path))
    
    return tar_path


def load_model_list(path: str) -> List[str]:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"Model list file not found: {path}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list) or not all(isinstance(x, str) for x in data):
        raise ValueError("Model file must be a JSON list of strings.")

    if len(data) == 0:
        raise ValueError("Model list cannot be empty.")

    return data

def check_mismatch(input_path = None, output_path = None, first_n_tokens = None,):
    """
        Function that checks if input and output files (after compression-decompression) match, 
        and returns True if they match, False otherwise. It also prints a warning if a mismatch is detected.
    """
    if output_path is None:
        output_path = "text_results.txt"

    with open(output_path, "r", encoding="utf-8", newline="") as f:
        reconstructed = f.read()

    with open(input_path, "r", encoding="utf-8", newline="") as f:
        original = f.read()

    if first_n_tokens is not None:
        original = original[:first_n_tokens]
        reconstructed = reconstructed[:first_n_tokens]
    
    if original != reconstructed:
        return False
    else:
        return True

if __name__ == "__main__":
    pass
