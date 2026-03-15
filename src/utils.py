"""
Utility helpers for bitstream packing and experiment metadata persistence.

This module provides:
- bit/byte conversion helpers for compact storage of bit strings
- save/load routines for global-mask compression artifacts
- JSON utilities for aggregating experiment results
"""

import struct
import json
import os
from datetime import datetime
import torch

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

def save_global_mask_file(
    args,
    first_token,
    bit_string,    # e.g. [1,1,0,0,...]
    bitmask_data
):
    """
    Save global-mask compression artifacts to a binary file.

    File layout (binary):
        1) JSON header line (UTF-8) + newline
        2) first_token values (batch_size * uint32)
        3) bit_string bytes length (uint32), padding (uint8), bit_string bytes
        4) bitmask_data length (uint32), bitmask_data bytes

    Args:
        args (argparse.Namespace): Experiment configuration with output_path.
        first_token (list[int]): First token for each batch.
        bit_string (list[int]): Bit list to be packed and stored.
        bitmask_data (bytes): Serialized bitmap describing allowed vocabulary.
    """
    file_path = args.output_path
    header = {
        "input_path": os.path.basename(args.input_path),
        "model_name": args.model_name,
        "context_length": args.context_length,
        "first_n_tokens": args.first_n_tokens,
        "retain_tokens": args.retain_tokens,
        "use_kv_cache": args.use_kv_cache,
        "batch_size": args.batch_size
    }
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

    Returns:
        header, first_token, bit_string(list[int]), bitmask_data
    """
    file_path = args.input_path
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
    args.context_length = header["context_length"]
    args.first_n_tokens = header["first_n_tokens"]
    args.retain_tokens = header["retain_tokens"]
    args.use_kv_cache = header["use_kv_cache"]
    args.batch_size = header["batch_size"]
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