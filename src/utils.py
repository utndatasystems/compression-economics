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
        "batch_size": args.batch_size,
        "engine": getattr(args, "engine", "transformer"),
        "encoding": getattr(args, "encoding", "AC"),
        "reduce_tokens": getattr(args, "reduce_tokens", True),
        "lora_path": getattr(args, "lora_path", None),
        "tensor_parallel_size": getattr(args, "tensor_parallel_size", 1),
        "gpu_memory_utilization": getattr(args, "gpu_memory_utilization", 0.9),
        "tensorrt_engine_dir": getattr(args, "tensorrt_engine_dir", None),
        "sglang_mem_fraction_static": getattr(args, "sglang_mem_fraction_static", 0.8),
        "sglang_enable_deterministic_inference": getattr(args, "sglang_enable_deterministic_inference", True),
        "llamacpp_model_path": getattr(args, "llamacpp_model_path", None),
        "llamacpp_binary": getattr(args, "llamacpp_binary", "llama-server"),
        "llamacpp_host": getattr(args, "llamacpp_host", "127.0.0.1"),
        "llamacpp_port": getattr(args, "llamacpp_port", 8080),
        "llamacpp_threads": getattr(args, "llamacpp_threads", 1),
        "llamacpp_n_gpu_layers": getattr(args, "llamacpp_n_gpu_layers", 0),
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
    args.engine = header.get("engine", "transformer")
    args.encoding = header.get("encoding", "AC")
    args.reduce_tokens = header.get("reduce_tokens", True)
    args.lora_path = header.get("lora_path", None)
    args.tensor_parallel_size = header.get("tensor_parallel_size", 1)
    args.gpu_memory_utilization = header.get("gpu_memory_utilization", 0.9)
    args.tensorrt_engine_dir = header.get("tensorrt_engine_dir", None)
    args.sglang_mem_fraction_static = header.get("sglang_mem_fraction_static", 0.8)
    args.sglang_enable_deterministic_inference = header.get("sglang_enable_deterministic_inference", True)
    args.llamacpp_model_path = header.get("llamacpp_model_path", None)
    args.llamacpp_binary = header.get("llamacpp_binary", "llama-server")
    args.llamacpp_host = header.get("llamacpp_host", "127.0.0.1")
    args.llamacpp_port = header.get("llamacpp_port", 8080)
    args.llamacpp_threads = header.get("llamacpp_threads", 1)
    args.llamacpp_n_gpu_layers = header.get("llamacpp_n_gpu_layers", 0)

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
    return (
        f"{filename}:{args.model_name}|ctx={args.context_length}|ret={args.retain_tokens}"
        f"|n={args.first_n_tokens}|kv={args.use_kv_cache}|batch={args.batch_size}"
        f"|reduce={args.reduce_tokens}|engine={args.engine}|enc={args.encoding}"
        f"|lora={args.lora_path}|tp={getattr(args, 'tensor_parallel_size', 1)}"
        f"|gpu_mem={getattr(args, 'gpu_memory_utilization', 0.9)}"
        f"|trt_engine={getattr(args, 'tensorrt_engine_dir', None)}"
        f"|sg_mem={getattr(args, 'sglang_mem_fraction_static', 0.8)}"
        f"|sg_det={getattr(args, 'sglang_enable_deterministic_inference', True)}"
        f"|llamacpp_model={getattr(args, 'llamacpp_model_path', None)}"
        f"|llamacpp_bin={getattr(args, 'llamacpp_binary', 'llama-server')}"
        f"|llamacpp_host={getattr(args, 'llamacpp_host', '127.0.0.1')}"
        f"|llamacpp_port={getattr(args, 'llamacpp_port', 8080)}"
        f"|llamacpp_threads={getattr(args, 'llamacpp_threads', 1)}"
        f"|llamacpp_ngl={getattr(args, 'llamacpp_n_gpu_layers', 0)}"
    )


def create_run_dir(base_dir="results"):
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    run_dir = os.path.join(base_dir, f"run_{timestamp}")
    os.makedirs(run_dir, exist_ok=False)
    os.makedirs(os.path.join(run_dir, "logs"), exist_ok=True)
    return run_dir


def save_params(args, run_dir):
    with open(os.path.join(run_dir, "params.json"), "w") as f:
        json.dump(vars(args), f, indent=2)
