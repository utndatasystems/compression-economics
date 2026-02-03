import struct
import json
import os

def bits_to_bytes(bits):
    """Convert list of bits (ints) to bytes."""
    bit_str = ''.join(str(b) for b in bits)
    padding = (8 - len(bit_str) % 8) % 8  # Pad to full byte
    bit_str = bit_str + '0' * padding
    return int(bit_str, 2).to_bytes(len(bit_str) // 8, 'big'), padding

def bytes_to_bits(byte_data, padding):
    """Convert bytes back to list of bits."""
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
    Save compression results to file with binary-encoded bit_string.
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

        # Write first_token values (batch_size tokens)
        for tok in first_token:
            f.write(struct.pack("I", tok))

        # Convert bit_string list to bytes
        bit_bytes, padding = bits_to_bytes(bit_string)
        f.write(struct.pack("I", len(bit_bytes)))
        f.write(struct.pack("B", padding))  # store padding
        f.write(bit_bytes)

        # Write bitmask_data
        f.write(struct.pack("I", len(bitmask_data)))
        f.write(bitmask_data)

def load_global_mask_file(args):
    """
    Load compression results from file.
    Returns:
        header, first_token, bit_string(list[int]), bitmask_data
    """
    file_path = args.input_path
    with open(file_path, "rb") as f:
        header_line = f.readline().decode("utf-8").strip()
        header = json.loads(header_line)

        first_token = [struct.unpack("I", f.read(4))[0] for _ in range(header["batch_size"])]

        # Read bit_string
        bit_len = struct.unpack("I", f.read(4))[0]
        padding = struct.unpack("B", f.read(1))[0]
        bit_bytes = f.read(bit_len)
        bit_string = bytes_to_bits(bit_bytes, padding)

        # Read bitmask_data
        bitmask_len = struct.unpack("I", f.read(4))[0]
        bitmask_data = f.read(bitmask_len)

    # Update args with loaded header values
    args.model_name = header["model_name"]
    args.context_length = header["context_length"]
    args.first_n_tokens = header["first_n_tokens"]
    args.retain_tokens = header["retain_tokens"]
    args.use_kv_cache = header["use_kv_cache"]
    args.batch_size = header["batch_size"]
    args.input_path = header["input_path"]

    return args, first_token, bit_string, bitmask_data
