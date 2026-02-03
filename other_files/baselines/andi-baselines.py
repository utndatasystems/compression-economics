import subprocess
import time
import os
import shutil

# Define tools with compression, decompression commands, and extensions
TOOLS = {
    "zip": {
        "compress": ["bash", "-c", 'zip "${0}.zip" "${0}"'],
        "decompress": ["bash", "-c", "unzip -p ${0} > ${0%.zip}"],
        "ext": ".zip"
    },
    "gzip-default": {
        "compress": ["gzip", "-k", "-f"],
        "decompress": ["gzip", "-d", "-k", "-f"],
        "ext": ".gz"
    },
    "gzip-9": {
        "compress": ["gzip", "-k", "-9", "-f"],
        "decompress": ["gzip", "-d", "-k", "-f"],
        "ext": ".gz"
    },
    "zstd-default": {
        "compress": ["zstd", "-k", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst"
    },
    "zstd-3": {
        "compress": ["zstd", "-k", "-3", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst"
    },
    "zstd-1": {
        "compress": ["zstd", "-k", "-1", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst"
    },
    "zstd-19": {
        "compress": ["zstd", "-k", "-19", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst"
    },
    "zstd-ultra": {
        "compress": ["zstd", "-k", "--ultra", "-22", "-f"],
        "decompress": ["zstd", "-d", "-k", "-f"],
        "ext": ".zst"
    },
    "xz-9e": {
        "compress": ["xz", "-k", "-9e", "-f"],
        "decompress": ["xz", "-d", "-k", "-f"],
        "ext": ".xz"
    },
    "brotli-q11": {
        "compress": ["brotli", "-k", "-q 11", "--force", "-f"],
        "decompress": ["brotli", "-d", "-k", "-f"],
        "ext": ".br"
    },
    "zpaq": {
        "compress": ["bash", "-c", 'zpaq add "${0}.zpaq" "${0}" -method 5'],
        "decompress": ["bash", "-c", 'zpaq extract "${0}" -force'],
        "ext": ".zpaq"
    },
    "paq8px": {
        "compress": ["bash", "-c", 'paq8px/build/paq8px -2 "${0}" "${0}.paq8px"'],
        "decompress": ["bash", "-c", 'paq8px/build/paq8px -d "${0}.paq8px" "${0}"'],
        "ext": ".paq8px"
    },
    "cmix": {
        "compress": ["bash", "-c", 'cmix/cmix -c "${0}" "${0}.cmix"'],
        "decompress": ["bash", "-c", 'cmix/cmix -d "${0}.cmix" "${0}"'],
        "ext": ".cmix"
    },
}


def run_command(cmd):
    start = time.time()
    subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    end = time.time()
    return end - start

def files_are_equal(file1, file2):
    with open(file1, 'rb') as f1, open(file2, 'rb') as f2:
        while True:
            b1 = f1.read(8192)
            b2 = f2.read(8192)
            if b1 != b2:
                return False
            if not b1:
                return True

def benchmark_tool(tool, input_file):
    print(f"\n=== {tool.upper()} on {input_file} ===")
    config = TOOLS[tool]

    compressed_file = input_file + config["ext"]
    decompressed_file = input_file + ".decompressed"

    input_size_bytes = os.path.getsize(input_file)

    # Compression benchmark
    compress_time = run_command(config["compress"] + [input_file])
    if not os.path.exists(compressed_file):
        print("Compression failed.")
        return
    compressed_size_bytes = os.path.getsize(compressed_file)

    compress_speed = input_size_bytes / compress_time
    compression_ratio = compressed_size_bytes / input_size_bytes

    print(f"Compression time: {compress_time:.4f} seconds")
    print(f"Compression speed: {compress_speed:.2f} bytes/second")
    print(f"Compressed size: {compressed_size_bytes} bytes")
    print(f"Compression ratio: {compression_ratio*100}%")

    # Prepare for decompression
    shutil.copy(compressed_file, decompressed_file + config["ext"])

    # Decompression benchmark
    decompress_time = run_command(config["decompress"] + [decompressed_file + config["ext"]])
    if not os.path.exists(decompressed_file):
        print("Decompression failed.")
        return
    decompress_speed = input_size_bytes / decompress_time
    integrity_ok = files_are_equal(input_file, decompressed_file)

    print(f"Decompression time: {decompress_time:.4f} seconds")
    print(f"Decompression speed: {decompress_speed:.2f} bytes/second")
    print(f"Integrity check: {'PASSED' if integrity_ok else 'FAILED'}")

    log_file.write(f'{input_file},{input_size_bytes},{tool},{compress_time},{compressed_size_bytes},{decompress_time}\n')

log_file = open('competitors.csv', 'w')


if __name__ == "__main__":
    input_files = [
        'data/text8.txt',
        'data/pytorrent.jsonl',
    ]

    log_file.write('Dataset,Dataset Size,Technique,Compression Duration,Compressed Size,Decompression Duration\n')

    for input_file in input_files:
        for tool in TOOLS:
            benchmark_tool(tool, input_file)
