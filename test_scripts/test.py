import os
import tempfile
import filecmp
import json

from test_scripts.compressor import compress, decompress

DATA_DIR = "data"
TEMP_DIR = "tmp"

def test_dataset(file_path):
    result = {
        "name": os.path.basename(file_path),
        "original_size": 0,
        "compressed_size": 0,
        "compression_ratio": 0.0,
        "compression_time": 0.0,
        "decompression_time": 0.0
    }
    # Read original
    with open(file_path, "rb") as original_file:
        # Compress to temporary file
        with tempfile.NamedTemporaryFile(dir=TEMP_DIR, delete=False) as compressed_file:
            compression_time = compress(original_file, compressed_file)

        compressed_path = compressed_file.name
        original_file_size = os.path.getsize(file_path)
        compressed_file_size = os.path.getsize(compressed_path)
        print(f"[{os.path.basename(file_path)}] Original size: {original_file_size} bytes")
        print(f"[{os.path.basename(file_path)}] Compressed size: {compressed_file_size} bytes")
        print(f"[{os.path.basename(file_path)}] Compression ratio: {compressed_file_size / original_file_size * 100:.2f}%")
        print(f"[{os.path.basename(file_path)}] Compression time: {compression_time:.6f} seconds")
        result["original_size"] = original_file_size
        result["compressed_size"] = compressed_file_size
        result["compression_ratio"] = compressed_file_size / original_file_size
        result["compression_time"] = compression_time

        # Decompress to another temp file
        with open(compressed_path, "rb") as comp_file:
            with tempfile.NamedTemporaryFile(dir=TEMP_DIR, delete=False) as decompressed_file:
                decompress_time = decompress(comp_file, decompressed_file)
        
        decompressed_path = decompressed_file.name
        print(f"[{os.path.basename(file_path)}] Decompression time: {decompress_time:.6f} seconds")
        result["decompression_time"] = decompress_time

        # Compare content
        if filecmp.cmp(file_path, decompressed_path, shallow=False):
            print(f"[{os.path.basename(file_path)}] ✅ Decompression verified")
        else:
            print(f"[{os.path.basename(file_path)}] ❌ The content is not the same!")
        os.remove(compressed_path)
        os.remove(decompressed_path)
        return result

def main():
    os.makedirs(TEMP_DIR, exist_ok=True)
    datasets_info = json.load(open("datasets_info.json", "r"))
    results = []
    for dataset in datasets_info:
        if os.path.isfile(dataset['path']):
            print(f"\nTesting [{dataset['name']}]...")
            result = test_dataset(dataset['path'])
            results.append(result)

    # Save results to JSON file
    with open("results.json", "w") as f:
        json.dump(results, f, indent=4)
        

if __name__ == "__main__":
    main()
