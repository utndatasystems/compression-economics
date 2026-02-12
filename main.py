"""
CLI entry point for running global-mask compression and decompression experiments.
"""

import argparse
import json
import os
from src.global_mask_compressor import run_global_mask_compression, run_global_mask_decompression
from src.utils import save_global_mask_file, load_global_mask_file, load_results, save_results, make_key

RESULTS_FILE = "compression_results.json"
COMPRESSION_FILE = "compression_data.bin"
DECOMPRESSION_FILE = "text_results.txt"

def main():
    """
    Parse CLI arguments and run compression or decompression.
    """
    # ========================
    # Parse command-line arguments
    # ========================
    parser = argparse.ArgumentParser(description="Run Global Mask Compression Experiment")
    parser.add_argument("--mode", type=str, choices=["compress", "decompress"], required=True,
                        help="Mode: compress or decompress")
    parser.add_argument("--input_path", type=str, default="data/text8",help="Input path: For compress mode, dataset path. For decompress mode, compression file path.")
    parser.add_argument("--output_path", type=str, help="Output path: For compress mode, compression file path. For decompress mode, reconstruction text file path.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter (if any)")
    parser.add_argument("--context_length", type=int, default=1000, help="Maximum context length")
    parser.add_argument("--retain_tokens", type=int, default=100, help="Tokens retained when context length exceeded (only with KV cache)")
    parser.add_argument("--first_n_tokens", type=int, default=1000, help="Number of tokens to compress")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache for compression")
    parser.set_defaults(use_kv_cache=True)
    parser.add_argument("--text_input", type=str, required=False, help="The direct text input for LLM inference.")
    parser.add_argument("--reduce_tokens", action="store_true", help="Restrict token space")
    parser.add_argument("--no_reduce_tokens", dest="reduce_tokens", action="store_false", help="Disable token space restriction")
    parser.set_defaults(reduce_tokens=True)
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for LLM inference")
    parser.add_argument("--engine", type=str, choices=["transformer"], default="transformer", help="Inference engine to use")
    parser.add_argument("--encoding", type=str, choices=["AC", "bitpacked", "huffman"], default="AC", help="Encoding method for compression")
    parser.add_argument("--print_results", action="store_true", help="Print detailed results")

    args = parser.parse_args()

    if args.mode == "compress":
        # ========================
        # Validate input paths
        if not args.input_path:
            parser.error("--input_path is required in compress mode")
        if not args.output_path:
            args.output_path = COMPRESSION_FILE

        # ========================
        # Check if experiment already exists
        # ========================
        results_db = load_results(RESULTS_FILE)
        exp_key = make_key(args)
        if exp_key in results_db:
            print(f"\n⚠️ Experiment already exists for {exp_key}, skipping run.")
            print(f"Stored Results: {results_db[exp_key]}")
            return

        # ========================
        # Print experiment settings
        # ========================
        print(f"\nRunning compression with parameters:")
        print(f"  Data path        : {args.input_path}")
        print(f"  Model            : {args.model_name}")
        print(f"  Context length   : {args.context_length}")
        print(f"  Retain tokens    : {args.retain_tokens}")
        print(f"  First n tokens   : {args.first_n_tokens}")
        print(f"  Use KV cache     : {args.use_kv_cache}")
        print(f"  Batch size       : {args.batch_size}")
        print(f"  Encoding         : {args.encoding}")

        # ========================
        # Run compression
        # ========================
        first_token, bit_string, bitmask_data, comp_stats, args = run_global_mask_compression(args)

        # ========================
        # Save results (JSON stats)
        # ========================
        results_db = load_results(RESULTS_FILE)
        exp_key = make_key(args)
        if exp_key not in results_db:
            results_db[exp_key] = {}
        results_db[exp_key]["compression"] = comp_stats
        save_results(results_db, RESULTS_FILE)

        # ========================
        # Save binary compression file
        # ========================
        save_global_mask_file(
            args,
            first_token=first_token,
            bit_string=bit_string,
            bitmask_data=bitmask_data
        )

        # ========================
        # Output compression results
        # ========================
        if args.print_results:
            print("\n\n===== Compression Results =====")
            for k, v in comp_stats.items():
                print(f"{k}: {v}")
        print("\n\n===== Compression Complete =====")
        print(f"Compression stats saved to: {RESULTS_FILE}")
        print(f"Compression data saved to: {args.output_path}")

    elif args.mode == "decompress":
        # ========================
        # Validate input paths
        # ========================
        
        if not args.input_path:
            args.input_path = COMPRESSION_FILE
        elif not args.input_path.endswith(".bin"):
            args.input_path = COMPRESSION_FILE
        if not args.output_path:
            args.output_path = DECOMPRESSION_FILE
        # ========================
        # Load binary compression file
        # ========================
        print(f"\nLoading compression file: {args.input_path}")
        args, first_token, bit_string, bitmask_data = load_global_mask_file(args)

        exp_key = make_key(args)

        print("\n===== Loaded Header =====")
        print(f"  Model            : {args.model_name}")
        print(f"  Context length   : {args.context_length}")
        print(f"  Retain tokens    : {args.retain_tokens}")
        print(f"  First n tokens   : {args.first_n_tokens}")
        print(f"  Use KV cache     : {args.use_kv_cache}")
        print(f"  Batch size       : {args.batch_size}")

        print("\n===== Decompress Data =====")
        _, results, decomp_stats = run_global_mask_decompression(
            args=args,
            first_tokens=first_token,
            bit_string=bit_string,
            bitmap=bitmask_data
        )

        # ========================
        # Save results (JSON stats)
        # ========================
        results_db = load_results(RESULTS_FILE)
        exp_key = make_key(args)
        if exp_key not in results_db:
            results_db[exp_key] = {}
        results_db[exp_key]["decompression"] = decomp_stats
        save_results(results_db, RESULTS_FILE)

        if args.print_results:
            print("\n\n===== Decompression Results =====")
            for k, v in decomp_stats.items():
                print(f"{k}: {v}")

        # Save the reconstructed text to a file
        with open(args.output_path, "w") as f:
            f.write(results)

        print("\n\n===== Decompression Complete =====")
        print(f"Decompression stats saved to: {RESULTS_FILE}")
        print(f"Reconstructed text saved to: {args.output_path}")
        

if __name__ == "__main__":
    main()
