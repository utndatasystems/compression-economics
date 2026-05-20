"""
CLI entry point for running global-mask compression and decompression experiments.
"""

import json
import os

from src.global_mask_compressor import run_global_mask_compression, run_global_mask_decompression, run_global_mask_speculative_decompression
from src.config import get_main_args
from src.utils import save_global_mask_file, load_global_mask_file, load_results, save_results, make_key, create_run_dir, save_params, check_mismatch
from src.prediction import get_token_predictor

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

    args = get_main_args()

    # Set Hugging Face token as environment variable if provided
    if args.HF_token is not None:
        print("Setting Hugging Face token from command-line argument.")
        os.environ["HF_TOKEN"] = args.HF_token
        os.environ["HUGGINGFACE_HUB_TOKEN"] = args.HF_token  

    # delete HF_token from args to avoid saving it in results JSON
    args.HF_token = None
    
    if args.mode == "compress":
            # ========================
            # Validate input paths
            if not args.input_path:
                raise ValueError("--input_path is required in compress mode")
            if not args.output_path:
                args.output_path = COMPRESSION_FILE

            # ========================
            # Check if experiment already exists
            # ========================
            results_db = load_results(RESULTS_FILE)
            exp_key = make_key(args)

            if exp_key in results_db and not args.force:
                print(f"\n⚠️  Experiment already exists for {exp_key}, skipping run.")
                print(f"Stored Results: {results_db[exp_key]}")
                print("Use --force to rerun and overwrite the stored results.")
                return
            elif exp_key in results_db and args.force:
                print(f"\n⚠️  Experiment already exists for {exp_key}, but --force was set. Rerunning.")

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
            print(f"  vLLM window size : {getattr(args, 'vllm_window_size', 1)}")
            print(f"  Engine           : {args.engine}")
            print(f"  Encoding         : {args.encoding}")
            if args.encoding == "AC_FAST":
                print(f"  AC_FAST backend  : {getattr(args, 'ac_fast_backend', 'auto')}")
        
            # We will retrieve model size stats *after* compression to avoid double-initializing the engine,
            # especially important for vLLM which aggressively reserves GPU memory.

            # ========================
            # Run compression
            # ========================
            compression_result = run_global_mask_compression(args)
            if len(compression_result) == 6:
                first_token, bit_string, bitmask_data, comp_stats, args, model_stats = compression_result
            else:
                first_token, bit_string, bitmask_data, comp_stats, args = compression_result
                token_predictor = get_token_predictor(args, bitmap_data=bitmask_data)
                try:
                    model_stats = {
                        "base_params": getattr(token_predictor, "base_params", 0),
                        "adapter_params": getattr(token_predictor, "adapter_params", 0),
                        "base_size_mb": getattr(token_predictor, "base_size_mb", 0.0),
                        "adapter_size_mb": getattr(token_predictor, "adapter_size_mb", 0.0),
                    }
                finally:
                    cleanup = getattr(token_predictor, "cleanup", None)
                    if callable(cleanup):
                        cleanup()

            total_params = model_stats["base_params"] + model_stats["adapter_params"]
            total_size_mb = model_stats["base_size_mb"] + model_stats["adapter_size_mb"]

            print(f'\nModel parameters:')
            print(f"Adapter parameters   : {model_stats['adapter_params']:,}")
            print(f"Base model parameters: {model_stats['base_params']:,}")
            print(f"Adapter size (MB).   : {model_stats['adapter_size_mb']:.2f}")
            print(f"Base model size (MB).: {model_stats['base_size_mb']:.2f}")

            comp_stats = {
                **comp_stats,
                "engine": args.engine,
                "total_params": total_params,
                "adapter_params": model_stats["adapter_params"],
                "base_model_params": model_stats["base_params"],
                "total_size_mb": round(total_size_mb, 2),
                "adapter_size_mb": round(model_stats["adapter_size_mb"], 2),
                "base_model_size_mb": round(model_stats["base_size_mb"], 2),}
            # TODO: add model dtype information

            # ========================
            # Save results (JSON stats)
            # ========================
            results_db = load_results(RESULTS_FILE)
            exp_key = make_key(args)
            if exp_key not in results_db:
                results_db[exp_key] = {}
            results_db[exp_key]["compression"] = comp_stats
            save_results(results_db, RESULTS_FILE) #add 

            # ========================
            # Save binary compression file
            # ========================
            save_global_mask_file(
                args,
                first_token=first_token,
                bit_string=bit_string,
                bitmask_data=bitmask_data)

            # ========================
            # Output compression results
            # ========================
            if args.print_results:
                print("\n\n===== Compression Results =====")
                for k, v in comp_stats.items():
                    print(f"{k}: {v}")
            print("\n\n===== Compression Complete =====")
            print(f"Compression stats saved to: {RESULTS_FILE}")
            #print(f"Compression data saved to: {args.output_path}")

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
        if args.spec_k is not None:
            spec_k = args.spec_k

        # ========================
        # Load binary compression file
        # ========================
        print(f"\nLoading compression file: {args.input_path}")
        args, first_token, bit_string, bitmask_data = load_global_mask_file(args)
        args.spec_k = spec_k if args.spec_k is not None else None

        exp_key = make_key(args)

        print("\n===== Loaded Header =====")
        print(f"  Model            : {args.model_name}")
        print(f"  Context length   : {args.context_length}")
        print(f"  Retain tokens    : {args.retain_tokens}")
        print(f"  First n tokens   : {args.first_n_tokens}")
        print(f"  Use KV cache     : {args.use_kv_cache}")
        print(f"  Batch size       : {args.batch_size}")
        print(f"  Engine           : {args.engine}")
        print(f"  Spec_k           : {args.spec_k}")

        print("\n===== Decompress Data =====")

        # NEW 
        if args.spec_k is not None:
            print(f"\nUsing spec_k = {args.spec_k} for draft token generation.")

            _, results, decomp_stats = run_global_mask_speculative_decompression(
                args=args,
                first_tokens=first_token,
                bit_string=bit_string,
                bitmap=bitmask_data)
            
        else:
            print(f"\nRunning standard decompression without speculative draft generation.")
            _, results, decomp_stats = run_global_mask_decompression(
                args=args,
                first_tokens=first_token,
                bit_string=bit_string,
                bitmap=bitmask_data)

        decomp_stats = {
            **decomp_stats,
            "engine": args.engine,
        }

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

        # Check if decompressed text matches original input (if available)
        if check_mismatch(input_path=args.input_path, output_path=args.output_path, first_n_tokens=args.first_n_tokens) == False:
            print("\n⚠️  Warning: Decompressed text does not match original input!")
        else:
            print("\n✅ Decompressed text matches original input.")

        print("\n\n===== Decompression Complete =====")
        print(f"Decompression stats saved to: {RESULTS_FILE}")
        print(f"Reconstructed text saved to: {args.output_path}")
        

if __name__ == "__main__":
    main()
