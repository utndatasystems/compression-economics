"""
CLI entry point for running global-mask compression and decompression experiments.
"""

import json
import os
from pathlib import Path

from src.global_mask_compressor import run_global_mask_compression, run_global_mask_decompression, run_global_mask_speculative_decompression
from src.config import get_main_args
from src.utils import save_global_mask_file, load_global_mask_file, load_results, save_results, make_key, create_run_dir, save_params, check_mismatch
from src.prediction import TokenPredictor

RUN_DIR = Path("artifacts/runs/current")
RESULTS_FILE = str(RUN_DIR / "compression_results.json")
COMPRESSION_FILE = str(RUN_DIR / "compression_data.bin")
DECOMPRESSION_FILE = str(RUN_DIR / "text_results.txt")

def main():
    """
    Parse CLI arguments and run compression or decompression.
    """
    # ========================
    # Parse command-line arguments
    # ========================

    args = get_main_args()
    RUN_DIR.mkdir(parents=True, exist_ok=True)

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
            print(f"  Encoding         : {args.encoding}")
        
            # add parameters to comp_stats for saving in results JSON
            token_predictor = TokenPredictor(args, bitmap_data=None)
            base_params, adapter_params = token_predictor.base_params, token_predictor.adapter_params
            base_size_mb, adapter_size_mb = token_predictor.base_size_mb, token_predictor.adapter_size_mb
            total_params = base_params + adapter_params
            total_size_mb = base_size_mb + adapter_size_mb

            print(f'\nModel parameters:')
            print(f"Adapter parameters   : {adapter_params:,}")
            print(f"Base model parameters: {base_params:,}")
            print(f"Adapter size (MB).   : {adapter_size_mb:.2f}")
            print(f"Base model size (MB).: {base_size_mb:.2f}")

            # ========================
            # Run compression
            # ========================
            first_token, bit_string, bitmask_data, comp_stats, args = run_global_mask_compression(args)

            comp_stats = {
                **comp_stats,
                "total_params": total_params,
                "adapter_params": adapter_params,
                "base_model_params": base_params,
                "total_size_mb": round(total_size_mb, 2),
                "adapter_size_mb": round(adapter_size_mb, 2),
                "base_model_size_mb": round(base_size_mb, 2),}
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
