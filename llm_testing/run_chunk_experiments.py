import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import math
from src.prediction import TokenPredictor
from src.llm_compressor import LLMCompressor
from src.global_mask_compressor import run_global_mask_compression



def run_no_masking_experiment(data_path, model_name="Qwen/Qwen2.5-0.5B", context_length=1024, first_n_tokens=10000):
    print(f"\n----- Running Experiment: No Token Masking (first_n_tokens={first_n_tokens}) -----")
    
    token_predictor = TokenPredictor(
        data_path=data_path,
        model_name=model_name,
        reduce_tokens=False,  # No masking
        first_n_tokens=first_n_tokens
    )
    
    llm_compressor = LLMCompressor()
    data_tokens = token_predictor.get_data_tokens()

    prompt_tokens = []
    for i in range(len(data_tokens) - 1):
        prompt_tokens.append(data_tokens[i])
        if len(prompt_tokens) > context_length:
            prompt_tokens.pop(0)

        print(f"\rProcessing token {i+1}/{len(data_tokens)}", end='')
        next_token_actual_index = data_tokens[i+1]
        
        _, probs_values = token_predictor.get_token_info(prompt_tokens)
        # probs_values_np = np.array(probs_values)
        probs_values_np = probs_values.cpu().numpy()
        
        llm_compressor.next_token(next_token_actual_index, probs_values_np)

    bit_string = llm_compressor.compress()
    final_size = len(bit_string)
    original_size = len(token_predictor.detokenize(data_tokens)) * 8

    print(f"\n--- Results for No Token Masking ---")
    print(f"Final Compressed Size: {final_size} bits")
    print(f"Overall Compression Ratio: {final_size / original_size * 100:.4f} %")

    return {
        "first_n_tokens": first_n_tokens,
        "chunk_size": 0,  # Special value for no masking
        "arithmetic_code_size_bits": final_size,
        "bitmap_size_bits": 0,
        "final_size_bits": final_size,
        "compression_ratio_percent": final_size / original_size * 100,
    }

def run_chunked_experiment(data_path, chunk_size, model_name="Qwen/Qwen2.5-0.5B", context_length=1024, first_n_tokens=10000):
    print(f"\n----- Running Experiment: first_n_tokens={first_n_tokens}, chunk_size={chunk_size} -----")
    
    # Initialize predictor with chunk_size
    token_predictor = TokenPredictor(
        data_path=data_path,
        model_name=model_name,
        reduce_tokens=True,
        chunk_size=chunk_size,
        first_n_tokens=first_n_tokens
    )
    
    llm_compressor = LLMCompressor()
    data_tokens = token_predictor.get_data_tokens()
    num_chunks = math.ceil(len(data_tokens) / chunk_size)

    total_arithmetic_code_size = 0
    total_bitmap_size = 0
    current_chunk_index = -1

    prompt_tokens = []
    for i in range(len(data_tokens) - 1):
        # Determine the chunk index for the *next* token to be compressed
        next_token_chunk_index = (i + 1) // chunk_size

        # If the next token is in a new chunk, update the active token mask
        if next_token_chunk_index != current_chunk_index:
            _, bitmap_size_bytes = token_predictor.set_active_chunk(next_token_chunk_index)
            total_bitmap_size += bitmap_size_bytes * 8  # convert bytes to bits
            current_chunk_index = next_token_chunk_index

        prompt_tokens.append(data_tokens[i])
        if len(prompt_tokens) > context_length:
            prompt_tokens.pop(0)

        print(f"\rProcessing token {i+1}/{len(data_tokens)}", end='')
        next_token_actual_index = data_tokens[i+1]

        # Get probabilities using the correct, active token mask
        candidate_token_ids, probs_values = token_predictor.get_token_info(prompt_tokens)
        # probs_values_np = np.array(probs_values)
        probs_values_np = probs_values.cpu().numpy()

        # With the corrected logic, this ValueError should no longer occur,
        # as the token is guaranteed to be in the vocabulary of its own chunk.
        try:
            token_idx_for_compressor = candidate_token_ids.index(next_token_actual_index)
        except ValueError:
            # This is now a critical error, as it indicates a flaw in the logic.
            print(f"\nFATAL: Token {next_token_actual_index} not in the vocabulary for its own chunk {current_chunk_index}. This should not happen.")
            # For safety, we stop compression here as it would lead to data corruption
            break

        llm_compressor.next_token(token_idx_for_compressor, probs_values_np)

    bit_string = llm_compressor.compress()
    total_arithmetic_code_size = len(bit_string)

    final_size = total_arithmetic_code_size + total_bitmap_size
    original_size = len(token_predictor.detokenize(data_tokens)) * 8

    print(f"\n--- Results for first_n_tokens={first_n_tokens}, chunk_size={chunk_size} ---")
    print(f"Total Arithmetic Code Size: {total_arithmetic_code_size} bits")
    print(f"Total Bitmap Size: {total_bitmap_size} bits")
    print(f"Final Compressed Size: {final_size} bits")
    print(f"Original Size: {original_size} bits")
    print(f"Overall Compression Ratio: {final_size / original_size * 100:.4f} %")

    return {
        "first_n_tokens": first_n_tokens,
        "chunk_size": chunk_size,
        "arithmetic_code_size_bits": total_arithmetic_code_size,
        "bitmap_size_bits": total_bitmap_size,
        "final_size_bits": final_size,
        "compression_ratio_percent": final_size / original_size * 100,
    }

def main():
    parser = argparse.ArgumentParser(description="Run chunked LLM compression experiments and plot results.")
    parser.add_argument("--data_path", type=str, required=True, help="Input text file path.")
    parser.add_argument("--output_plot", type=str, default="chunked_compression_plot.png", help="Base name for the output plot file.")
    parser.add_argument("--output_json", type=str, default="chunked_compression_results.json", help="Output JSON file to store results.")
    parser.add_argument("--chunk_sizes", type=int, nargs='+', default=None, help="List of chunk sizes to test. If not provided, only baseline experiments are run.")
    parser.add_argument("--first_n_tokens", type=int, default=100000, help="Total number of tokens to process from the beginning of the file.")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache for global mask experiment.")
    parser.add_argument("--context_length", type=int, default=1024, help="Context length for experiments.")
    parser.add_argument("--retain_tokens", type=int, default=1000, help="Number of tokens to retain when using KV cache.")
    parser.add_argument("--skip_no_masking", action="store_true", help="Skip the 'no masking' baseline experiment.")


    args = parser.parse_args()

    results = []
    if os.path.exists(args.output_json):
        with open(args.output_json, 'r') as f:
            try:
                results = json.load(f)
                print(f"Loaded {len(results)} existing results from {args.output_json}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {args.output_json}. Starting with empty results.")

    completed_experiments = {(d.get('first_n_tokens'), d.get('chunk_size')) for d in results}






    if not args.skip_no_masking and (args.first_n_tokens, 0) not in completed_experiments:
        no_masking_result = run_no_masking_experiment(
            data_path=args.data_path, 
            first_n_tokens=args.first_n_tokens,
            context_length=args.context_length
        )
        results.append(no_masking_result)
        with open(args.output_json, 'w') as f: json.dump(results, f, indent=4)

    if not args.chunk_sizes:
        if (args.first_n_tokens, -1) not in completed_experiments:
            _, _, global_mask_result = run_global_mask_compression(
                data_path=args.data_path, 
                first_n_tokens=args.first_n_tokens,
                use_kv_cache=args.use_kv_cache
            )
            results.append(global_mask_result)
            with open(args.output_json, 'w') as f: json.dump(results, f, indent=4)

    if args.chunk_sizes:
        for chunk_size in args.chunk_sizes:
            if (args.first_n_tokens, chunk_size) in completed_experiments:
                print(f"Skipping experiment for first_n_tokens={args.first_n_tokens}, chunk_size={chunk_size} as it already exists.")
                continue
            if chunk_size > args.first_n_tokens:
                print(f"Skipping chunk size {chunk_size} as it is larger than first_n_tokens ({args.first_n_tokens})")
                continue
            
            result = run_chunked_experiment(
                data_path=args.data_path,
                chunk_size=chunk_size,
                first_n_tokens=args.first_n_tokens,
                context_length=args.context_length
            )
            results.append(result)

            with open(args.output_json, 'w') as f: json.dump(results, f, indent=4)

    current_results = [r for r in results if r.get('first_n_tokens') == args.first_n_tokens]
    
    if args.chunk_sizes:
        plot_full_chart(current_results, args.first_n_tokens, args.output_plot)
    else:
        plot_baseline_chart(current_results, args.first_n_tokens, args.output_plot)

def plot_full_chart(current_results, first_n_tokens, output_plot_base):
    no_masking_data = next((r for r in current_results if r.get('chunk_size') == 0), None)
    global_mask_data = next((r for r in current_results if r.get('chunk_size') == -1), None)
    plot_results = [r for r in current_results if r.get('chunk_size') > 0]

    if not plot_results: return

    plot_results.sort(key=lambda d: d['chunk_size'])
    
    all_plot_data = plot_results + ([global_mask_data] if global_mask_data else [])
    chunk_labels = [str(d['chunk_size']) if d['chunk_size'] != -1 else 'Global Mask' for d in all_plot_data]
    
    arithmetic_sizes_bytes = [d['arithmetic_code_size_bits'] / 8 for d in all_plot_data]
    bitmap_sizes_bytes = [d['bitmap_size_bits'] / 8 for d in all_plot_data]
    compression_ratios = [d['compression_ratio_percent'] for d in all_plot_data]

    plt.figure(figsize=(14, 8))
    # Plot the bars
    plt.bar(chunk_labels, arithmetic_sizes_bytes, color='#4c72b0', label='Arithmetic Code Size')
    plt.bar(chunk_labels, bitmap_sizes_bytes, bottom=arithmetic_sizes_bytes, color='#dd8452', label='Bitmap Size')

    # Add text labels
    for i in range(len(chunk_labels)):
        arith_size_bytes = arithmetic_sizes_bytes[i]
        bitmap_size_bytes = bitmap_sizes_bytes[i]
        total_size_bytes = arith_size_bytes + bitmap_size_bytes
        ratio = compression_ratios[i]

        # Add overall compression ratio on top
        plt.text(i, total_size_bytes, f'{ratio:.2f}%', ha='center', va='bottom', fontsize=9)

        if total_size_bytes > 0:
            # Label for Arithmetic Code part
            if arith_size_bytes > 0:
                arith_percent = (arith_size_bytes / total_size_bytes) * 100
                plt.text(i, arith_size_bytes / 2, f"{arith_size_bytes/1024:.1f} KB\n({arith_percent:.1f}%)",
                         ha='center', va='center', color='white', fontsize=8, fontweight='bold')

            # Label for Bitmap part
            if bitmap_size_bytes > 0:
                bitmap_percent = (bitmap_size_bytes / total_size_bytes) * 100
                plt.text(i, arith_size_bytes + (bitmap_size_bytes / 2), f"{bitmap_size_bytes/1024:.1f} KB\n({bitmap_percent:.1f}%)",
                         ha='center', va='center', color='white', fontsize=8, fontweight='bold')

    if no_masking_data:
        plt.axhline(y=no_masking_data['final_size_bits'] / 8, color='red', linestyle='--', 
                    label=f'No Masking Baseline ({no_masking_data["compression_ratio_percent"]:.2f}%)')

    plt.xlabel('Chunk Size (Number of Tokens)')
    plt.ylabel('Total Size (Bytes)')
    plt.title(f'Compression Size Breakdown by Chunk Size (First {first_n_tokens} Tokens)')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = output_plot_base.replace('.png', f'_{first_n_tokens}.png')
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")

def plot_baseline_chart(current_results, first_n_tokens, output_plot_base):
    no_masking_data = next((r for r in current_results if r.get('chunk_size') == 0), None)
    global_mask_data = next((r for r in current_results if r.get('chunk_size') == -1), None)

    if not no_masking_data and not global_mask_data:
        print("No baseline data available to plot.")
        return

    plt.figure(figsize=(8, 6))

    # Plot Global Mask as a stacked bar if it exists
    if global_mask_data:
        labels = ['Global Mask']
        arithmetic_size_bytes = global_mask_data['arithmetic_code_size_bits'] / 8
        bitmap_size_bytes = global_mask_data['bitmap_size_bits'] / 8
        total_size_bytes = arithmetic_size_bytes + bitmap_size_bytes
        ratio = global_mask_data['compression_ratio_percent']

        plt.bar(labels, [arithmetic_size_bytes], color='#4c72b0', label='Arithmetic Code Size')
        plt.bar(labels, [bitmap_size_bytes], bottom=[arithmetic_size_bytes], color='#dd8452', label='Bitmap Size')

        # Add overall compression ratio on top
        plt.text(0, total_size_bytes, f'{ratio:.2f}%', ha='center', va='bottom', fontsize=10)

        if total_size_bytes > 0:
            if arithmetic_size_bytes > 0:
                arith_percent = (arithmetic_size_bytes / total_size_bytes) * 100
                plt.text(0, arithmetic_size_bytes / 2, f"{arithmetic_size_bytes/1024:.1f} KB\n({arith_percent:.1f}%)",
                         ha='center', va='center', color='white', fontsize=9, fontweight='bold')
            if bitmap_size_bytes > 0:
                bitmap_percent = (bitmap_size_bytes / total_size_bytes) * 100
                plt.text(0, arithmetic_size_bytes + (bitmap_size_bytes / 2), f"{bitmap_size_bytes/1024:.1f} KB\n({bitmap_percent:.1f}%)",
                         ha='center', va='center', color='white', fontsize=9, fontweight='bold')

    if no_masking_data:
        plt.axhline(y=no_masking_data['final_size_bits'] / 8, color='red', linestyle='--', 
                    label=f'No Masking Baseline ({no_masking_data["compression_ratio_percent"]:.2f}%)')

    plt.ylabel('Total Size (Bytes)')
    plt.title(f'Baseline Comparison (First {first_n_tokens} Tokens)')
    if not global_mask_data:
        plt.xticks([])
    
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()

    plot_filename = output_plot_base.replace('.png', f'_baselines_{first_n_tokens}.png')
    plt.savefig(plot_filename)
    print(f"\nPlot saved to {plot_filename}")


if __name__ == "__main__":
    main()