import argparse
import json
import os
import matplotlib.pyplot as plt
import numpy as np
from run_compression import run_compression_analysis

def main():
    parser = argparse.ArgumentParser(description="Run LLM compression experiments and plot results.")
    parser.add_argument("--data_path", type=str, default=None,
                        help="The input text file path for LLM inference.")
    parser.add_argument("--text_input", type=str, default=None,
                        help="The direct text input for LLM inference.")
    parser.add_argument("--output_json", type=str, default="compression_results.json",
                        help="Output JSON file to store results.")
    parser.add_argument("--output_plot", type=str, default="compression_plot.png",
                        help="Output file for the plot.")

    args = parser.parse_args()

    if args.data_path is None and args.text_input is None:
        parser.error("Either --data_path or --text_input must be provided.")
    if args.data_path and args.text_input:
        parser.error("Only one of --data_path or --text_input can be provided.")

    # Define experiment settings
    experiment_settings = [
        {"roaring_ranking_n": None, "reduce_tokens": False},
        {"roaring_ranking_n": None, "reduce_tokens": True},
        {"roaring_ranking_n": 5, "reduce_tokens": True},
        {"roaring_ranking_n": 10, "reduce_tokens": True},
        {"roaring_ranking_n": 100, "reduce_tokens": True},
        {"roaring_ranking_n": 1000, "reduce_tokens": True},
        {"roaring_ranking_n": 5000, "reduce_tokens": True},
        {"roaring_ranking_n": 10000, "reduce_tokens": True},
    ]

    # Load existing results if the JSON file exists
    results = []
    if os.path.exists(args.output_json):
        with open(args.output_json, 'r') as f:
            try:
                results = json.load(f)
                print(f"Loaded {len(results)} existing results from {args.output_json}")
            except json.JSONDecodeError:
                print(f"Warning: Could not decode JSON from {args.output_json}. Starting with empty results.")

    # Create a set of completed settings for efficient lookup
    completed_settings = set()
    for res in results:
        setting_tuple = (res.get('roaring_ranking_n'), res.get('reduce_tokens'))
        completed_settings.add(setting_tuple)

    # Determine which settings still need to be run
    settings_to_run = []
    for setting in experiment_settings:
        setting_tuple = (setting.get('roaring_ranking_n'), setting.get('reduce_tokens'))
        if setting_tuple not in completed_settings:
            settings_to_run.append(setting)

    if not settings_to_run:
        print("All specified experiment settings have already been run. Nothing to do.")
    else:
        print(f"Remaining settings to run: {settings_to_run}")

    for i, setting in enumerate(settings_to_run):
        print(f"\n--- Running experiment {i+1}/{len(settings_to_run)}: {setting} ---")
        try:
            result = run_compression_analysis(
                data_path=args.data_path,
                text_input=args.text_input,
                roaring_ranking_n=setting["roaring_ranking_n"],
                reduce_tokens=setting["reduce_tokens"],
                model_name="Qwen/Qwen2.5-0.5B",
                first_n_tokens=10000,
                context_length=1000
            )
            results.append(result)

            # Save results to JSON after each experiment
            with open(args.output_json, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"\nSaved {len(results)} results to {args.output_json}")

        except Exception as e:
            print(f"Error running experiment {setting}: {e}")


    # Plotting
    if results:
        # Sort results by roaring_ranking_n for proper plotting
        results.sort(key=lambda d: d['roaring_ranking_n'] if d['roaring_ranking_n'] is not None else -1)
        x_sorted = [d['roaring_ranking_n'] if d['roaring_ranking_n'] is not None else 0 for d in results]

        bitmap_sizes = [d['bitmap_size_bits'] for d in results]
        final_sizes = [d['final_size_bits'] for d in results]

        plt.figure(figsize=(10, 6))
        plt.plot(x_sorted, bitmap_sizes, label='Bitmap Size', color='skyblue', marker='o')
        plt.plot(x_sorted, final_sizes, label='Total Size', color='lightgreen', marker='o')

        # Add horizontal lines for baseline comparisons (dynamically)
        roaring_distinct_result = next((item for item in results if item["roaring_ranking_n"] is None and item["reduce_tokens"] is True), None)
        no_masking_result = next((item for item in results if item["roaring_ranking_n"] is None and item["reduce_tokens"] is False), None)

        if roaring_distinct_result:
            plt.axhline(y=roaring_distinct_result['final_size_bits'], color='red', linestyle='--',
                        label=f'Roaring on distinct token_id ({roaring_distinct_result["compression_ratio_percent"]:.4f} % compression ratio)')
        if no_masking_result:
            plt.axhline(y=no_masking_result['final_size_bits'], color='orange', linestyle='--',
                        label=f'No token masking ({no_masking_result["compression_ratio_percent"]:.4f} % compression ratio)')

        plt.xlabel('Average probability taken from first N tokens (for Roaring Ranking)')
        plt.ylabel('Size (bits)')
        plt.title('Roaring Bitmap Storage Size Breakdown')
        plt.legend()
        plt.grid(axis='y')
        plt.xticks(x_sorted, [str(x) for x in x_sorted], rotation=45)
        plt.tight_layout()
        plt.savefig(args.output_plot)
        print(f"Plot saved to {args.output_plot}")
    else:
        print("No results to plot.")

if __name__ == "__main__":
    main()
