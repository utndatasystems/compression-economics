import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import argparse

from sympy import comp

from other_files.llm_compression_plots.eval_utils import load_model_results, lift_compression_args, include_peft
    
def main():
    parser = argparse.ArgumentParser("Plot compression experiment results")
    parser.add_argument("--results_file", type=str, default=None, help="Directory containing run subdirectories with results")
    parser.add_argument("--x", type=str, default="total_params", help="X-axis variable (e.g., first_n_tokens, retain_tokens)")
    parser.add_argument("--y", type=str, default="compression_factor", help="Y-axis variable (e.g., compression_factor, bits_per_token)")
    #parser.add_argument("--y", type=str, default="bits_per_token", help="Y-axis variable (e.g., bits_per_token, compression_ratio)")
    #parser.add_argument("--groupby", type=str, default="peft", help="Variable to group by for separate lines (e.g., encoding, model_name)") # encoding

    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()

    if not args.results_file:
        # take current directory
        args.results_file = "compression_results.json"

    print(f"Loading results from {args.results_file}")    
    dict_results = load_model_results(args.results_file)

    # lift compression args to top level for easier plotting
    dict_results = {k: lift_compression_args(v) for k, v in dict_results.items()}

    # remove 'compression' level from dict_results for easier plotting
    dict_results = {k: v.get("compression", {}) for k, v in dict_results.items()}
    
    dict_results = include_peft(dict_results)

    #save dict_results to json for inspection
    with open("loaded_results.json", "w") as f:
        json.dump(dict_results, f, indent=2)

    print(f"Loaded {len(dict_results)} runs")

    # convert dictionary to df columns
    df = pd.DataFrame.from_dict(dict_results, orient="index")
    df = df.reset_index()
    df.rename(columns={"index": "run_key"}, inplace=True)
    print(df.head())

    #plt.figure(figsize=(8, 5))

    for key in df["run_key"].unique():
        # plot all runs 
        run_df = df[df["run_key"] == key]
        plt.plot(run_df[args.x], run_df[args.y], marker="o",
                    label=key)
        

    plt.xlabel(args.x)
    plt.ylabel(args.y)
    plt.title(f"{args.y} vs {args.x}")

    # put legend outside of plot, below plot
    plt.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=1)
    plt.grid(True)

    plt.savefig("plot", bbox_inches="tight")

    #if args.save:
    #    # TODO: fix path handling
    #    plt.savefig(args.save, bbox_inches="tight")
    #    print(f"Saved figure to {args.save}")
    #else:
    #    plt.show()


if __name__ == "__main__":
    main()