
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import re


def load_run(results_path):
    if not (os.path.exists(results_path)):
        return pd.DataFrame([])

    with open(results_path) as f:
        results = json.load(f)

    comp = results.get("compression", {})
    if not comp:
        return pd.DataFrame([])

    run_name = os.path.basename(os.path.dirname(results_path))

    record = {
        "run": run_name,
        **comp,
    }
    print('done')
    return pd.DataFrame([record])


def load_model_results(file_path, selected_datasets=None, selected_n=None):
    """
    Loads model JSON results and flattens into a dictionary of:
        dataset → model → compression/decompression data.
    
    Keeps only the **fastest compression time** entry for each dataset+model.
    Adds ctx and ret to the output.
    Filters results by dataset and n if specified.
    """
    with open(file_path, "r") as f:
        results = json.load(f)

    # Temporary store: dataset → model → best entry (fastest compression time)
    temp_store = defaultdict(lambda: defaultdict(dict))

    #TODO: convert temp_store to final output format in a separate step, to avoid nested defaultdict in output

    print(results)
    return results


def lift_compression_args(data):
    comp = data.get("compression")
    args = comp.pop("args", None)

    if isinstance(args, dict):
        comp.update(args)
    return data


def add_rank(data: dict) -> dict:
    """
    Given a dictionary of runs, extract the rank from the experiment key
    and add it as a new 'rank' key in each sub-dictionary.

    Parameters:
        data (dict): Dictionary of experiment runs.

    Returns:
        dict: The same dictionary with 'rank' added to each run.
    """
    rank_pattern = re.compile(r"/r(\d+)_")

    for key, value in data.items():
        match = rank_pattern.search(key)
        if match:
            value["rank"] = int(match.group(1))
        else:
            value["rank"] = None

    return data

def include_peft(results: dict) -> dict:
    for key, value in results.items():
        # Only process experiment entries (dicts with lora_path)
        if not isinstance(value, dict):
            continue

        lora_path = value.get("lora_path")

        if not isinstance(lora_path, str):
            value["peft"] = None
            continue

        path_lower = lora_path.lower()

        if "vera" in path_lower:
            value["peft"] = "vera"
        elif "lora" in path_lower:
            value["peft"] = "lora"
        else:
            value["peft"] = None

    return results