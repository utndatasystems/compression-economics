import os
import json
import re
from collections import defaultdict

import pandas as pd


def load_run(results_path: str) -> pd.DataFrame:
    """
    Load a single experiment run from a JSON results file.

    Parameters:
        results_path (str): Path to the JSON results file.

    Returns:
        pd.DataFrame: DataFrame with one row containing compression data.
                      Returns empty DataFrame if file or data does not exist.
    """
    if not os.path.exists(results_path):
        return pd.DataFrame([])

    with open(results_path, "r") as f:
        results = json.load(f)

    compression_data = results.get("compression", {})
    if not compression_data:
        return pd.DataFrame([])

    run_name = os.path.basename(os.path.dirname(results_path))
    record = {"run": run_name, **compression_data}

    return pd.DataFrame([record])


def load_model_results(file_path: str, selected_datasets: list = None, selected_n: int = None) -> dict:
    """
    Load and flatten model JSON results.

    Keeps only the **fastest compression time** entry for each dataset+model.
    Filters results by dataset and n if specified.

    Parameters:
        file_path (str): Path to the results JSON file.
        selected_datasets (list, optional): List of dataset names to filter.
        selected_n (int, optional): Number of tokens to filter by.

    Returns:
        dict: Flattened results dictionary: dataset → model → compression/decompression data.
    """
    if not os.path.exists(file_path):
        return {}

    with open(file_path, "r") as f:
        results = json.load(f)

    # Temporary store: dataset → model → best entry (fastest compression time)
    temp_store = defaultdict(lambda: defaultdict(dict))

    # TODO: implement filtering and flattening logic
    # for dataset_name, dataset_data in results.items():
    #     for model_name, runs in dataset_data.items():
    #         select fastest compression time, filter by selected_datasets and selected_n

    return results


def lift_compression_args(data: dict) -> dict:
    """
    Lift the 'args' dictionary inside compression data to top-level keys.

    Parameters:
        data (dict): Dictionary containing 'compression' entry.

    Returns:
        dict: Updated dictionary with args merged into compression data.
    """
    compression = data.get("compression")
    if not compression:
        return data

    args = compression.pop("args", None)
    if isinstance(args, dict):
        compression.update(args)

    return data


def add_dataset(data: dict) -> dict:
    """
    Extract dataset name from experiment key and add as 'dataset'.

    The pattern expected is "<dataset>:" in the experiment key.

    Parameters:
        data (dict): Dictionary of experiment runs.

    Returns:
        dict: Updated dictionary with 'dataset' key added to each run.
    """
    dataset_pattern = re.compile(r"([^:/]+):")

    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        match = dataset_pattern.search(key)
        value["dataset"] = match.group(1) if match else None

    return data


def add_rank(data: dict) -> dict:
    """
    Extract the experiment rank from experiment key and add as 'rank'.

    The pattern expected is "/r<rank>_" in the experiment key.

    Parameters:
        data (dict): Dictionary of experiment runs.

    Returns:
        dict: Updated dictionary with 'rank' key added to each run.
    """
    rank_pattern = re.compile(r"/r(\d+)_")

    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        match = rank_pattern.search(key)
        value["rank"] = int(match.group(1)) if match else None

    return data


def include_peft(results: dict) -> dict:
    """
    Annotate each run with PEFT type based on 'lora_path'.

    Rules:
        - 'vera' in path → "vera"
        - 'lora' in path → "lora"
        - None or missing → None

    Parameters:
        results (dict): Dictionary of experiment runs.

    Returns:
        dict: Updated dictionary with 'peft' key for each run.
    """
    for run_data in results.values():
        if not isinstance(run_data, dict):
            continue

        lora_path = run_data.get("lora_path")
        if not isinstance(lora_path, str):
            run_data["peft"] = None
            continue

        path_lower = lora_path.lower()
        if "vera" in path_lower:
            run_data["peft"] = "vera"
        elif "lora" in path_lower:
            run_data["peft"] = "lora"
        else:
            run_data["peft"] = None

    return results