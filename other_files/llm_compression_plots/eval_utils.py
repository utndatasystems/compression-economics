
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict

def load_runs_old(results_dir):
    records = []

    for run in os.listdir(results_dir):
        run_dir = os.path.join(results_dir, run)
        if not run.startswith("run_") or not os.path.isdir(run_dir):
            continue

        params_path = os.path.join(run_dir, "params.json")
        results_path = os.path.join(run_dir, "compression_results.json")

        if not (os.path.exists(params_path) and os.path.exists(results_path)):
            continue

        with open(params_path) as f:
            params = json.load(f)

        with open(results_path) as f:
            results = json.load(f)

        comp = results.get("compression", {})
        if not comp:
            continue

        record = {
            "run": run,
            **params,
            **comp,
        }
        records.append(record)

    return pd.DataFrame(records)


def load_run(results_path):
    if not (os.path.exists(results_path)):
        return pd.DataFrame([])

    with open(results_path) as f:
        results = json.load(f)

    comp = results.get("compression", {})
    #mod = params.get("model_params", {})

    #print(json.dumps(results, indent=2))
    if not comp:
        return pd.DataFrame([])

    #if not mod:
    #    return pd.DataFrame([])

    run_name = os.path.basename(os.path.dirname(results_path))

    record = {
        "run": run_name,
        # **params,
        **comp,
        #**mod,
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

    """for key, value in results.items():
        dataset_name, model_info = key.split(":", 1)
        parts = model_info.split("|")

        model_name = parts[0]
        n_value = None
        ctx_value = None
        ret_value = None
        batch_value = None

        # Extract ctx, ret, n, batch
        for p in parts:
            if p.startswith("n="):
                n_value = int(p.split("=")[1])
            elif p.startswith("ctx="):
                ctx_value = int(p.split("=")[1])
            elif p.startswith("ret="):
                ret_value = int(p.split("=")[1])
            elif p.startswith("batch="):
                batch_value = int(p.split("=")[1])

        # Filter datasets/n values if specified
        if selected_datasets and dataset_name not in selected_datasets:
            continue
        if selected_n and n_value not in selected_n:
            continue

        comp = value.get("compression", {})
        decomp = value.get("decompression", {})

        compression_time = comp.get("total_compression_time_sec", float("inf"))

        # Keep fastest compression time (smallest total_compression_time_sec)
        existing = temp_store[dataset_name].get(model_name)
        if not existing or compression_time < existing["compression_time"]:
            temp_store[dataset_name][model_name] = {
                "batch": batch_value,
                "original_size_bits": comp.get("original_size_bits", 0),
                "ac_bits": comp.get("arithmetic_code_size_bits", 0),
                "bitmap_bits": comp.get("bitmap_size_bits", 0),
                "compressed_size_bits": comp.get("final_size_bits", 0),
                "compression_time": compression_time,
                "decompression_time": decomp.get("decompression_time_sec", 0),
                "ctx": ctx_value,
                "ret": ret_value
            }

    # Convert to final output
    datasets = defaultdict(dict)
    for dataset_name, models in temp_store.items():
        for model_name, info in models.items():
            datasets[dataset_name][model_name] = info
    return datasets"""


def lift_compression_args(data):
    comp = data.get("compression")
    args = comp.pop("args", None)

    if isinstance(args, dict):
        comp.update(args)
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