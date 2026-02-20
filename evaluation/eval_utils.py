
import os
import json
import argparse
import pandas as pd
import matplotlib.pyplot as plt

def load_runs(results_dir):
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