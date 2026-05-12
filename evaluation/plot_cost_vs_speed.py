#!/usr/bin/env python3
"""Plot total speed versus total cost from one or more main.py results JSON files."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ENGINE_HARDWARE_DEFAULTS = {
    "transformer": "gpu",
    "vllm": "gpu",
    "sglang": "gpu",
    "onnxruntime": "cpu",
    "tensorrt": "gpu",
    "llamacpp": "cpu",
    "llamacpp_direct": "cpu",
}

ENGINE_COLORS = {
    "transformer": "#1f77b4",
    "vllm": "#ff7f0e",
    "sglang": "#2ca02c",
    "onnxruntime": "#d62728",
    "tensorrt": "#9467bd",
    "llamacpp": "#8c564b",
    "llamacpp_direct": "#e377c2",
}

ENGINE_MARKERS = {
    "transformer": "o",
    "vllm": "s",
    "sglang": "^",
    "onnxruntime": "D",
    "tensorrt": "P",
    "llamacpp": "X",
    "llamacpp_direct": "v",
}


def parse_engine_map(entries, value_parser=str):
    """Parse repeated engine=value CLI entries into a dictionary."""
    parsed = {}
    for entry in entries:
        if "=" not in entry:
            raise ValueError(f"Expected engine=value, got: {entry}")
        engine, raw_value = entry.split("=", 1)
        parsed[engine.strip()] = value_parser(raw_value.strip())
    return parsed


def resolve_engine_hardware(overrides):
    """Merge hardware overrides into the default engine mapping."""
    hardware = dict(ENGINE_HARDWARE_DEFAULTS)
    hardware.update(overrides)
    return hardware


def resolve_engine_costs(engine_hardware, gpu_cost, cpu_cost, overrides):
    """Resolve an hourly cost for each engine."""
    costs = {}
    for engine, hardware_kind in engine_hardware.items():
        costs[engine] = gpu_cost if hardware_kind == "gpu" else cpu_cost
    costs.update(overrides)
    return costs


def build_plot_points(result_paths, engine_costs, engine_hardware, engines=None, models=None):
    """Load result JSON files and derive one point per completed experiment."""
    allowed_engines = set(engines or [])
    allowed_models = set(models or [])
    points = []

    for result_path in result_paths:
        payload = json.loads(Path(result_path).read_text(encoding="utf-8"))
        for experiment_key, result in payload.items():
            compression = result.get("compression")
            decompression = result.get("decompression")
            if not compression or not decompression:
                continue

            args = compression.get("args") or decompression.get("args") or {}
            engine = args.get("engine", "transformer")
            model_name = args.get("model_name", "unknown")

            if allowed_engines and engine not in allowed_engines:
                continue
            if allowed_models and model_name not in allowed_models:
                continue

            comp_time = compression.get("total_compression_time")
            decomp_time = decompression.get("total_decompression_time")
            input_tokens = compression.get("input_tokens_count")
            if not comp_time or not decomp_time or not input_tokens:
                continue

            total_time = comp_time + decomp_time
            if total_time <= 0:
                continue

            hourly_cost = engine_costs.get(
                engine,
                engine_costs.get(engine_hardware.get(engine, "gpu"), 0.0),
            )
            points.append(
                {
                    "engine": engine,
                    "model_name": model_name,
                    "label": f"{engine}:{model_name.split('/')[-1]}",
                    "results_file": str(result_path),
                    "experiment_key": experiment_key,
                    "compression_factor": compression.get("compression_factor"),
                    "total_cost_usd": total_time * hourly_cost / 3600,
                    "total_speed_tok_s": input_tokens / total_time,
                    "total_wall_time_s": total_time,
                    "input_tokens_count": input_tokens,
                }
            )

    return points


def write_summary(points, output_path):
    """Write the plotted points as a TSV for inspection."""
    header = [
        "engine",
        "model_name",
        "label",
        "total_cost_usd",
        "total_speed_tok_s",
        "total_wall_time_s",
        "input_tokens_count",
        "compression_factor",
        "results_file",
        "experiment_key",
    ]
    lines = ["\t".join(header)]
    for point in points:
        lines.append("\t".join(str(point.get(column, "")) for column in header))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_points(points, output_path, title, annotate=False, x_scale="linear", y_scale="linear"):
    """Render the cost-versus-speed scatter plot."""
    if not points:
        raise ValueError("No complete compression/decompression runs were found to plot")

    fig, ax = plt.subplots(figsize=(10, 6))
    engines = sorted({point["engine"] for point in points})

    for engine in engines:
        engine_points = [point for point in points if point["engine"] == engine]
        ax.scatter(
            [point["total_cost_usd"] for point in engine_points],
            [point["total_speed_tok_s"] for point in engine_points],
            c=ENGINE_COLORS.get(engine, "#333333"),
            marker=ENGINE_MARKERS.get(engine, "o"),
            s=90,
            alpha=0.85,
            label=engine,
            edgecolors="black",
            linewidths=0.4,
        )

        if annotate:
            for point in engine_points:
                ax.annotate(
                    point["label"],
                    (point["total_cost_usd"], point["total_speed_tok_s"]),
                    xytext=(5, 4),
                    textcoords="offset points",
                    fontsize=8,
                )

    ax.set_title(title)
    ax.set_xlabel("Total Cost (USD)")
    ax.set_ylabel("Total Speed (tokens/s)")
    ax.set_xscale(x_scale)
    ax.set_yscale(y_scale)
    ax.grid(True, alpha=0.25)
    ax.legend(title="Engine")
    fig.tight_layout()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot total speed versus total cost from benchmark results JSON files"
    )
    parser.add_argument(
        "--results-json",
        nargs="+",
        required=True,
        help="One or more results JSON files written by main.py",
    )
    parser.add_argument("--output", default="figures/cost_vs_speed.png", help="Output image path")
    parser.add_argument("--summary-output", default=None, help="Optional TSV summary of plotted points")
    parser.add_argument("--engines", nargs="+", default=None, help="Optional engine allow-list")
    parser.add_argument("--models", nargs="+", default=None, help="Optional model allow-list")
    parser.add_argument("--gpu-cost", type=float, default=0.804, help="Default hourly GPU cost in USD")
    parser.add_argument("--cpu-cost", type=float, default=0.019, help="Default hourly CPU cost in USD")
    parser.add_argument("--engine-cost", action="append", default=[], help="Override hourly cost with engine=usd")
    parser.add_argument("--engine-hardware", action="append", default=[], help="Override hardware class with engine=cpu|gpu")
    parser.add_argument("--annotate", action="store_true", help="Annotate each point with engine:model")
    parser.add_argument("--title", default="Cost vs Speed by Engine/Model", help="Plot title")
    parser.add_argument("--x-scale", choices=["linear", "log"], default="linear", help="X-axis scale")
    parser.add_argument("--y-scale", choices=["linear", "log"], default="linear", help="Y-axis scale")
    return parser.parse_args()


def main():
    args = parse_args()
    hardware_overrides = parse_engine_map(args.engine_hardware)
    hardware = resolve_engine_hardware(hardware_overrides)
    cost_overrides = parse_engine_map(args.engine_cost, float)
    costs = resolve_engine_costs(hardware, args.gpu_cost, args.cpu_cost, cost_overrides)

    points = build_plot_points(
        result_paths=args.results_json,
        engine_costs=costs,
        engine_hardware=hardware,
        engines=args.engines,
        models=args.models,
    )
    plot_points(
        points,
        output_path=args.output,
        title=args.title,
        annotate=args.annotate,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
    )
    if args.summary_output:
        write_summary(points, args.summary_output)

    print(f"Plotted {len(points)} points to {args.output}")
    if args.summary_output:
        print(f"Wrote summary TSV to {args.summary_output}")


if __name__ == "__main__":
    main()