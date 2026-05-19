import os
import argparse
from pathlib import Path

ENGINES = {
    "vllm": {
        "engine_arg": "vllm",
        "env": "vllm-env"
    },
    "sglang": {
        "engine_arg": "sglang",
        "env": "sglang-env"
    },
    "transformers": {
        "engine_arg": "transformer",
        "env": "transformers-env"
    },
    "tensorrt-llm": {
        "engine_arg": "tensorrt",
        "env": "tensorrt-llm-env"
    },
    "lamacpp-gpu": {
        "engine_arg": "llamacpp_direct",
        "env": "lamacpp-gpu-env",
        "extra_args": "--llamacpp_n_gpu_layers 99"
    },
}

MODELS = [
    "Qwen2.5-0.5B",
    "Qwen2.5-1.5B",
    "Qwen2.5-7B",
    "Qwen3-0.6B",
    "Qwen3-1.7B",
    "Qwen3-8B"
]

DATASETS = {
    "text8": "data/text8",
    "combined_100mb": "data/combined_100mb.py",
    "yelp_150mb": "data/yelp_150mb.json"
}

BATCH_SIZE = 256
CONTEXT_LENGTH = 128
# FIRST_N_TOKENS = 50000000 
FIRST_N_TOKENS = 1000

JOBS_DIR_ROOT = Path("jobs")

def build_model_args(engine_key, model_name):
    if engine_key in ["lamacpp-gpu", "lamacpp-cpu"]:
        return f"--model_name Qwen/{model_name} --llamacpp_model_path gguf_models/{model_name}.gguf"
    elif engine_key == "tensorrt-llm":
        return f"--model_name Qwen/{model_name} --tensorrt_engine_dir trt_engines/{model_name}"
    elif engine_key == "onnx":
        return f"--model_name Qwen/{model_name} --onnx_model_dir models/onnx/{model_name}"
    else:
        return f"--model_name models/{model_name}"

def generate_script(hardware, engine_key, dataset_key, engine_config, dataset_path):
    job_name = f"{engine_key}_{dataset_key}"
    jobs_dir = JOBS_DIR_ROOT / hardware
    script_path = jobs_dir / f"{job_name}.sh"
    results_file = f"artifacts/{hardware}/compression_results.json"
    
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={hardware}_{job_name}",
        f"#SBATCH --output=jobs/{hardware}/logs/{job_name}_%j.out",
        f"#SBATCH --error=jobs/{hardware}/logs/{job_name}_%j.err",
        "#SBATCH --time=48:00:00",
        "#SBATCH --gres=gpu:1" if engine_key != "lamacpp-cpu" else "#SBATCH --cpus-per-task=16",
        "",
        "set -euo pipefail",
        "",
        f"ENV_ACTIVATE=\"{engine_config['env']}/bin/activate\"",
        "if [ -f \"$ENV_ACTIVATE\" ]; then",
        "    source \"$ENV_ACTIVATE\"",
        "else",
        "    echo \"Warning: Environment $ENV_ACTIVATE not found. Make sure to run setup_envs.sh\"",
        "fi",
        "",
        f"echo \"Starting dataset: {dataset_key} on {engine_key}...\"",
        ""
    ]
    
    for model in MODELS:
        model_args = build_model_args(engine_key, model)
        extra_args = engine_config.get("extra_args", "")
        
        cmd = (
            f"python main.py --mode compress "
            f"--input_path \"{dataset_path}\" "
            f"--engine {engine_config['engine_arg']} "
            f"{model_args} "
            f"--batch_size {BATCH_SIZE} "
            f"--context_length {CONTEXT_LENGTH} "
            f"--first_n_tokens {FIRST_N_TOKENS} "
            f"--results_file \"{results_file}\" "
        )
        if extra_args:
            cmd += f"{extra_args} "
            
        lines.append(f"echo \"Running {model}...\"")
        lines.append(cmd)
        lines.append("")
        
    lines.append("echo \"All models complete for this job.\"")
    
    with open(script_path, "w") as f:
        f.write("\n".join(lines))
    os.chmod(script_path, 0o755)

def main():
    parser = argparse.ArgumentParser(description="Generate slurm wrappers for a specific GPU class.")
    parser.add_argument("--hardware", nargs="+", default=["a100"], help="List of hardware names (e.g., a100 h100 l40s)")
    args = parser.parse_args()

    for hw in args.hardware:
        jobs_dir = JOBS_DIR_ROOT / hw
        jobs_dir.mkdir(parents=True, exist_ok=True)
        (jobs_dir / "logs").mkdir(parents=True, exist_ok=True)
        os.makedirs(f"artifacts/{hw}", exist_ok=True)
        
        for engine_key, engine_config in ENGINES.items():
            for dataset_key, dataset_path in DATASETS.items():
                generate_script(hw, engine_key, dataset_key, engine_config, dataset_path)
                
        print(f"Generated {len(ENGINES) * len(DATASETS)} job scripts for hardware '{hw}' in {jobs_dir.absolute()}")

if __name__ == "__main__":
    main()
