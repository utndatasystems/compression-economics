"""
CLI entry point for running global-mask compression and decompression experiments.
"""

import argparse
import json
import os


from src.global_mask_compressor import run_global_mask_compression, run_global_mask_decompression
from src.utils import save_global_mask_file, load_global_mask_file, load_results, save_results, make_key, create_run_dir, save_params
from src.prediction import TokenPredictor, create_predictor

RESULTS_FILE = "compression_results_grid_search.json"
COMPRESSION_FILE = "compression_data.bin"
DECOMPRESSION_FILE = "text_results.txt"

def build_arg_parser():
    """
    Build the command-line parser for compression experiments.
    """
    parser = argparse.ArgumentParser(description="Run Global Mask Compression Experiment")
    parser.add_argument("--mode", type=str, choices=["compress", "decompress"], required=True, help="Mode: compress or decompress")
    parser.add_argument("--input_path", type=str, default="data/text8",help="Input path: For compress mode, dataset path. For decompress mode, compression file path.")
    parser.add_argument("--output_path", type=str, help="Output path: For compress mode, compression file path. For decompress mode, reconstruction text file path.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B", help="Model name")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter (if any)")
    parser.add_argument("--context_length", type=int, default=1000, help="Maximum context length")
    parser.add_argument("--retain_tokens", type=int, default=100, help="Tokens retained when context length exceeded (only with KV cache)")
    parser.add_argument("--first_n_tokens", type=int, default=10001, help="Number of tokens to compress")
    parser.add_argument("--use_kv_cache", action="store_true", help="Enable KV cache for compression")
    parser.add_argument("--no_use_kv_cache", dest="use_kv_cache", action="store_false", help="Disable KV cache for compression")
    parser.set_defaults(use_kv_cache=True)
    parser.add_argument("--text_input", type=str, required=False, help="The direct text input for LLM inference.")
    parser.add_argument("--reduce_tokens", action="store_true", help="Restrict token space")
    parser.add_argument("--no_reduce_tokens", dest="reduce_tokens", action="store_false", help="Disable token space restriction")
    parser.set_defaults(reduce_tokens=True)
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for LLM inference")
    parser.add_argument("--engine", type=str, choices=["transformer", "vllm", "onnxruntime", "tensorrt", "sglang", "llamacpp", "llamacpp_direct", "mlx"], default="transformer", help="Inference engine to use")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="Tensor parallel size for vLLM")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.9, help="GPU memory utilization for vLLM")
    parser.add_argument("--tensorrt_engine_dir", type=str, default=None, help="Path to a prebuilt TensorRT-LLM engine directory")
    parser.add_argument("--sglang_mem_fraction_static", type=float, default=0.8, help="GPU memory fraction reserved by SGlang")
    parser.add_argument("--sglang_enable_deterministic_inference", action="store_true", help="Enable deterministic SGlang inference")
    parser.add_argument("--no_sglang_enable_deterministic_inference", dest="sglang_enable_deterministic_inference", action="store_false", help="Disable deterministic SGlang inference")
    parser.set_defaults(sglang_enable_deterministic_inference=True)
    parser.add_argument("--sglang_use_streaming_session_kv", action="store_true", help="Reuse SGlang streaming sessions for append-only KV-cache state")
    parser.add_argument("--no_sglang_use_streaming_session_kv", dest="sglang_use_streaming_session_kv", action="store_false", help="Disable SGlang streaming-session KV reuse")
    parser.set_defaults(sglang_use_streaming_session_kv=False)
    parser.add_argument("--llamacpp_model_path", type=str, default=None, help="Path to a local GGUF model for llama.cpp backends")
    parser.add_argument("--llamacpp_binary", type=str, default="llama-server", help="Path to the llama-server binary used by --engine llamacpp")
    parser.add_argument("--llamacpp_host", type=str, default="127.0.0.1", help="Host for the managed llama-server process used by --engine llamacpp")
    parser.add_argument("--llamacpp_port", type=int, default=8080, help="Port for the managed llama-server process used by --engine llamacpp")
    parser.add_argument("--llamacpp_threads", type=int, default=max(1, os.cpu_count() or 1), help="CPU threads for llama.cpp server evaluation or direct prompt evaluation")
    parser.add_argument("--llamacpp_direct_threads_batch", type=int, default=0, help="Batch-processing thread count for --engine llamacpp_direct (0 = auto)")
    parser.add_argument("--llamacpp_direct_n_batch", type=int, default=0, help="Prompt-processing batch size for --engine llamacpp_direct (0 = context length)")
    parser.add_argument("--llamacpp_direct_n_ubatch", type=int, default=0, help="Physical micro-batch size for --engine llamacpp_direct (0 = n_batch)")
    parser.add_argument("--llamacpp_direct_use_mmap", dest="llamacpp_direct_use_mmap", action="store_true", help="Enable mmap-backed model loading for --engine llamacpp_direct")
    parser.add_argument("--no_llamacpp_direct_use_mmap", dest="llamacpp_direct_use_mmap", action="store_false", help="Disable mmap-backed model loading for --engine llamacpp_direct")
    parser.add_argument("--llamacpp_direct_use_mlock", dest="llamacpp_direct_use_mlock", action="store_true", help="Lock GGUF weights in RAM for --engine llamacpp_direct")
    parser.add_argument("--no_llamacpp_direct_use_mlock", dest="llamacpp_direct_use_mlock", action="store_false", help="Do not lock GGUF weights in RAM for --engine llamacpp_direct")
    parser.set_defaults(llamacpp_direct_use_mmap=True, llamacpp_direct_use_mlock=False)
    parser.add_argument("--llamacpp_n_gpu_layers", type=int, default=0, help="Number of llama.cpp layers to offload to GPU")
    parser.add_argument("--mlx_model_source", type=str, default=None, help="Local MLX model directory or MLX-compatible Hugging Face repo")
    parser.add_argument("--mlx_tokenizer_source", type=str, default=None, help="Optional tokenizer source for the MLX backend")
    parser.add_argument("--onnx_model_dir", type=str, default=None, help="Path to a local exported ONNX model directory")
    parser.add_argument("--onnx_tokenizer_source", type=str, default=None, help="Optional tokenizer source for the ONNX Runtime backend")
    parser.add_argument("--onnx_execution_provider", type=str, choices=["CPUExecutionProvider"], default="CPUExecutionProvider", help="Execution provider for the ONNX Runtime backend")
    parser.add_argument("--onnx_intra_op_threads", type=int, default=max(1, os.cpu_count() or 1), help="ONNX Runtime intra-op CPU threads")
    parser.add_argument("--onnx_inter_op_threads", type=int, default=1, help="ONNX Runtime inter-op CPU threads")
    parser.add_argument("--onnx_graph_optimization_level", type=str, choices=["ORT_DISABLE_ALL", "ORT_ENABLE_BASIC", "ORT_ENABLE_EXTENDED", "ORT_ENABLE_ALL"], default="ORT_ENABLE_ALL", help="Graph optimization level for the ONNX Runtime backend")
    parser.add_argument("--encoding", type=str, choices=["AC", "bitpacked", "huffman"], default="AC", help="Encoding method for compression")
    parser.add_argument("--print_results", action="store_true", help="Print detailed results")
    parser.add_argument("--results_file", type=str, default=RESULTS_FILE, help="Path to the aggregated experiment-results JSON file")
    return parser


def main():
    """
    Parse CLI arguments and run compression or decompression.
    """
    parser = build_arg_parser()

    args = parser.parse_args()

    if args.mode == "compress":
            # ========================
            # Validate input paths
            if not args.input_path:
                parser.error("--input_path is required in compress mode")
            if not args.output_path:
                args.output_path = COMPRESSION_FILE

            if args.engine == "onnxruntime" and args.use_kv_cache:
                print("\nONNX Runtime uses full-prompt inference in this repo; disabling KV cache.")
                args.use_kv_cache = False

            # ========================
            # Check if experiment already exists
            # ========================
            results_db = load_results(args.results_file)
            exp_key = make_key(args)
            if exp_key in results_db:
                print(f"\n⚠️  Experiment already exists for {exp_key}, skipping run.")
                print(f"Stored Results: {results_db[exp_key]}")
                return

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
            # For vLLM, estimate params from HF config to avoid creating a
            # heavyweight LLM engine just for counting (the real engine is
            # created inside run_global_mask_compression).
            if args.engine in {"vllm", "tensorrt", "sglang"}:
                from transformers import AutoConfig
                from src.hf_cache import get_model_cache_dir as _cache_dir
                _cfg = AutoConfig.from_pretrained(args.model_name, cache_dir=_cache_dir())
                _h = getattr(_cfg, "hidden_size", 0)
                _n = getattr(_cfg, "num_hidden_layers", 0)
                _v = getattr(_cfg, "vocab_size", 0)
                base_params = _n * 12 * _h * _h + _v * _h
                base_size_mb = base_params * 2 / (1024 ** 2)
                adapter_params, adapter_size_mb = 0, 0.0
            elif args.engine == "onnxruntime":
                from src.onnxruntime_prediction import (
                    estimate_onnx_artifact_size_mb,
                    estimate_onnx_parameter_count,
                    probe_onnxruntime_backend_support,
                )

                supported, reason = probe_onnxruntime_backend_support(args)
                if not supported:
                    raise ValueError(f"ONNX Runtime backend is not available: {reason}")

                base_params = estimate_onnx_parameter_count(args)
                base_size_mb = estimate_onnx_artifact_size_mb(args.onnx_model_dir)
                adapter_params, adapter_size_mb = 0, 0.0
            elif args.engine == "mlx":
                from src.mlx_prediction import (
                    estimate_mlx_artifact_size_mb,
                    estimate_mlx_parameter_count,
                    get_mlx_model_source,
                    probe_mlx_backend_support,
                )

                supported, reason = probe_mlx_backend_support(args)
                if not supported:
                    raise ValueError(f"MLX backend is not available: {reason}")

                base_params = estimate_mlx_parameter_count(args)
                base_size_mb = estimate_mlx_artifact_size_mb(get_mlx_model_source(args))
                adapter_params, adapter_size_mb = 0, 0.0
            elif args.engine == "llamacpp":
                base_params = 0
                base_size_mb = 0.0
                adapter_params, adapter_size_mb = 0, 0.0
            elif args.engine == "llamacpp_direct":
                from src.llamacpp_direct_prediction import probe_llamacpp_direct_support

                supported, reason = probe_llamacpp_direct_support(args)
                if not supported:
                    raise ValueError(f"llama.cpp direct backend is not available: {reason}")

                base_params = 0
                base_size_mb = os.path.getsize(args.llamacpp_model_path) / (1024 ** 2)
                adapter_params, adapter_size_mb = 0, 0.0
            else:
                token_predictor = create_predictor(args, bitmap_data=None)
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

            # ========================
            # Save results (JSON stats)
            # ========================
            results_db = load_results(args.results_file)
            exp_key = make_key(args)
            if exp_key not in results_db:
                results_db[exp_key] = {}
            results_db[exp_key]["compression"] = comp_stats
            save_results(results_db, args.results_file) #add 

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
            print(f"Compression stats saved to: {args.results_file}")
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

        if args.engine == "onnxruntime" and args.use_kv_cache:
            print("\nONNX Runtime uses full-prompt inference in this repo; disabling KV cache.")
            args.use_kv_cache = False
        # ========================
        # Load binary compression file
        # ========================
        print(f"\nLoading compression file: {args.input_path}")
        args, first_token, bit_string, bitmask_data = load_global_mask_file(args)

        exp_key = make_key(args)

        print("\n===== Loaded Header =====")
        print(f"  Model            : {args.model_name}")
        print(f"  Context length   : {args.context_length}")
        print(f"  Retain tokens    : {args.retain_tokens}")
        print(f"  First n tokens   : {args.first_n_tokens}")
        print(f"  Use KV cache     : {args.use_kv_cache}")
        print(f"  Batch size       : {args.batch_size}")

        print("\n===== Decompress Data =====")
        _, results, decomp_stats = run_global_mask_decompression(
            args=args,
            first_tokens=first_token,
            bit_string=bit_string,
            bitmap=bitmask_data
        )

        # ========================
        # Save results (JSON stats)
        # ========================
        results_db = load_results(args.results_file)
        exp_key = make_key(args)
        if exp_key not in results_db:
            results_db[exp_key] = {}
        results_db[exp_key]["decompression"] = decomp_stats
        save_results(results_db, args.results_file)

        if args.print_results:
            print("\n\n===== Decompression Results =====")
            for k, v in decomp_stats.items():
                print(f"{k}: {v}")

        # Save the reconstructed text to a file
        with open(args.output_path, "w") as f:
            f.write(results)

        print("\n\n===== Decompression Complete =====")
        print(f"Decompression stats saved to: {args.results_file}")
        print(f"Reconstructed text saved to: {args.output_path}")
        

if __name__ == "__main__":
    main()
