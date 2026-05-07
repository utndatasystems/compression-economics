import argparse

from src.utils import load_model_list


def get_adapter_training_args() -> argparse.Namespace:
     
    model_list = load_model_list("./models.json")

    parser = argparse.ArgumentParser(description="Adapter Training Script")

    # path related
    parser.add_argument("--text_file", type=str, default="./data/text8", help="Path to text file for training")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save LoRA adapters")

    # model related
    parser.add_argument("--model_id", type=str, choices=model_list, default="Qwen/Qwen2.5-0.5B", help="Base model ID")
    parser.add_argument("--HF_token", type=str, default=None, help="Hugging Face token for rate limits (if needed)")
    
    # Adapter related
    parser.add_argument("--adapter_type", type=str, default="vera", choices=["lora", "vera", None], help="Type of adapter to train (e.g., lora, vera)")
    parser.add_argument("--r", type=int, default=8, help="Adapter rank")
    parser.add_argument("--la", type=int, default=32, help="LoRA alpha")

    # fine-tuning related
    parser.add_argument("--epoch", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--min_lr", type=float, default=1e-6, help="Minimum learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type")

    # Wandb related
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--wandb_project", type=str, default="adapter-finetuning", help="Weights & Biases project name")

    # quantization related
    parser.add_argument(
        "--mode",
        type=str,
        default="finetune",
        choices=["finetune", "quantize"],
        help="Whether to fine-tune with adapters or just quantize",
    )
    parser.add_argument(
        "--use_bnb",
        action="store_true",
        help="Use bitsandbytes quantization when loading the base model"
    )

    parser.add_argument(
        "--quantization_bits",
        type=int,
        default=None,
        choices=[4, 8],
        help="BitsAndBytes quantization bits for base model loading"
    )

    parser.add_argument(
        "--bnb_compute_dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Compute dtype used by BitsAndBytes"
    )

    parser.add_argument(
        "--bnb_4bit_quant_type",
        type=str,
        default="nf4",
        choices=["nf4", "fp4"],
        help="4-bit quantization type for BitsAndBytes"
    )

    parser.add_argument(
        "--bnb_4bit_use_double_quant",
        action="store_true",
        help="Use nested/double quantization for 4-bit BitsAndBytes"
    )

    parser.add_argument(
        "--adapter_save_dtype",
        type=str,
        default=None,
        choices=["fp32", "fp16", "bf16"],
        help="Cast LoRA/VeRA adapter weights to this dtype before saving"
    )

    args = parser.parse_args()

    if args.save_dir is None:
        if args.adapter_type is not None:
            args.save_dir = f"./adapters/{args.adapter_type}"
        elif args.mode == "quantize":
            args.save_dir = "./quantized_models"
            args.use_bnb = True
        else:
            raise ValueError("save_dir must be specified if adapter_type is None and mode is not quantize")

    return args


def get_main_args() -> argparse.Namespace:
    model_list = load_model_list("./models.json")

    parser = argparse.ArgumentParser(description="Run Global Mask Compression Experiment")

    # path related
    parser.add_argument("--mode", type=str, choices=["compress", "decompress"], required=True, help="Mode: compress or decompress")
    parser.add_argument("--input_path", type=str, default="data/text8", help="Input path: For compress mode, dataset path. For decompress mode, compression file path.")
    parser.add_argument("--output_path", type=str, help="Output path: For compress mode, compression file path. For decompress mode, reconstruction text file path.")
    parser.add_argument("--lora_path", type=str, default=None, help="Path to LoRA adapter (if any)")
    parser.add_argument("--text_input", type=str, required=False, help="The direct text input for LLM inference.")

    # model related
    parser.add_argument("--model_name", type=str, choices=model_list, default="Qwen/Qwen2.5-0.5B", help="Model name",)
    parser.add_argument("--HF_token", type=str, default=None, help="Hugging Face token for rate limits (if needed)")
    
    # inference related
    parser.add_argument("--context_length", type=int, default=1000, help="Maximum context length")
    parser.add_argument("--retain_tokens", type=int, default=100, help="Tokens retained when context length exceeded (only with KV cache)")
    parser.add_argument("--first_n_tokens", type=int, default=10001, help="Number of tokens to compress")
    parser.add_argument("--use_kv_cache", dest="use_kv_cache", action="store_true", help="Enable KV cache for incremental inference",)
    parser.add_argument("--no_kv_cache", dest="use_kv_cache", action="store_false", help="Disable KV cache for incremental inference",)
    parser.set_defaults(use_kv_cache=True)

    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for LLM inference")
    
    # compressiong related
    parser.add_argument("--reduce_tokens", action="store_true", help="Restrict token space")
    parser.add_argument("--no_reduce_tokens", dest="reduce_tokens", action="store_false", help="Disable token space restriction")
    parser.set_defaults(reduce_tokens=True)
    parser.add_argument("--engine", type=str, choices=["transformer"], default="transformer", help="Inference engine to use")
    parser.add_argument("--encoding", type=str, choices=["AC", "bitpacked", "huffman", "PMATIC"], default="AC", help="Encoding method for compression")
    parser.add_argument("--spec_k", type=int, default=None, help="Number of speculative tokens to generate for speculative compression/decompression")
    parser.add_argument("--draft_model_name", type=str, choices=model_list, default=None, help="Draft model name for speculative decompression (if different from teacher)")
    
    # other
    parser.add_argument("--print_results", action="store_true", help="Print detailed results")
    parser.add_argument("--force", action="store_true", help="Run the experiment even if results already exist.",)
    
    args = parser.parse_args()

    # Detect seq2seq models (T5)
    args.is_seq2seq = "t5" in args.model_name.lower()

    # Detect Mamba models
    args.is_mamba = "mamba" in args.model_name.lower()

    # Disable KV cache for T5
    if args.is_seq2seq or args.is_mamba:
        if args.use_kv_cache:
            print("⚠️ KV cache disabled for T5 and Mamba models.")
        args.use_kv_cache = False
    
    return args


def get_quantize_model_args() -> argparse.Namespace:
    model_list = load_model_list("./models.json")

    parser = argparse.ArgumentParser(description="Quantize a model with optional adapter")
    # path related
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to pre-trained adapter")
    parser.add_argument("--save_dir", type=str, default="./output", help="Directory to save quantized models")

    # model related
    parser.add_argument("--model_id", type=str, choices=model_list, required=True, help="Base model ID or path")

    # quantization related
    parser.add_argument("--quantization_bits", type=int, choices=[4, 8], required=True)
    
    args = parser.parse_args()
    return args
