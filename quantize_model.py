import os
import argparse
import json
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.utils import count_parameters, estimate_model_size_mb


# Set device
if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    device, use_fp16, use_bf16 = "cuda", True, False
elif torch.backends.mps.is_available():
    device, use_fp16, use_bf16 = "mps", False, False
    torch.set_float32_matmul_precision("high")
else:
    device, use_fp16, use_bf16 = "cpu", False, False

print(f"Using device: {device}")


def main():
    parser = argparse.ArgumentParser(description="Quantize a model with optional adapter")
    parser.add_argument("--model_id", type=str, required=True, help="Base model ID or path")
    parser.add_argument("--adapter_path", type=str, default=None, help="Path to pre-trained adapter to load")
    parser.add_argument("--save_dir", type=str, default="./output", help="Directory to save quantized model")
    parser.add_argument("--quantization_bits", type=int, choices=[4, 8], required=True, help="Quantization bit-width")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Extract clean model name
    model_name = os.path.basename(args.model_id.rstrip("/"))

    # Configure quantization
    quant_config = BitsAndBytesConfig(
        load_in_4bit=(args.quantization_bits == 4),
        load_in_8bit=(args.quantization_bits == 8),
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quant_config,
    )

    # Load adapter if provided
    if args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, device_map="auto")

    # Create hierarchical save path
    save_path = os.path.join(
        args.save_dir,
        model_name,
        f"quantized_{args.quantization_bits}bit"
    )

    os.makedirs(save_path, exist_ok=True)

    # Save quantized model
    model.save_pretrained(save_path)
    print(f"Quantized model saved at {save_path} ({args.quantization_bits}-bit)")

    # Save metadata
    total_params, adapter_params = count_parameters(model)
    total_size, adapter_size = estimate_model_size_mb(model)

    meta = {
        "model_id": args.model_id,
        "adapter_loaded": args.adapter_path is not None,
        "quantization_bits": args.quantization_bits,
        "total_params": total_params,
        "adapter_params": adapter_params,
        "total_size_mb": round(total_size, 2),
        "adapter_size_mb": round(adapter_size, 2),
    }

    meta_path = os.path.join(save_path, "meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Metadata saved at:", meta_path)


if __name__ == "__main__":
    main()