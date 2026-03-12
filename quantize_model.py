import os
import json
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from src.utils import count_parameters, estimate_model_size_mb
from src.config import get_quantize_model_args


# Set device
if torch.cuda.is_available():
    torch.set_float32_matmul_precision("high")
    device = "cuda"
elif torch.backends.mps.is_available():
    torch.set_float32_matmul_precision("high")
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


def main():
    args = get_quantize_model_args()

    # Clean names
    model_name = os.path.basename(args.model_id.rstrip("/"))

    adapter_name = None
    if args.adapter_path:
        adapter_name = os.path.basename(args.adapter_path.rstrip("/"))

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
        dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quant_config,
    )

    # Load adapter if provided
    if args.adapter_path:
        config_path = os.path.join(args.adapter_path, "adapter_config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(
                f"No adapter_config.json found in {args.adapter_path}"
            )

        model = PeftModel.from_pretrained(model, args.adapter_path, device_map="auto")

    # Build hierarchical save path
    if adapter_name:
        save_path = os.path.join(
            args.save_dir,
            model_name,
            adapter_name,
            f"quantized_{args.quantization_bits}bit",
        )
    else:
        save_path = os.path.join(
            args.save_dir,
            model_name,
            f"quantized_{args.quantization_bits}bit",
        )

    os.makedirs(save_path, exist_ok=True)

    # Save quantized model
    model.save_pretrained(save_path)
    print(f"Quantized model saved at {save_path}")

    # Metadata
    total_params, adapter_params = count_parameters(model)
    total_size, adapter_size = estimate_model_size_mb(model)

    meta = {
        "model_id": args.model_id,
        "adapter_path": args.adapter_path,
        "adapter_name": adapter_name,
        "quantization_bits": args.quantization_bits,
        "total_params": total_params,
        "adapter_params": adapter_params,
        "total_size_mb": round(total_size, 2),
        "adapter_size_mb": round(adapter_size, 2),
    }

    meta_path = os.path.join(save_path, "meta.json")

    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print("Metadata saved to:", meta_path)


if __name__ == "__main__":
    main()
