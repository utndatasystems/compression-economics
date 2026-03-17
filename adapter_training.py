import os
import argparse
import json
import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig)

from peft import LoraConfig, VeraConfig, get_peft_model

from src.hf_cache import get_model_cache_dir


def count_parameters(model):
    total_params = 0
    trainable_params = 0

    for p in model.parameters():
        numel = p.numel()
        total_params += numel
        if p.requires_grad:
            trainable_params += numel
    return total_params, trainable_params

def estimate_model_size_mb(model):
    total_bytes = 0
    trainable_bytes = 0

    for p in model.parameters():
        bytes_ = p.numel() * p.element_size()
        total_bytes += bytes_
        if p.requires_grad:
            trainable_bytes += bytes_
    return total_bytes / (1024 ** 2), trainable_bytes / (1024 ** 2)


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
    parser = argparse.ArgumentParser(description="LoRA Training Script")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model ID")
    parser.add_argument("--text_file", type=str, default="./data/text8", help="Path to text file for training")
    parser.add_argument("--adapter_type", type=str, default=None, choices=["lora", "vera", None], help="Type of adapter to train (e.g., lora, vera)")
    parser.add_argument("--save_dir", type=str, default=None, help="Directory to save LoRA adapters")
    parser.add_argument("--r", type=int, default=8, help="Adapter rank")
    parser.add_argument("--la", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epoch", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    parser.add_argument("--mode", type=str, default="finetune",
                    choices=["finetune", "quantize"],
                    help="Whether to fine-tune with adapters or just quantize")
    parser.add_argument("--quantization_bits", type=int, default=None,
                    choices=[4, 8], 
                    help="Quantize model to 4-bit or 8-bit")
    args = parser.parse_args()

    if args.save_dir is None:
        if args.adapter_type is not None:
            args.save_dir = f"./adapters/{args.adapter_type}"
        elif args.mode == "quantize":
            args.save_dir = "./quantized_models"
        else:
            raise ValueError("save_dir must be specified if adapter_type is None and mode is not quantize")

    if args.adapter_type == "lora":
        run_name = f"r{args.r}_la{args.la}_lr{args.lr}_ls{args.lr_scheduler_type}_bs{args.batch_size}_ep{args.epoch}_gas{args.gradient_accumulation_steps}"
    elif args.adapter_type == "vera":
        run_name = f"r{args.r}_lr{args.lr}_ls{args.lr_scheduler_type}_bs{args.batch_size}_ep{args.epoch}_gas{args.gradient_accumulation_steps}"
    elif args.adapter_type is None and args.mode == "quantize":
        run_name = f"quant_{args.quantization_bits}bit"
    else:
        raise ValueError(f"Unknown adapter type: {args.adapter_type}")

    lora_path = f"{args.save_dir}/{os.path.basename(args.text_file)}/{run_name}"

    # if lora_path exist skip
    if os.path.exists(lora_path):
        print(f"Skipping training for existing path: {lora_path}")
        return

    if args.mode == "quantize" and args.quantization_bits is not None:
        print(f"Quantization mode selected with {args.quantization_bits}-bit quantization. No training will be performed.")
    
    if args.mode == "finetune":
        if args.adapter_type not in ["lora", "vera"]:
            raise ValueError(f"Unknown adapter type: {args.adapter_type}")

        print("Training arguments:")
        print(f"    Adapter\t\t\t: {args.adapter_type}")
        print(f"    Epochs\t\t\t: {args.epoch}")
        print(f"    Learning rate\t\t: {args.lr}")
        print(f"    LR scheduler\t\t: {args.lr_scheduler_type}")
        print(f"    Batch size\t\t\t: {args.batch_size}")
        print(f"    Gradient accumulation\t: {args.gradient_accumulation_steps}")
    
    # Load tokenizer
    cache_dir = get_model_cache_dir()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    # Load text data
    with open(args.text_file, "r") as f:
        full_text = f.read()
    dataset = Dataset.from_dict({"text": [full_text]})

    # Tokenize dataset
    def tokenize_fn(examples):
        tokens = tokenizer(examples["text"][0])
        input_ids = tokens["input_ids"]

        block_size = 256
        chunks = []

        for i in range(0, len(input_ids) - block_size, block_size):
            chunks.append(input_ids[i:i+block_size])

        return {"input_ids": chunks,}
    
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],)

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,)

    # Load model and attach LoRA
    #model = AutoModelForCausalLM.from_pretrained(
    #    args.model_id,
    #    dtype="auto",
    #    device_map={"": device},)
    

    quant_config = None

    if args.mode == "quantize":
        if args.quantization_bits is None:
            raise ValueError("Specify --quantization_bits 4 or 8 for quantization mode")

        quant_config = BitsAndBytesConfig(
            load_in_4bit=(args.quantization_bits == 4),
            load_in_8bit=(args.quantization_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        cache_dir=cache_dir,
        device_map="auto",
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        quantization_config=quant_config,
        )
    
    if args.mode == "finetune":
        if args.adapter_type == "vera":
            adapter_config = VeraConfig(
                r=args.r,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                vera_dropout=0.0,
            )

        elif args.adapter_type == "lora":
            adapter_config = LoraConfig(
                r=args.r,
                lora_alpha=args.la,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )

        model = get_peft_model(model, adapter_config)
    
    total_params, adapter_params = count_parameters(model)
    total_size_mb, adapter_size_mb = estimate_model_size_mb(model)

    base_model_params = total_params - adapter_params
    base_model_size_mb = total_size_mb - adapter_size_mb

    if args.mode == "finetune":
        print('Model parameters:')
        print(f"    Total parameters: {total_params:,}")
        print(f"    Adapter parameters: {adapter_params:,}")
        print(f"    Base model parameters: {base_model_params:,}")
        print(f"    Total size (MB): {total_size_mb:.2f}")
        print(f"    Adapter size (MB): {adapter_size_mb:.2f}")
        print(f"    Base model size (MB): {base_model_size_mb:.2f}")

    elif args.mode == "quantize":
        print('Quantized model parameters (no training):')
        print(f"    Total parameters: {total_params:,}")
        print(f"    Total size (MB): {total_size_mb:.2f}")
    
    training_args = TrainingArguments(
        output_dir=args.save_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_train_epochs=args.epoch,
        learning_rate=args.lr,
        lr_scheduler_type=args.lr_scheduler_type,
        warmup_steps=args.warmup_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        logging_strategy="steps",
        logging_steps=0.1,
        save_strategy="no",
        # save_strategy="steps",
        # save_steps=0.3,
        weight_decay=0.0,
        report_to="none",
        dataloader_pin_memory=(device == "cuda"),
        remove_unused_columns=False,)

    # 7. Train
    if args.mode == "finetune":
        model.print_trainable_parameters()
        model = torch.compile(model)

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
        )
        trainer.train()
    else:
        print("Quantization complete. No training performed.")

    # 8. Save LoRA adapter
    #model.save_pretrained(lora_path)

    if args.mode == "finetune":
        model.save_pretrained(lora_path)

    elif args.mode == "quantize":
        quant_path = f"{args.save_dir}/quantized_{args.quantization_bits}bit"
        os.makedirs(quant_path, exist_ok=True)
        model.save_pretrained(quant_path)
        args.r = None  # No adapter rank for quantization-only mode
        args.la = None  # No LoRA alpha for quantization-only mode


    meta = {
        "model_id": args.model_id,
        "adapter": args.adapter_type,
        "rank": args.r,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "epochs": args.epoch,
        "final_loss": trainer.state.log_history[-1]["train_loss"] if args.mode == "finetune" else None,
        "train_runtime": trainer.state.log_history[-1]["train_runtime"] if args.mode == "finetune" else None,
        "base_model_params": base_model_params,
        "adapter_params": adapter_params,
        "total_model_params": total_params,
        "base_model_size_mb": round(base_model_size_mb, 2),
        "adapter_size_mb": round(adapter_size_mb, 2), 
        "total_model_size_mb": round(total_size_mb, 2),
        "quantization_bits": args.quantization_bits if args.mode == "quantize" else None,
    }

    #with open(f"{lora_path}/meta.json", "w") as f:
    #    json.dump(meta, f, indent=2)

    # Decide where to save metadata
    if args.mode == "finetune":
        meta_path = lora_path
    elif args.mode == "quantize":
        meta_path = quant_path

    os.makedirs(meta_path, exist_ok=True)

    with open(os.path.join(meta_path, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print("Metadata saved to:", os.path.join(meta_path, "meta.json"))

if __name__ == "__main__":
    main()