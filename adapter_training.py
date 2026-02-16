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
)
from peft import LoraConfig, VeraConfig, get_peft_model

if torch.cuda.is_available():
    torch.set_float32_matmul_precision('high')
    device = "cuda"
    use_fp16 = True
    use_bf16 = False
elif torch.backends.mps.is_available():
    device = "mps"
    use_fp16 = False
    use_bf16 = False
    torch.set_float32_matmul_precision("high")
else:
    device = "cpu"
    use_fp16 = False
    use_bf16 = False

print(f"Using device: {device}")


def main():
    parser = argparse.ArgumentParser(description="LoRA Training Script")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen2.5-0.5B", help="Base model ID")
    parser.add_argument("--text_file", type=str, default="./data/text8", help="Path to text file for training")
    parser.add_argument("--adapter_type", type=str, default="lora", help="Type of adapter to train (e.g., lora, vera)")
    parser.add_argument("--out_dir", type=str, default=None, help="Directory to save LoRA adapters")
    parser.add_argument("--r", type=int, default=8, help="LoRA rank")
    parser.add_argument("--epoch", type=int, default=2, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2, help="Gradient accumulation steps")
    parser.add_argument("--lr_scheduler_type", type=str, default="constant", help="Learning rate scheduler type")
    parser.add_argument("--warmup_steps", type=int, default=0, help="Number of warmup steps for learning rate scheduler")
    args = parser.parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join("./adapters/", args.adapter_type)

    print("Training arguments:")
    print(f"    Adapter\t\t\t: {args.adapter_type}")
    print(f"    Epochs\t\t\t: {args.epoch}")
    print(f"    Learning rate\t\t: {args.lr}")
    print(f"    LR scheduler\t\t: {args.lr_scheduler_type}")
    print(f"    Batch size\t\t\t: {args.batch_size}")
    print(f"    Gradient accumulation\t: {args.gradient_accumulation_steps}")
    

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, cache_dir=".cache")
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

        return {
            "input_ids": chunks,
        }
    dataset = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    # Load model and attach LoRA
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype="auto",
        device_map={"": device},
    )
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
            lora_alpha=args.r,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

    model = get_peft_model(model, adapter_config)
    model.print_trainable_parameters()
    model = torch.compile(model)

    training_args = TrainingArguments(
        output_dir=args.out_dir,
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
        remove_unused_columns=False,
    )

    # 7. Train
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    trainer.train()

    # 8. Save LoRA adapter
    run_name = f"r{args.r}_lr{args.lr}_ls{args.lr_scheduler_type}_bs{args.batch_size}_ep{args.epoch}"
    lora_path = f"{args.out_dir}/{os.path.basename(args.text_file)}/{run_name}"
    model.save_pretrained(lora_path)


    meta = {
        "model_id": args.model_id,
        "adapter": args.adapter_type,
        "rank": args.r,
        "learning_rate": args.lr,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "lr_scheduler_type": args.lr_scheduler_type,
        "epochs": args.epoch,
        "final_loss": trainer.state.log_history[-1]["train_loss"],
        "train_runtime": trainer.state.log_history[-1]["train_runtime"],
    }

    with open(f"{lora_path}/meta.json", "w") as f:
        json.dump(meta, f, indent=2)




if __name__ == "__main__":
    main()