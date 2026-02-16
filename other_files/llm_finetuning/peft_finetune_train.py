import argparse
import os
import torch
import time
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model
from src.global_mask_compressor import run_global_mask_compression
from src.prediction import TokenDataPreparer, TokenPredictor
from src.utils import load_results, make_key

TRAIN_FILE = "train_results.json"


class PrintLossCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        logs = kwargs.get("logs", {})
        loss = logs.get("loss")
        if loss is not None:
            print(f"Step {state.global_step}, Loss: {loss:.4f}")

def run_compression(args):
    """
    Run the global mask compression experiment.
    """
    results_db = load_results(TRAIN_FILE)
    exp_key = make_key(args)
    if exp_key in results_db:
        print(f"\n⚠️ Training run already exists for {exp_key}, skipping run.")
        print(f"Stored Results: {results_db[exp_key]}")
        return

    print(f"\nRunning training with parameters:")
    print(f"  Data path        : {args.input_path}")
    print(f"  Model            : {args.model_name}")
    print(f"  Context length   : {args.context_length}")
    print(f"  Retain tokens    : {args.retain_tokens}")
    print(f"  First n tokens   : {args.first_n_tokens}")
    print(f"  Use KV cache     : {args.use_kv_cache}")
    print(f"  Batch size       : {args.batch_size}")
    print(f"  Encoding         : {args.encoding}")

    first_token, bit_string, bitmask_data, comp_stats, args = run_global_mask_compression(args)
    return bitmask_data


def setup_peft_model(model_name, dtype=torch.float16):
    """
    Load base model and tokenizer, then apply LoRA.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=".cache", use_fast=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)

    peft_config = LoraConfig(
        r= 2, #16,
        lora_alpha= 4, #32,
        target_modules=["q_proj", "v_proj"],  # adjust based on architecture
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    return model, tokenizer


def tokenize_data(args, tokenizer):
    """
    Tokenizes text using the TokenDataPreparer and returns a HuggingFace-compatible dataset.
    """
    t0_tokenize = time.perf_counter()

    token_data_preparer = TokenDataPreparer(args)
    data_tokens = token_data_preparer.get_data_tokens()
    args = token_data_preparer.get_args()

    if args.first_n_tokens is None:
        args.first_n_tokens = len(data_tokens)

    # Split tokens into contiguous batches
    chunk_length = len(data_tokens) // args.batch_size
    extra = len(data_tokens) % args.batch_size

    batches = []
    start = 0
    for i in range(args.batch_size):
        size = chunk_length + (1 if i < extra else 0)
        end = start + size
        batches.append(data_tokens[start:end])
        start = end

    # Capture first token from each batch
    first_tokens = [batch[0] for batch in batches if batch]

    # Get compressed bitmap
    bitmask_data = token_data_preparer.get_bitmap()
    total_bitmap_size = len(bitmask_data) * 8
    tokenize_time = time.perf_counter() - t0_tokenize

    # Convert batches into HuggingFace-style dataset
    from datasets import Dataset

    hf_data = {"input_ids": batches}
    hf_dataset = Dataset.from_dict(hf_data)
    hf_dataset.set_format(type="torch", columns=["input_ids"])

    return hf_dataset, bitmask_data, first_tokens, tokenize_time, total_bitmap_size


def main():
    parser = argparse.ArgumentParser(description="Fine-tune LLM with PEFT/LoRA")
    parser.add_argument("--mode", type=str, choices=["train"], default="train", help="Mode: train model")
    # introduce argument for full train mode or with PEFT
    parser.add_argument("--input_path", type=str, default="data/text8")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--context_length", type=int, default=100) #1000
    parser.add_argument("--retain_tokens", type=int, default=100)
    parser.add_argument("--first_n_tokens", type=int, default=100) #1000
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--text_input", type=str, default=None)
    parser.add_argument("--reduce_tokens", action="store_true")
    parser.add_argument("--no_reduce_tokens", dest="reduce_tokens", action="store_false")
    parser.set_defaults(reduce_tokens=True)
    parser.add_argument("--engine", type=str, choices=["transformer"], default="transformer")
    parser.add_argument("--encoding", type=str, choices=["AC", "bitpacked", "huffman"], default="AC")
    parser.add_argument("--print_results", action="store_true")
    args = parser.parse_args()

    # Load PEFT model
    model, tokenizer = setup_peft_model(args.model_name)

    # Tokenize dataset
    train_dataset, bitmask_data, first_tokens, tokenize_time, total_bitmap_size = tokenize_data(args, tokenizer)
    print(f"Tokenization complete in {tokenize_time:.2f}s, total bitmap size: {total_bitmap_size} bits")

    # Data collator for causal LM
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        learning_rate=args.lr,
        num_train_epochs= 2, #args.epochs,
        logging_steps = 1, # 10
        fp16=True,
        save_strategy= "no", #"epoch",
        report_to="none",
        weight_decay=0.01,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
        callbacks=[PrintLossCallback],
    )

    trainer.train()


if __name__ == "__main__":
    main()
