import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model

MODEL_ID = "Qwen/Qwen2.5-0.5B"
TEXT_FILE = "./data/text8"
OUT_DIR = "./lora"
r = 8


if torch.cuda.is_available():
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


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=".cache")
tokenizer.pad_token = tokenizer.eos_token

# 2. Load text dataset
dataset = load_dataset(
    "text",
    data_files={"train": TEXT_FILE},
)

# 3. Tokenize + chunk
def tokenize_fn(examples):
    outputs = tokenizer(
        examples["text"],
        truncation=True,
        max_length=1024,
        padding=False,
    )
    # outputs["labels"] = [
    #     ids.copy() for ids in outputs["input_ids"]
    # ]
    return outputs

dataset = dataset.map(
    tokenize_fn,
    batched=True,
    remove_columns=["text"],
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
)

# 4. Load base model
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map={"": device},
)

# 5. Attach LoRA
lora_config = LoraConfig(
    r=r,
    lora_alpha=r,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.0,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

training_args = TrainingArguments(
    output_dir=OUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=100,
    learning_rate=1e-3,
    fp16=use_fp16,
    bf16=use_bf16,
    logging_strategy="steps",
    logging_steps=0.1,
    save_strategy="steps",
    save_steps=0.3,
    weight_decay=0.0,
    report_to="none",
    dataloader_pin_memory=(device == "cuda"),
)

# 7. Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator,
)

trainer.train()

# 8. Save LoRA adapter
lora_path = f"{OUT_DIR}/{os.path.basename(TEXT_FILE)}/lora_r{r}"
model.save_pretrained(lora_path)