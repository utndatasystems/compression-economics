import os
import torch
import json

from .utils import get_device, count_parameters, estimate_model_size_mb

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer)
import wandb
from peft import LoraConfig, VeraConfig, get_peft_model


class AdapterTrainer:

    def __init__(self, args):
        self.args = args
        self.quant_config = None

        self.device_init()
        self.run_name_init()
        self.path_init()
        self.tokenizer_init()
        self.data_init()

        if self.args.mode == "quantize" and self.args.quantization_bits is not None:
            print(f"Quantization mode selected with {self.args.quantization_bits}-bit quantization. No training will be performed.")
        if self.args.mode == "quantize":
            self.quantization_init()
        self.model_init()
        if self.args.mode == "finetune":
            self.adapter_init()

        self.get_parameters_stats()

    def device_init(self):
        self.device, self.use_fp16, self.use_bf16 = get_device()

    def run_name_init(self):
        if self.args.adapter_type == "lora":
            self.run_name = f"r{self.args.r}_la{self.args.la}_lr{self.args.lr}_ls{self.args.lr_scheduler_type}_bs{self.args.batch_size}_ep{self.args.epoch}_gas{self.args.gradient_accumulation_steps}"
        elif self.args.adapter_type == "vera":
            self.run_name = f"r{self.args.r}_lr{self.args.lr}_ls{self.args.lr_scheduler_type}_bs{self.args.batch_size}_ep{self.args.epoch}_gas{self.args.gradient_accumulation_steps}"
        elif self.args.adapter_type is None and self.args.mode == "quantize":
            self.run_name = f"quant_{self.args.quantization_bits}bit"
        else:
            raise ValueError(f"Unknown adapter type: {self.args.adapter_type}")

        # Initialize Weights & Biases for finetuning mode only 
        if self.args.mode == "finetune":
            wandb.init(
                project=self.args.wandb_project,  # e.g., "adapter-finetuning"
                name=self.run_name,
                config=vars(self.args)
            )
        
    def path_init(self):
        if self.args.mode == "quantize":
            self.output_path = f"{self.args.save_dir}/quantized_{self.args.quantization_bits}bit"
        elif self.args.mode == "finetune":
            self.output_path = f"{self.args.save_dir}/{os.path.basename(self.args.text_file)}/{self.run_name}"
        else:
            raise ValueError(f"Unknown mode: {self.args.mode}")
        if os.path.exists(self.output_path):
            print(f"Skipping training for existing path: {self.output_path}")
        else:
            os.makedirs(self.output_path, exist_ok=True)

    def tokenizer_init(self):
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model_id, cache_dir=".cache")
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def data_init(self):
        with open(self.args.text_file, "r") as f:
            full_text = f.read()
        dataset = Dataset.from_dict({"text": [full_text]})

        def tokenize_fn(examples):
            tokens = self.tokenizer(examples["text"][0])
            input_ids = tokens["input_ids"]

            block_size = 256
            chunks = []

            for i in range(0, len(input_ids) - block_size, block_size):
                chunks.append(input_ids[i:i+block_size])

            return {"input_ids": chunks,}
        
        self.dataset = dataset.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
        )

        self.data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )

    def quantization_init(self):
        if self.args.quantization_bits is None:
            raise ValueError("Specify --quantization_bits 4 or 8 for quantization mode")

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=(self.args.quantization_bits == 4),
            load_in_8bit=(self.args.quantization_bits == 8),
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        # No adapter rank for quantization-only mode
        self.args.r = None
        self.args.la = None
    
    def adapter_init(self):
        if self.args.adapter_type not in ["lora", "vera"]:
            raise ValueError(f"Unknown adapter type: {self.args.adapter_type}")

        if self.args.adapter_type == "vera":
            adapter_config = VeraConfig(
                r=self.args.r,
                target_modules=["q_proj", "v_proj"],
                bias="none",
                vera_dropout=0.0,
            )

        elif self.args.adapter_type == "lora":
            adapter_config = LoraConfig(
                r=self.args.r,
                lora_alpha=self.args.la,
                target_modules=["q_proj", "v_proj"],
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
            )

        self.model = get_peft_model(self.model, adapter_config)

    def model_init(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_id,
            device_map="auto",
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            quantization_config=self.quant_config,
        )

    def get_parameters_stats(self):
        self.total_params, self.adapter_params = count_parameters(self.model)
        self.total_size_mb, self.adapter_size_mb = estimate_model_size_mb(self.model)

        self.base_model_params = self.total_params - self.adapter_params
        self.base_model_size_mb = self.total_size_mb - self.adapter_size_mb

        if self.args.mode == "finetune":
            print('Model parameters:')
            print(f"    Total parameters: {self.total_params:,}")
            print(f"    Adapter parameters: {self.adapter_params:,}")
            print(f"    Base model parameters: {self.base_model_params:,}")
            print(f"    Total size (MB): {self.total_size_mb:.2f}")
            print(f"    Adapter size (MB): {self.adapter_size_mb:.2f}")
            print(f"    Base model size (MB): {self.base_model_size_mb:.2f}")

        elif self.args.mode == "quantize":
            print('Quantized model parameters (no training):')
            print(f"    Total parameters: {self.total_params:,}")
            print(f"    Total size (MB): {self.total_size_mb:.2f}")

    def get_training_args(self):

        return TrainingArguments(
            output_dir=self.args.save_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.epoch,
            learning_rate=self.args.lr,
            lr_scheduler_type=self.args.lr_scheduler_type,
            warmup_steps=self.args.warmup_steps,
            fp16=self.use_fp16,
            bf16=self.use_bf16,
            logging_strategy="steps",
            logging_steps=10,  # log every 10 steps # 0.1
            save_strategy="no",
            weight_decay=0.0,
            report_to="wandb",  # enable wandb reporting, before 'none'
            dataloader_pin_memory=(self.device == "cuda"),
            remove_unused_columns=False,
        )
    
    def finetune(self):
        print("Training arguments:")
        print(f"    Adapter\t\t\t: {self.args.adapter_type}")
        print(f"    Epochs\t\t\t: {self.args.epoch}")
        print(f"    Learning rate\t\t: {self.args.lr}")
        print(f"    LR scheduler\t\t: {self.args.lr_scheduler_type}")
        print(f"    Batch size\t\t\t: {self.args.batch_size}")
        print(f"    Gradient accumulation\t: {self.args.gradient_accumulation_steps}")

        model.print_trainable_parameters()
        model = torch.compile(model)
        training_args = self.get_training_args()

        self.trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=self.data_collator,
        )
        self.trainer.train()

        self.save_model()

    def quantize(self):
        self.save_model()

    def save_model(self):
        self.model.save_pretrained(self.output_path)

        meta = {
            "model_id": self.args.model_id,
            "adapter": self.args.adapter_type,
            "rank": self.args.r,
            "learning_rate": self.args.lr,
            "batch_size": self.args.batch_size,
            "gradient_accumulation_steps": self.args.gradient_accumulation_steps,
            "lr_scheduler_type": self.args.lr_scheduler_type,
            "epochs": self.args.epoch,
            "final_loss": self.trainer.state.log_history[-1]["train_loss"] if self.args.mode == "finetune" else None,
            "train_runtime": self.trainer.state.log_history[-1]["train_runtime"] if self.args.mode == "finetune" else None,
            "base_model_params": self.base_model_params,
            "adapter_params": self.adapter_params,
            "total_model_params": self.total_params,
            "base_model_size_mb": round(self.base_model_size_mb, 2),
            "adapter_size_mb": round(self.adapter_size_mb, 2), 
            "total_model_size_mb": round(self.total_size_mb, 2),
            "quantization_bits": self.args.quantization_bits if self.args.mode == "quantize" else None,
        }
        with open(os.path.join(self.output_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        print("Metadata saved to:", os.path.join(self.output_path, "meta.json"))

        if self.args.mode == "finetune":
            wandb.log({
                "final_loss": self.trainer.state.log_history[-1]["train_loss"],
                "train_runtime": self.trainer.state.log_history[-1]["train_runtime"],
                "total_model_params": self.total_params,
                "adapter_params": self.adapter_params,
                })
            wandb.finish()