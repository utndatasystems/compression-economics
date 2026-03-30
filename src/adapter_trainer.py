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
from peft import (
    LoraConfig,
    VeraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)


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

        if self.args.use_bnb:
            self.quantization_init()

        self.model_init()
        if self.args.mode == "finetune":
            self.prepare_model_for_kbit_training_if_needed()
            self.adapter_init()

        self.get_parameters_stats()

        self.print_dtype_summary(stage="after_init")

    def device_init(self):
        self.device, self.use_fp16, self.use_bf16 = get_device()

    def run_name_init(self):
        quant_part = ""
        if self.args.use_bnb and self.args.quantization_bits is not None:
            quant_part = f"_bnb{self.args.quantization_bits}bit_{self.args.bnb_compute_dtype}"

        if self.args.adapter_type == "lora":
            self.run_name = (
                f"r{self.args.r}_la{self.args.la}_lr{self.args.lr}"
                f"_ls{self.args.lr_scheduler_type}_bs{self.args.batch_size}"
                f"_ep{self.args.epoch}_gas{self.args.gradient_accumulation_steps}"
                f"{quant_part}"
            )
        elif self.args.adapter_type == "vera":
            self.run_name = (
                f"r{self.args.r}_lr{self.args.lr}"
                f"_ls{self.args.lr_scheduler_type}_bs{self.args.batch_size}"
                f"_ep{self.args.epoch}_gas{self.args.gradient_accumulation_steps}"
                f"{quant_part}"
            )
        elif self.args.adapter_type is None and self.args.mode == "quantize":
            self.run_name = f"quant_{self.args.quantization_bits}bit_{self.args.bnb_compute_dtype}"
        else:
            raise ValueError(f"Unknown adapter type: {self.args.adapter_type}")

        # Initialize Weights & Biases for finetuning mode only 
        if self.args.mode == "finetune" and self.args.use_wandb:
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
            raise ValueError("When --use_bnb is enabled, specify --quantization_bits as 4 or 8")

        if self.args.quantization_bits not in [4, 8]:
            raise ValueError("quantization_bits must be 4 or 8")

        bnb_compute_dtype = self.resolve_torch_dtype(self.args.bnb_compute_dtype)

        self.quant_config = BitsAndBytesConfig(
            load_in_4bit=(self.args.quantization_bits == 4),
            load_in_8bit=(self.args.quantization_bits == 8),
            bnb_4bit_compute_dtype=bnb_compute_dtype,
            bnb_4bit_use_double_quant=self.args.bnb_4bit_use_double_quant,
            bnb_4bit_quant_type=self.args.bnb_4bit_quant_type,
        )
    
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
        model_kwargs = {}

        if self.args.use_bnb:
            model_dtype = self.resolve_torch_dtype(self.args.bnb_compute_dtype)
            model_kwargs["quantization_config"] = self.quant_config
            model_kwargs["torch_dtype"] = model_dtype
        else:
            model_kwargs["torch_dtype"] = (
                torch.float16 if self.device == "cuda" else torch.float32
            )

        self.model = AutoModelForCausalLM.from_pretrained(
            self.args.model_id,
            device_map="auto",
            cache_dir=".cache",
            **model_kwargs,
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
        if self.args.lr_scheduler_type == "cosine_warmup_with_min_lr":
            lr_scheduler_kwargs = {
                "min_lr": self.args.min_lr
            }
        else:
            lr_scheduler_kwargs = None

        optim_name = "paged_adamw_32bit" if self.args.use_bnb else "adamw_torch"

        if self.args.use_bnb:
            fp16 = self.use_fp16 and self.args.bnb_compute_dtype == "fp16"
            bf16 = self.use_bf16 and self.args.bnb_compute_dtype == "bf16"
        else:
            fp16 = self.use_fp16
            bf16 = self.use_bf16

        return TrainingArguments(
            output_dir=self.args.save_dir,
            per_device_train_batch_size=self.args.batch_size,
            gradient_accumulation_steps=self.args.gradient_accumulation_steps,
            num_train_epochs=self.args.epoch,
            learning_rate=self.args.lr,
            lr_scheduler_type=self.args.lr_scheduler_type,
            lr_scheduler_kwargs=lr_scheduler_kwargs,
            warmup_steps=self.args.warmup_steps,
            fp16=fp16,
            bf16=bf16,
            logging_strategy="steps",
            logging_steps=10,  # log every 10 steps # 0.1
            save_strategy="no",
            weight_decay=0.0,
            report_to="wandb" if self.args.use_wandb else "none",
            dataloader_pin_memory=(self.device == "cuda"),
            remove_unused_columns=False,
            optim=optim_name,
        )
    
    def finetune(self):
        print("Training arguments:")
        print(f"    Adapter\t\t\t: {self.args.adapter_type}")
        print(f"    Epochs\t\t\t: {self.args.epoch}")
        print(f"    Learning rate\t\t: {self.args.lr}")
        print(f"    LR scheduler\t\t: {self.args.lr_scheduler_type}")
        print(f"    Batch size\t\t\t: {self.args.batch_size}")
        print(f"    Gradient accumulation\t: {self.args.gradient_accumulation_steps}")
        if self.args.use_bnb:
            print(f"    Quantization bits\t\t: {self.args.quantization_bits}")
            print(f"    BnB compute dtype\t\t: {self.args.bnb_compute_dtype}")

        
        self.print_dtype_summary(stage="before_train")

        self.model.print_trainable_parameters()
        # self.model = torch.compile(self.model)
        training_args = self.get_training_args()

        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            data_collator=self.data_collator,
        )
        self.trainer.train()

        self.print_dtype_summary(stage="after_train_before_save")

        self.save_model()

    def quantize(self):
        self.save_model()

    def save_model(self):
        final_loss = self._get_last_log_value("train_loss") if self.args.mode == "finetune" else None
        train_runtime = self._get_last_log_value("train_runtime") if self.args.mode == "finetune" else None

        if self.args.mode == "finetune" and self.args.adapter_save_dtype is not None:
            self.cast_adapter_weights_for_saving(self.args.adapter_save_dtype)
            self.print_dtype_summary(stage="after_adapter_cast_before_save")

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
            "final_loss": final_loss,
            "train_runtime": train_runtime,
            "base_model_params": self.base_model_params,
            "adapter_params": self.adapter_params,
            "total_model_params": self.total_params,
            "base_model_size_mb": round(self.base_model_size_mb, 2),
            "adapter_size_mb": round(self.adapter_size_mb, 2), 
            "total_model_size_mb": round(self.total_size_mb, 2),
            "use_bnb": self.args.use_bnb,
            "quantization_bits": self.args.quantization_bits if self.args.use_bnb else None,
            "bnb_compute_dtype": self.args.bnb_compute_dtype if self.args.use_bnb else None,
            "bnb_4bit_quant_type": self.args.bnb_4bit_quant_type if self.args.use_bnb and self.args.quantization_bits == 4 else None,
            "bnb_4bit_use_double_quant": self.args.bnb_4bit_use_double_quant if self.args.use_bnb and self.args.quantization_bits == 4 else None,
        }
        with open(os.path.join(self.output_path, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        
        print("Metadata saved to:", os.path.join(self.output_path, "meta.json"))

        if self.args.mode == "finetune" and self.args.use_wandb:
            wandb.log({
                "final_loss": self.trainer.state.log_history[-1]["train_loss"],
                "train_runtime": self.trainer.state.log_history[-1]["train_runtime"],
                "total_model_params": self.total_params,
                "adapter_params": self.adapter_params,
                })
            wandb.finish()

    def resolve_torch_dtype(self, dtype_str):
        mapping = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }
        if dtype_str not in mapping:
            raise ValueError(f"Unsupported dtype: {dtype_str}")
        return mapping[dtype_str]
    
    def prepare_model_for_kbit_training_if_needed(self):
        if self.args.use_bnb and self.args.quantization_bits in [4, 8]:
            self.model = prepare_model_for_kbit_training(self.model)

    def _get_last_log_value(self, key):
        if not hasattr(self, "trainer"):
            return None
        for item in reversed(self.trainer.state.log_history):
            if key in item:
                return item[key]
        return None
    
    def cast_adapter_weights_for_saving(self, target_dtype_str):
        if target_dtype_str is None:
            return

        target_dtype = self.resolve_torch_dtype(target_dtype_str)
        changed = 0

        for name, param in self.model.named_parameters():
            if "lora_" in name or "vera_" in name:
                param.data = param.data.to(target_dtype)
                changed += 1

        print(f"Casted {changed} adapter tensors to {target_dtype}")

    def print_dtype_summary(self, stage=""):
        base_counts = self.get_base_model_param_dtype_counts()
        adapter_counts = self.get_adapter_param_dtype_counts()

        print(f"\nDtype summary [{stage}]")
        print("  Base model parameter dtypes:")
        if base_counts:
            for k, v in sorted(base_counts.items()):
                print(f"    {k}: {v}")
        else:
            print("    None found")

        print("  Adapter parameter dtypes:")
        if adapter_counts:
            for k, v in sorted(adapter_counts.items()):
                print(f"    {k}: {v}")
        else:
            print("    None found")

        # show a few concrete examples
        printed_base = False
        printed_adapter = False
        for name, param in self.model.named_parameters():
            if not printed_base and "lora_" not in name and "vera_" not in name:
                print(f"  Example base param: {name} -> {param.dtype}, requires_grad={param.requires_grad}")
                printed_base = True
            if not printed_adapter and ("lora_" in name or "vera_" in name):
                print(f"  Example adapter param: {name} -> {param.dtype}, requires_grad={param.requires_grad}")
                printed_adapter = True
            if printed_base and printed_adapter:
                break
        print()

    def get_base_model_param_dtype_counts(self):
        counts = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name or "vera_" in name:
                continue
            dtype_str = str(param.dtype)
            counts[dtype_str] = counts.get(dtype_str, 0) + 1
        return counts

    def get_adapter_param_dtype_counts(self):
        counts = {}
        for name, param in self.model.named_parameters():
            if "lora_" in name or "vera_" in name:
                dtype_str = str(param.dtype)
                counts[dtype_str] = counts.get(dtype_str, 0) + 1
        return counts