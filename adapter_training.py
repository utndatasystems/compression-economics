import os

from src.config import get_adapter_training_args
from src.adapter_trainer import AdapterTrainer


def main():
    args = get_adapter_training_args()
    trainer = AdapterTrainer(args)
    # if exist skip
    if os.path.exists(trainer.output_path):
        print(f"Skipping training for existing path: {trainer.output_path}")
        return
    
    if args.mode == "quantize":
        trainer.quantize()
    if args.mode == "finetune":
        trainer.finetune()
    
if __name__ == "__main__":
    main()
