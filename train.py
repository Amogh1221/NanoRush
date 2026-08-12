import os
import sys
import json
import torch

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from config import GPTConfig
from tokenizer import Tokenizer
from trainer import Trainer


def setup_environment(config: GPTConfig):
    if config.tf32 and config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    if config.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")
        print(f"AMP dtype: {config.dtype}")


def main():
    config_path = "config.json"
    if os.path.exists(config_path):
        print(f"Loading config from {config_path}")
        with open(config_path) as f:
            config = GPTConfig(**json.load(f))
    else:
        config = GPTConfig()
        config.save("config.json")
        print(f"Created default config at {config_path}")

    setup_environment(config)

    tokenizer = Tokenizer()
    config.vocab_size = tokenizer.vocab_size

    trainer = Trainer(config, tokenizer)

    resume_path = "checkpoints/latest.pt"
    if os.path.exists(resume_path):
        trainer.load_checkpoint(resume_path)

    try:
        trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted, saving checkpoint...")
        trainer.save_checkpoint("checkpoints/latest.pt", step_num=trainer.iter_num)
        print("Checkpoint saved. Exiting.")
        sys.exit(0)


if __name__ == "__main__":
    main()
