import os
import sys
import json
import torch
import argparse
from huggingface_hub import login, hf_hub_download

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

from config import GPTConfig
from tokenizer import Tokenizer
from trainer import Trainer


def sync_huggingface(repo_id: str):
    print("Syncing dataset and tokenizer from HuggingFace...")
    os.makedirs("data", exist_ok=True)
    os.makedirs("tokenizer", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Download dataset and tokenizer
    try:
        hf_hub_download(repo_id=repo_id, filename="train.bin", repo_type="dataset", local_dir="data")
        hf_hub_download(repo_id=repo_id, filename="val.bin", repo_type="dataset", local_dir="data")
        hf_hub_download(repo_id=repo_id, filename="tokenizer/tokenizer.json", repo_type="dataset", local_dir=".")
    except Exception as e:
        print(f"Failed to download dataset or tokenizer: {e}")
    
    # Download latest checkpoint and logs if they exist
    print("Checking for existing checkpoints and logs...")
    try:
        hf_hub_download(repo_id=repo_id, filename="checkpoints/latest.pt", repo_type="dataset", local_dir=".")
        print("Successfully downloaded latest.pt")
    except Exception as e:
        print("No existing checkpoint found on HuggingFace.")
        
    try:
        hf_hub_download(repo_id=repo_id, filename="logs/training_log.txt", repo_type="dataset", local_dir=".")
        print("Successfully downloaded training_log.txt")
    except Exception as e:
        print("No existing training log found on HuggingFace.")
        
    # Disable HF progress bars for background uploads during training
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace WRITE Token")
    args = parser.parse_args()
    
    print("Authenticating with HuggingFace...")
    login(token=args.hf_token)
    os.environ["HF_TOKEN"] = args.hf_token
    
    repo_id = "Amogh1221/nanorush_training"
    sync_huggingface(repo_id)

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
