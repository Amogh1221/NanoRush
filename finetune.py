"""
finetune.py — NanoRush Chat Fine-Tuning Script
================================================
Fine-tunes pre-trained NanoRush checkpoints on the UltraChat 200k dataset
to create conversational AI models.

Features:
  - Fine-tunes multiple base checkpoints for comparison (epoch-1, epoch-2, latest)
  - Full H100/A100 optimizations (TF32, bfloat16 autocast, torch.compile)
  - Auto-scales batch size based on available VRAM
  - Saves fine-tuned models as compact safetensors (~560MB each)
  - Uploads results to HuggingFace under models/ directory
  - Gradient accumulation for larger effective batch sizes
  - Cosine learning rate schedule with warmup

Usage:
  python finetune.py --hf_token YOUR_TOKEN
  python finetune.py --hf_token YOUR_TOKEN --base_checkpoint latest
  python finetune.py --hf_token YOUR_TOKEN --base_checkpoint all --epochs 3
"""

import os
import sys
import math
import time
import argparse
from datetime import datetime

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from datasets import load_dataset
from safetensors.torch import save_file
from huggingface_hub import hf_hub_download, HfApi, login

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer


# ── Constants ────────────────────────────────────────────────────────────────
HF_REPO = "Amogh1221/nanorush_training"
AVAILABLE_CHECKPOINTS = ["epoch-1", "epoch-2", "latest"]


# ── Dataset ──────────────────────────────────────────────────────────────────

class ChatDataset(Dataset):
    """Formats UltraChat conversations into tokenized training pairs."""

    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages = self.data[idx]["messages"]

        # Format: "User: ...\nAssistant: ...\n"
        text = ""
        for msg in messages:
            role = "User" if msg["role"] == "user" else "Assistant"
            text += f"{role}: {msg['content']}\n"

        # Tokenize and append end-of-text token
        tokens = self.tokenizer.encode(text)
        tokens.append(self.tokenizer.eot_token)

        # Truncate to max_length
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]

        # Pad with zeros
        padded = tokens + [0] * (self.max_length - len(tokens))

        # X = tokens[0..N-1], Y = tokens[1..N]
        x = torch.tensor(padded[:-1], dtype=torch.long)
        y = torch.tensor(padded[1:], dtype=torch.long)
        return x, y


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_cosine_lr(step, total_steps, max_lr=2e-5, min_lr=2e-6, warmup_steps=100):
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / (warmup_steps + 1)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def format_time(seconds):
    """Format seconds as HH:MM:SS."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def auto_batch_size():
    """Pick batch size based on available VRAM (for full 4096 context window)."""
    if not torch.cuda.is_available():
        return 1
    vram_gb = torch.cuda.get_device_properties(0).total_mem / 1e9
    if vram_gb >= 70:      # H100 80GB / A100 80GB
        return 16
    elif vram_gb >= 35:    # A100 40GB
        return 8
    elif vram_gb >= 20:    # RTX 3090/4090
        return 4
    else:                  # T4 16GB
        return 2


# ── Core Fine-Tuning ────────────────────────────────────────────────────────

def finetune_checkpoint(
    checkpoint_name: str,
    dataset,
    tokenizer,
    config: GPTConfig,
    device: torch.device,
    epochs: int = 3,
    max_lr: float = 2e-5,
    hf_token: str = None,
    use_compile: bool = True,
):
    """Fine-tune a single pre-trained checkpoint and save the result."""

    print(f"\n{'═' * 60}")
    print(f"  FINE-TUNING: {checkpoint_name}")
    print(f"{'═' * 60}")

    # ── Download checkpoint ──────────────────────────────────────────────
    ckpt_filename = f"checkpoints/{checkpoint_name}.pt"
    local_path = ckpt_filename

    if not os.path.exists(local_path):
        print(f"Downloading {ckpt_filename} from HuggingFace...")
        hf_hub_download(
            repo_id=HF_REPO,
            filename=ckpt_filename,
            repo_type="dataset",
            local_dir=".",
        )

    # ── Load model ───────────────────────────────────────────────────────
    model = GPT(config).to(device)
    ckpt = torch.load(local_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    pretrain_step = ckpt.get("iter_num", "unknown")
    del ckpt
    torch.cuda.empty_cache()
    print(f"Loaded weights from pre-training step {pretrain_step}")

    # ── Compile for speed (H100/A100) ────────────────────────────────────
    if use_compile and hasattr(torch, "compile"):
        print("Compiling model with torch.compile (this takes ~60s the first time)...")
        model = torch.compile(model, mode="default")

    # ── DataLoader ───────────────────────────────────────────────────────
    batch_size = auto_batch_size()
    chat_dataset = ChatDataset(dataset, tokenizer, max_length=config.block_size)
    dataloader = DataLoader(
        chat_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )

    total_steps = len(dataloader) * epochs
    print(f"Batch size: {batch_size}")
    print(f"Steps per epoch: {len(dataloader)}")
    print(f"Total steps: {total_steps}")
    print(f"Learning rate: {max_lr}")

    # ── Optimizer ────────────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    # ── Training Loop ────────────────────────────────────────────────────
    model.train()
    global_step = 0
    start_time = time.time()
    best_loss = float("inf")
    log_interval = 50

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        for step, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)

            # Update learning rate
            lr = get_cosine_lr(global_step, total_steps, max_lr=max_lr)
            for pg in optimizer.param_groups:
                pg["lr"] = lr

            # Forward pass with bfloat16 autocast (H100 Tensor Cores)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), y.view(-1)
                )

            # Backward pass
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step_loss = loss.item()
            epoch_loss += step_loss
            epoch_steps += 1
            global_step += 1

            # Logging
            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                steps_per_sec = global_step / elapsed
                eta = (total_steps - global_step) / max(steps_per_sec, 1e-6)
                avg_loss = epoch_loss / epoch_steps

                print(
                    f"  [{checkpoint_name}] Epoch {epoch+1}/{epochs} | "
                    f"Step {global_step}/{total_steps} | "
                    f"Loss: {step_loss:.4f} | Avg: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | "
                    f"Speed: {steps_per_sec:.1f} steps/s | "
                    f"ETA: {format_time(eta)}"
                )

        # End of epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        print(f"\n  ── Epoch {epoch+1} Complete ──")
        print(f"  Average Loss: {avg_epoch_loss:.4f}")
        print(f"  Perplexity:   {math.exp(min(avg_epoch_loss, 20)):.2f}")

    # ── Save as Safetensors ──────────────────────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n  Training complete in {format_time(elapsed)}")

    os.makedirs("models", exist_ok=True)
    model_name = f"nano-chat-{checkpoint_name}"
    safetensors_path = f"models/{model_name}.safetensors"

    print(f"  Saving {safetensors_path}...")
    # Extract weights from compiled model if needed
    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    weights = {k: v.bfloat16() for k, v in raw_model.state_dict().items()}
    save_file(weights, safetensors_path)

    file_size_mb = os.path.getsize(safetensors_path) / (1024 * 1024)
    print(f"  Saved! ({file_size_mb:.0f} MB)")

    # ── Upload to HuggingFace ────────────────────────────────────────────
    if hf_token:
        print(f"  Uploading {model_name} to HuggingFace...")
        try:
            api = HfApi(token=hf_token)
            api.upload_file(
                path_or_fileobj=safetensors_path,
                path_in_repo=f"models/{model_name}.safetensors",
                repo_id=HF_REPO,
                repo_type="dataset",
                commit_message=f"Upload fine-tuned model: {model_name} "
                               f"({epochs} epochs, lr={max_lr}, "
                               f"loss={avg_epoch_loss:.4f})",
            )
            print(f"  ✓ Uploaded to {HF_REPO}/models/{model_name}.safetensors")
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")

    # ── Cleanup GPU memory for next model ────────────────────────────────
    del model, optimizer, weights
    torch.cuda.empty_cache()

    return avg_epoch_loss


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NanoRush Chat Fine-Tuning")
    parser.add_argument("--hf_token", type=str, required=True, help="HuggingFace WRITE token")
    parser.add_argument(
        "--base_checkpoint",
        type=str,
        default="latest",
        choices=["epoch-1", "epoch-2", "latest", "all"],
        help="Which pre-training checkpoint to fine-tune (default: all)",
    )
    parser.add_argument("--epochs", type=int, default=1, help="Number of fine-tuning epochs (default: 1)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Peak learning rate (default: 2e-5)")
    parser.add_argument("--subset", type=int, default=0, help="Use N conversations (0 = full dataset)")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # ── Auth ─────────────────────────────────────────────────────────────
    login(token=args.hf_token)
    os.environ["HF_TOKEN"] = args.hf_token

    # ── Device Setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / 1e9
        print(f"GPU: {gpu_name} ({vram:.1f} GB)")

        # Enable H100/A100 Tensor Core acceleration
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("TF32 acceleration: ENABLED")
    else:
        print("WARNING: No GPU found, falling back to CPU (this will be very slow)")

    # ── Download Tokenizer ───────────────────────────────────────────────
    os.makedirs("tokenizer", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    print("Downloading tokenizer...")
    hf_hub_download(
        repo_id=HF_REPO,
        filename="tokenizer/tokenizer.json",
        repo_type="dataset",
        local_dir=".",
    )
    tokenizer = Tokenizer()

    # ── Load Config ──────────────────────────────────────────────────────
    config = GPTConfig()
    config.vocab_size = tokenizer.vocab_size
    n_params = sum(
        p.numel()
        for p in GPT(GPTConfig(vocab_size=config.vocab_size)).parameters()
    )
    print(f"Model parameters: {n_params:,}")

    # ── Load Dataset ─────────────────────────────────────────────────────
    print("Downloading UltraChat 200k dataset...")
    if args.subset > 0:
        split = f"train_sft[:{args.subset}]"
        print(f"Using subset: {args.subset:,} conversations")
    else:
        split = "train_sft"
        print("Using FULL dataset")

    dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split=split)
    print(f"Loaded {len(dataset):,} conversations")

    # ── Determine which checkpoints to fine-tune ─────────────────────────
    if args.base_checkpoint == "all":
        checkpoints = AVAILABLE_CHECKPOINTS
    else:
        checkpoints = [args.base_checkpoint]

    print(f"\nWill fine-tune {len(checkpoints)} model(s): {checkpoints}")
    print(f"Epochs per model: {args.epochs}")
    print(f"Peak learning rate: {args.lr}")

    # ── Fine-tune each checkpoint ────────────────────────────────────────
    results = {}
    total_start = time.time()

    for ckpt_name in checkpoints:
        try:
            final_loss = finetune_checkpoint(
                checkpoint_name=ckpt_name,
                dataset=dataset,
                tokenizer=tokenizer,
                config=config,
                device=device,
                epochs=args.epochs,
                max_lr=args.lr,
                hf_token=args.hf_token,
                use_compile=not args.no_compile,
            )
            results[ckpt_name] = final_loss
        except Exception as e:
            print(f"\n  ✗ FAILED to fine-tune {ckpt_name}: {e}")
            results[ckpt_name] = None
            import traceback
            traceback.print_exc()

    # ── Final Summary ────────────────────────────────────────────────────
    total_elapsed = time.time() - total_start
    print(f"\n{'═' * 60}")
    print(f"  FINE-TUNING COMPLETE — {format_time(total_elapsed)}")
    print(f"{'═' * 60}")
    print(f"  {'Checkpoint':<15} {'Final Loss':<12} {'Perplexity':<12} {'Model File'}")
    print(f"  {'─' * 55}")
    for name, loss in results.items():
        if loss is not None:
            ppl = math.exp(min(loss, 20))
            print(f"  {name:<15} {loss:<12.4f} {ppl:<12.2f} models/nano-chat-{name}.safetensors")
        else:
            print(f"  {name:<15} {'FAILED':<12} {'—':<12}")
    print(f"{'═' * 60}")

    if results:
        best = min(
            ((k, v) for k, v in results.items() if v is not None),
            key=lambda x: x[1],
            default=None,
        )
        if best:
            print(f"\n  🏆 Best model: nano-chat-{best[0]} (loss: {best[1]:.4f})")
    print()


if __name__ == "__main__":
    main()
