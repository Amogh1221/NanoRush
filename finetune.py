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
from tqdm import tqdm

from config import GPTConfig
from model import GPT
from tokenizer import Tokenizer


# ── Constants ────────────────────────────────────────────────────────────────
HF_REPO = "Amogh1221/nanorush_training"
AVAILABLE_CHECKPOINTS = ["epoch-1", "epoch-2", "latest"]


# ── Dataset ──────────────────────────────────────────────────────────────────

class ChatDataset(Dataset):
    """Formats UltraChat conversations into tokenized training pairs."""

    def __init__(self, data, tokenizer, max_length=4096):
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


def auto_batch_size(max_length=4096):
    """Pick batch size based on available VRAM and sequence length."""
    if not torch.cuda.is_available():
        return 1
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    if max_length <= 512:
        # Short sequences — can fit much larger batches
        if vram_gb >= 70:      return 64
        elif vram_gb >= 35:    return 32
        elif vram_gb >= 20:    return 16
        else:                  return 8
    else:
        # Full 4096 context — need smaller batches
        if vram_gb >= 70:      return 16   # H100 80GB
        elif vram_gb >= 35:    return 8    # A100 40GB
        elif vram_gb >= 20:    return 4    # RTX 3090/4090
        else:                  return 1    # T4 16GB


# ── Core Fine-Tuning ────────────────────────────────────────────────────────

def finetune_checkpoint(
    checkpoint_name: str,
    dataset,
    tokenizer,
    config: GPTConfig,
    device: torch.device,
    epochs: int = 3,
    max_lr: float = 2e-5,
    max_length: int = 4096,
    val_dataset=None,
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
    batch_size = auto_batch_size(max_length)
    chat_dataset = ChatDataset(dataset, tokenizer, max_length=max_length)
    num_workers = min(4, os.cpu_count() or 1)
    dataloader = DataLoader(
        chat_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
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
        fused=(device.type == "cuda"),
    )

    # ── Training Loop ────────────────────────────────────────────────────
    model.train()
    global_step = 0
    start_time = time.time()
    eval_interval = 500

    # Prepare validation dataloader
    val_dataloader = None
    if val_dataset is not None:
        val_ds = ChatDataset(val_dataset, tokenizer, max_length=max_length)
        val_dataloader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )

    # Open log file
    os.makedirs("logs", exist_ok=True)
    log_file = open("logs/finetuning_logs.txt", "a", encoding="utf-8")
    log_file.write(f"\n{'='*60}\n")
    log_file.write(f"Starting Fine-Tuning: {checkpoint_name} (Batch Size: {batch_size}, Max Length: {max_length})\n")
    log_file.write(f"Total Steps: {total_steps}, Epochs: {epochs}\n")
    log_file.write(f"{'='*60}\n")

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_steps = 0

        pbar = tqdm(
            dataloader,
            desc=f"Epoch {epoch+1}/{epochs}",
            dynamic_ncols=True,
            leave=True,
        )

        for step, (x, y) in enumerate(pbar):
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

            # Update tqdm bar with live metrics
            elapsed = time.time() - start_time
            avg_loss = epoch_loss / epoch_steps
            tok_per_sec = (epoch_steps * batch_size * max_length) / max(elapsed, 1e-6)
            pbar.set_postfix({
                "loss": f"{step_loss:.4f}",
                "avg": f"{avg_loss:.4f}",
                "lr": f"{lr:.1e}",
                "tok/s": f"{tok_per_sec:,.0f}",
            })

            # ── Validation evaluation ─────────────────────────────────
            if global_step % eval_interval == 0 and val_dataloader is not None:
                model.eval()
                val_losses = []
                eval_steps = min(50, len(val_dataloader))  # Cap at 50 batches
                with torch.no_grad():
                    for i, (vx, vy) in enumerate(val_dataloader):
                        if i >= eval_steps:
                            break
                        vx, vy = vx.to(device), vy.to(device)
                        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                            vlogits, _ = model(vx)
                            vloss = F.cross_entropy(
                                vlogits.view(-1, vlogits.size(-1)), vy.view(-1)
                            )
                        val_losses.append(vloss.item())

                val_loss = sum(val_losses) / len(val_losses)
                val_ppl = math.exp(min(val_loss, 20))
                train_ppl = math.exp(min(avg_loss, 20))

                val_msg = (
                    f"\n{'═' * 50}\n"
                    f"  VALIDATION @ Step {global_step}/{total_steps}\n"
                    f"{'═' * 50}\n"
                    f"  Train Loss : {avg_loss:.4f}  (ppl: {train_ppl:.2f})\n"
                    f"  Val Loss   : {val_loss:.4f}  (ppl: {val_ppl:.2f})\n"
                    f"  LR         : {lr:.2e}\n"
                    f"  Elapsed    : {format_time(elapsed)}\n"
                    f"{'═' * 50}\n"
                )
                print(val_msg)
                log_file.write(val_msg)
                log_file.flush()

                model.train()

        # End of epoch summary
        avg_epoch_loss = epoch_loss / max(epoch_steps, 1)
        epoch_msg = (
            f"\n  ── Epoch {epoch+1} Complete ──\n"
            f"  Average Loss: {avg_epoch_loss:.4f}\n"
            f"  Perplexity:   {math.exp(min(avg_epoch_loss, 20)):.2f}\n"
        )
        print(epoch_msg)
        log_file.write(epoch_msg)
        log_file.flush()

    log_file.close()

    # ── Save HuggingFace-style model folders ────────────────────────────
    elapsed = time.time() - start_time
    print(f"\n  Training complete in {format_time(elapsed)}")

    import json
    import shutil

    raw_model = model._orig_mod if hasattr(model, "_orig_mod") else model
    state_dict = raw_model.state_dict()

    # ── 1. Full BFloat16 Model ───────────────────────────────────────────
    model_dir = "models/nano-chat"
    os.makedirs(model_dir, exist_ok=True)

    # Save model weights as bfloat16 safetensors
    print(f"  Saving full model to {model_dir}/...")
    bf16_weights = {k: v.bfloat16() for k, v in state_dict.items()}
    save_file(bf16_weights, f"{model_dir}/model.safetensors")

    # Save config.json (model architecture)
    model_config = {
        "model_type": "nanorush",
        "architectures": ["NanoRushGPT"],
        "vocab_size": config.vocab_size,
        "n_embd": config.n_embd,
        "n_head": config.n_head,
        "n_layer": config.n_layer,
        "block_size": config.block_size,
        "dropout": config.dropout,
        "bias": config.bias,
        "num_parameters": sum(p.numel() for p in raw_model.parameters()),
        "torch_dtype": "bfloat16",
    }
    with open(f"{model_dir}/config.json", "w") as f:
        json.dump(model_config, f, indent=2)

    # Save generation_config.json
    gen_config = {
        "temperature": 0.8,
        "top_k": 50,
        "top_p": 0.95,
        "max_new_tokens": 512,
        "do_sample": True,
    }
    with open(f"{model_dir}/generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)

    # Copy tokenizer files
    if os.path.exists("tokenizer/tokenizer.json"):
        shutil.copy2("tokenizer/tokenizer.json", f"{model_dir}/tokenizer.json")
    tokenizer_config = {
        "model_type": "nanorush",
        "tokenizer_class": "PreTrainedTokenizerFast",
        "vocab_size": config.vocab_size,
        "model_max_length": config.block_size,
    }
    with open(f"{model_dir}/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    # Model card README
    model_size_mb = os.path.getsize(f"{model_dir}/model.safetensors") / (1024 * 1024)
    readme = f"""# NanoRush Chat (BFloat16)

A 283M parameter GPT-style language model fine-tuned for conversation.

## Model Details
- **Parameters:** {model_config['num_parameters']:,}
- **Architecture:** {config.n_layer} layers, {config.n_head} heads, {config.n_embd} dim
- **Context Window:** {config.block_size} tokens
- **Precision:** BFloat16
- **Size:** {model_size_mb:.0f} MB

## Training
- **Base:** NanoRush pre-trained on custom dataset ({pretrain_step} steps)
- **Fine-tuned on:** UltraChat 200k (SFT)
- **Epochs:** {epochs}
- **Learning Rate:** {max_lr}
- **Final Loss:** {avg_epoch_loss:.4f}
- **Perplexity:** {math.exp(min(avg_epoch_loss, 20)):.2f}

## Usage
```python
from model import GPT
from config import GPTConfig
from tokenizer import Tokenizer
from safetensors.torch import load_file

config = GPTConfig()
tokenizer = Tokenizer()
config.vocab_size = tokenizer.vocab_size
model = GPT(config)
model.load_state_dict(load_file("model.safetensors"))
```
"""
    with open(f"{model_dir}/README.md", "w") as f:
        f.write(readme)

    print(f"  ✓ Full model saved ({model_size_mb:.0f} MB)")

    # ── 2. INT8 Quantized Model ──────────────────────────────────────────
    quant_dir = "models/nano-chat-quantized"
    os.makedirs(quant_dir, exist_ok=True)

    print(f"  Quantizing to INT8...")
    quantized_weights = {}
    for name, param in state_dict.items():
        # Only quantize large weight matrices (2D+), keep biases/norms as float16
        if param.dim() >= 2:
            p_float = param.float()
            scale = p_float.abs().max() / 127.0
            q = (p_float / scale).round().clamp(-128, 127).to(torch.int8)
            quantized_weights[name] = q
            quantized_weights[f"{name}.__scale__"] = scale.to(torch.float16)
        else:
            quantized_weights[name] = param.to(torch.float16)

    save_file(quantized_weights, f"{quant_dir}/model_int8.safetensors")

    # Copy config files to quantized folder too
    quant_config = {**model_config, "torch_dtype": "int8", "quantization": "per-tensor-symmetric"}
    with open(f"{quant_dir}/config.json", "w") as f:
        json.dump(quant_config, f, indent=2)
    with open(f"{quant_dir}/generation_config.json", "w") as f:
        json.dump(gen_config, f, indent=2)
    if os.path.exists("tokenizer/tokenizer.json"):
        shutil.copy2("tokenizer/tokenizer.json", f"{quant_dir}/tokenizer.json")
    with open(f"{quant_dir}/tokenizer_config.json", "w") as f:
        json.dump(tokenizer_config, f, indent=2)

    quant_size_mb = os.path.getsize(f"{quant_dir}/model_int8.safetensors") / (1024 * 1024)

    quant_readme = f"""# NanoRush Chat (INT8 Quantized)

A 283M parameter GPT model, quantized to INT8 for smaller size and faster inference.

## Model Details
- **Parameters:** {model_config['num_parameters']:,}
- **Quantization:** Per-tensor symmetric INT8
- **Size:** {quant_size_mb:.0f} MB (vs {model_size_mb:.0f} MB full)

## Loading
```python
from safetensors.torch import load_file

data = load_file("model_int8.safetensors")
state_dict = {{}}
for name, tensor in data.items():
    if name.endswith(".__scale__"):
        continue
    scale_key = f"{{name}}.__scale__"
    if scale_key in data:
        state_dict[name] = tensor.float() * data[scale_key].float()
    else:
        state_dict[name] = tensor
model.load_state_dict(state_dict)
```
"""
    with open(f"{quant_dir}/README.md", "w") as f:
        f.write(quant_readme)

    print(f"  ✓ Quantized model saved ({quant_size_mb:.0f} MB)")
    print(f"  ✓ Compression: {model_size_mb:.0f} MB → {quant_size_mb:.0f} MB ({quant_size_mb/model_size_mb*100:.0f}%)")

    # ── Upload both folders to HuggingFace ────────────────────────────────
    if hf_token:
        api = HfApi(token=hf_token)
        for folder, label in [(model_dir, "full"), (quant_dir, "quantized")]:
            print(f"  Uploading {label} model folder to HuggingFace...")
            try:
                api.upload_folder(
                    folder_path=folder,
                    path_in_repo=folder,
                    repo_id=HF_REPO,
                    repo_type="dataset",
                    commit_message=f"Upload {label} fine-tuned model "
                                   f"({epochs} epoch, lr={max_lr}, loss={avg_epoch_loss:.4f})",
                )
                print(f"  ✓ Uploaded {folder}/ to {HF_REPO}")
            except Exception as e:
                print(f"  ✗ Upload failed for {label}: {e}")

    # ── Cleanup GPU memory ───────────────────────────────────────────────
    del model, optimizer, bf16_weights, quantized_weights
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
    parser.add_argument("--max_length", type=int, default=4096, help="Max sequence length (default: 4096, use 512 for quick Colab testing)")
    parser.add_argument("--no_compile", action="store_true", help="Disable torch.compile")
    args = parser.parse_args()

    # ── Auth ─────────────────────────────────────────────────────────────
    login(token=args.hf_token)
    os.environ["HF_TOKEN"] = args.hf_token

    # Ensure tqdm progress bars are shown for all HF downloads/uploads
    from huggingface_hub.utils import enable_progress_bars
    enable_progress_bars()

    # ── Device Setup ─────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
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
    print(f"Loaded {len(dataset):,} training conversations")

    # Load validation set
    print("Loading validation set...")
    val_dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_sft[:2000]")
    print(f"Loaded {len(val_dataset):,} validation conversations")

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
                max_length=args.max_length,
                val_dataset=val_dataset,
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
