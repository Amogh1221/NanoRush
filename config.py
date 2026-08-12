from dataclasses import dataclass, asdict
import json


@dataclass
class GPTConfig:
    # Model architecture — GPT-2 Small (124M)
    vocab_size: int = 32768
    n_embd: int = 768
    n_head: int = 12
    n_layer: int = 40
    block_size: int = 4096
    dropout: float = 0.0
    bias: bool = False

    # Training
    batch_size: int = 2
    gradient_accumulation_steps: int = 40
    max_iters: int = 80000
    learning_rate: float = 6e-4
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    warmup_iters: int = 2000
    lr_decay_iters: int = 80000
    min_lr: float = 6e-5
    eval_interval: int = 500
    eval_iters: int = 200
    log_interval: int = 10
    save_interval: int = 1000
    gen_interval: int = 5000
    max_new_tokens_gen: int = 256
    num_generations: int = 3

    # System
    device: str = "cuda"
    dtype: str = "bfloat16"
    compile: bool = False       # keep False on Windows
    fused_adam: bool = True
    tf32: bool = True

    # Dataset
    dataset: str = "train.bin"
    data_dir: str = "data"

    # Generation defaults
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.95

    # EMA
    ema_decay: float = 0.999
    use_ema: bool = True

    # Gradient clipping
    grad_clip: float = 1.0

    # Gradient checkpointing stride (0 = off, 1 = all blocks, 2 = every 2nd, etc.)
    gradient_checkpointing: int = 1

    # Preload dataset into RAM (disable for datasets larger than available RAM)
    preload: bool = True

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str):
        with open(path) as f:
            return cls(**json.load(f))
