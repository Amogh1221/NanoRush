"""
dataset.py  —  nano_brain dataset loader
=========================================
Supports two input formats:

  1. Pre-tokenised binary (.bin) — RECOMMENDED
     Produced by tokenize_dataset.py.
     Files are uint16 numpy arrays memory-mapped from disk.
     - Train: data/train.bin
     - Val:   data/val.bin
     - Load time: milliseconds, regardless of dataset size.
     - RAM:  virtually zero (OS pages data in on demand).

  2. Plain-text (.txt) — legacy / small datasets only
     Reads the whole file, tokenises with tiktoken, and splits 90/10.
     For datasets > 1 GB this is slow (minutes) and uses a lot of RAM.
     Included for backwards-compatibility with corpus.txt.

Usage in trainer.py (unchanged — the API is the same):
    data = load_data(config.data_dir, config.dataset, tokenizer)
    train_loader, val_loader = create_dataloaders(data, ...)

For binary format, set in config.json:
    "dataset": "train.bin"
The loader automatically finds the matching "val.bin" in the same dir.
"""

import os
import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

log = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset classes
# ──────────────────────────────────────────────────────────────────────────────


class BinDataset(Dataset):
    """
    Dataset that reads uint16 token ids from a .bin file.

    When preload=True (default), the entire file is loaded into RAM as a
    torch.Tensor — zero disk reads during training.  Requires enough system
    memory to hold the dataset (~1 GB per 500M tokens).

    When preload=False, uses np.memmap (zero-copy, OS-paged) for datasets
    larger than available RAM.

    Returns (x, y) pairs of shape (block_size,) with dtype torch.long.
    """

    def __init__(self, bin_path: str, block_size: int, preload: bool = True):
        self.block_size = block_size
        self.preload = preload
        if preload:
            arr = np.fromfile(bin_path, dtype=np.uint16)
            self.data = torch.from_numpy(arr.astype(np.int32))
            del arr
        else:
            self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        n_tokens = len(self.data)
        n_blocks = n_tokens - block_size
        if n_blocks <= 0:
            raise ValueError(
                f"Dataset at {bin_path} has {n_tokens} tokens but "
                f"block_size={block_size} requires at least {block_size + 1}."
            )
        log.info(
            "Loaded %-10s  %s tokens  →  %s windows  (preload=%s)",
            Path(bin_path).name,
            f"{n_tokens:,}",
            f"{n_blocks:,}",
            preload,
        )

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        if self.preload:
            x = self.data[idx : idx + self.block_size]
            y = self.data[idx + 1 : idx + self.block_size + 1]
        else:
            x = torch.from_numpy(
                self.data[idx : idx + self.block_size].astype(np.int64)
            )
            y = torch.from_numpy(
                self.data[idx + 1 : idx + self.block_size + 1].astype(np.int64)
            )
        return x, y


class TextDataset(Dataset):
    """
    In-memory dataset backed by a 1-D torch.long tensor.
    Used for plain-text (.txt) inputs loaded with load_data().
    """

    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int):
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + self.block_size + 1]
        return x, y


# ──────────────────────────────────────────────────────────────────────────────
# load_data — returns (train_dataset, val_dataset) or a flat tensor
# ──────────────────────────────────────────────────────────────────────────────


def _resolve_path(data_dir: str, filename: str) -> str:
    """Search a few candidate locations and return the first that exists."""
    candidates = [
        os.path.join(data_dir, filename),
        os.path.join("..", data_dir, filename),
        filename,
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    raise FileNotFoundError(
        f"Dataset file '{filename}' not found. Searched: {candidates}"
    )


def load_data(data_dir: str, dataset_name: str, tokenizer):
    """
    Load training data.

    - If dataset_name ends with '.bin': returns a (train_path, val_path)
      tuple of resolved file paths for BinDataset.
    - If dataset_name ends with '.txt': tokenises and returns a torch.Tensor
      (legacy path).
    """
    path = _resolve_path(data_dir, dataset_name)

    if path.endswith(".bin"):
        # Binary path — return (train_path, val_path)
        train_path = path
        val_path = path.replace("train.bin", "val.bin")
        if not os.path.exists(val_path):
            raise FileNotFoundError(
                f"Found {train_path} but cannot find matching {val_path}. "
                "Run tokenize_dataset.py to regenerate both files."
            )
        log.info("Binary dataset mode (memmap)")
        log.info("  train : %s  (%.1f GB)", train_path,
                 os.path.getsize(train_path) / 1e9)
        log.info("  val   : %s  (%.1f GB)", val_path,
                 os.path.getsize(val_path) / 1e9)
        return (train_path, val_path)

    # ── Legacy: plain-text .txt ───────────────────────────────────────────
    log.warning(
        "Loading plain-text dataset (slow for large files). "
        "Run tokenize_dataset.py for faster loading."
    )
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    data = torch.tensor(tokenizer.encode(text), dtype=torch.long)
    log.info("Tokenised %s  →  %s tokens", path, f"{len(data):,}")
    return data


# ──────────────────────────────────────────────────────────────────────────────
# create_dataloaders
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────────────────────────────────────────────────────
# Fast tensor loader (replaces DataLoader — avoids torch.randperm OOM)
# ──────────────────────────────────────────────────────────────────────────────


def load_bin_tensors(data_dir: str, dataset_name: str, preload: bool = True) -> tuple:
    """
    Load train.bin and val.bin as uint16 numpy arrays.

    Returns (train_data, val_data) where each is a flat 1D uint16 array.
    preload=True  → np.fromfile (9 GB for 4.5B tokens in RAM).
    preload=False → np.memmap (OS-paged, near-zero RAM).

    The Trainer converts slices to torch.long on the fly in get_batch.
    """
    import numpy as np
    from pathlib import Path

    path = _resolve_path(data_dir, dataset_name)
    if not path.endswith(".bin"):
        raise ValueError("load_bin_tensors requires .bin files")

    train_path = path
    val_path = path.replace("train.bin", "val.bin")
    if not os.path.exists(val_path):
        raise FileNotFoundError(
            f"Found {train_path} but cannot find {val_path}. "
            "Run tokenize_dataset.py first."
        )

    def _load_one(p: str):
        if preload:
            arr = np.fromfile(p, dtype=np.uint16)
            log.info("Loaded %-12s  %s tokens", Path(p).name, f"{len(arr):,}")
            return arr
        else:
            mm = np.memmap(p, dtype=np.uint16, mode="r")
            log.info("Mmap   %-12s  %s tokens", Path(p).name, f"{len(mm):,}")
            return mm

    return _load_one(train_path), _load_one(val_path)


def create_dataloaders(
    data,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
    pin_memory: bool = True,
    split_ratio: float = 0.9,
    preload: bool = True,
):
    """
    Build train and val DataLoaders.

    data: either
      - tuple (train_bin_path, val_bin_path) → BinDataset
      - torch.Tensor                         → TextDataset (split by split_ratio)
    """
    if isinstance(data, tuple):
        train_path, val_path = data
        train_dataset = BinDataset(train_path, block_size, preload=preload)
        val_dataset   = BinDataset(val_path,   block_size, preload=preload)
    else:
        # Legacy in-memory tensor
        n = int(split_ratio * len(data))
        train_dataset = TextDataset(data[:n], block_size)
        val_dataset   = TextDataset(data[n:], block_size)

    # On Windows num_workers > 0 requires the __main__ guard in the training
    # script, and can sometimes deadlock.  The trainer passes num_workers=0
    # by default which is safe everywhere.
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=(num_workers > 0),
        drop_last=True,
    )

    return train_loader, val_loader
