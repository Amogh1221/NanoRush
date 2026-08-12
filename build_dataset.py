#!/usr/bin/env python3
"""
build_dataset.py  —  Cosmopedia corpus downloader
===================================================
Downloads text from HuggingFaceTB/cosmopedia (all 8 configs, proportionally)
and writes a single plain-text corpus file for tokenizer training.

Usage:
    python build_dataset.py
    # Enter target size in GB (0.1 - 100)

Output:
    data/corpus.txt          – single UTF-8 text file (docs separated by \\n\\n\\n)
    data/dataset_stats.json  – metadata
"""

import gc
import json
import logging
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_dataset")

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

BASE = Path(__file__).resolve().parent
OUTPUT_DIR = BASE / "data"

# ──────────────────────────────────────────────────────────────
#  Cosmopedia configs and target proportions
# ──────────────────────────────────────────────────────────────

# All 8 available configs with proportions (must sum to 1.0)
COSMOPEDIA_CONFIGS = [
    ("web_samples_v2", 0.25),   # large web text (newest)
    ("web_samples_v1", 0.20),   # large web text
    ("stories",        0.15),   # narrative / creative text
    ("auto_math_text", 0.15),   # math-heavy text
    ("wikihow",        0.10),   # how-to instructions
    ("stanford",       0.07),   # academic / stanford courses
    ("openstax",       0.05),   # textbook content
    ("khanacademy",    0.03),   # educational Q&A
]

# ──────────────────────────────────────────────────────────────
#  Text cleaning
# ──────────────────────────────────────────────────────────────


def clean_text(text: str) -> str:
    """Normalize Unicode, fix line endings, strip control chars, collapse blank lines."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def is_valid_document(text: str) -> bool:
    """Reject very short documents."""
    return len(text) >= 300


# ──────────────────────────────────────────────────────────────
#  Per-config download
# ──────────────────────────────────────────────────────────────


def download_config(
    config_name: str,
    target_bytes: int,
    outf,
    seen_hashes: set,
) -> tuple[int, int]:
    """
    Stream one Cosmopedia config subset until target_bytes is reached.

    Returns (bytes_written, docs_written).
    """
    log.info("  Loading config: %s  (target %.0f MB)", config_name, target_bytes / 1e6)

    dataset = load_dataset(
        "HuggingFaceTB/cosmopedia",
        name=config_name,
        split="train",
        streaming=True,
    )

    current_size = 0
    total_docs = 0
    batch: list[str] = []
    FLUSH_EVERY = 500

    pbar = tqdm(
        total=target_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
        desc=f"  {config_name}",
        leave=True,
    )

    for example in dataset:
        if current_size >= target_bytes:
            break

        raw = example.get("text", "")
        if not raw or not raw.strip():
            continue

        text = clean_text(raw)
        if not is_valid_document(text):
            continue

        h = hash(text)
        if h in seen_hashes:
            continue
        seen_hashes.add(h)

        doc_bytes = len(text.encode("utf-8"))
        batch.append(text)
        current_size += doc_bytes
        total_docs += 1
        pbar.update(doc_bytes)

        if len(batch) >= FLUSH_EVERY:
            for doc in batch:
                outf.write(doc + "\n\n\n")
            outf.flush()
            batch = []
            gc.collect()

    # Flush remaining
    for doc in batch:
        outf.write(doc + "\n\n\n")
    outf.flush()

    pbar.close()
    log.info("  ✓ %s: %d docs, %.2f MB", config_name, total_docs, current_size / 1e6)
    return current_size, total_docs


# ──────────────────────────────────────────────────────────────
#  Statistics
# ──────────────────────────────────────────────────────────────


def compute_statistics(corpus_path: Path, total_docs: int, elapsed: float) -> dict:
    file_size = corpus_path.stat().st_size if corpus_path.exists() else 0
    est_tokens = max(1, int(file_size / 4.3))
    return {
        "source": "HuggingFaceTB/cosmopedia",
        "configs": [c for c, _ in COSMOPEDIA_CONFIGS],
        "total_documents": total_docs,
        "total_bytes": file_size,
        "estimated_tokens": est_tokens,
        "final_file_size_bytes": file_size,
        "elapsed_seconds": round(elapsed, 1),
    }


def print_statistics(stats: dict):
    hr = "=" * 64
    print()
    print(hr)
    print("  DATASET STATISTICS")
    print(hr)
    print(f"  Source          : {stats['source']}")
    print(f"  Configs         : {', '.join(stats['configs'])}")
    print(f"  Documents       : {stats['total_documents']:>12,}")
    print(f"  Total bytes     : {stats['total_bytes']:>12,}  ({stats['total_bytes'] / 1e9:.2f} GB)")
    print(f"  Estimated tokens: {stats['estimated_tokens']:>12,}")
    print(f"  Elapsed         : {stats['elapsed_seconds']:.0f}s")
    print(hr)


# ──────────────────────────────────────────────────────────────
#  CLI
# ──────────────────────────────────────────────────────────────


def main():
    print()
    print("=" * 56)
    print("  nanorush — Cosmopedia Dataset Downloader")
    print("=" * 56)
    print()
    print("  Source : HuggingFaceTB/cosmopedia (8 configs)")
    print("  Output : data/corpus.txt")
    print()

    try:
        raw = input("Enter target dataset size in GB (0.1 - 100): ").strip()
        target_gb = float(raw)
        if target_gb < 0.1 or target_gb > 100:
            log.error("Target must be between 0.1 and 100 GB")
            sys.exit(1)
    except (ValueError, EOFError):
        log.error("Invalid input")
        sys.exit(1)

    target_bytes = int(target_gb * 1_000_000_000)

    log.info("")
    log.info("Target: %.2f GB (%s bytes)", target_gb, f"{target_bytes:,}")
    log.info("─" * 52)
    log.info("Config breakdown:")
    for cfg, pct in COSMOPEDIA_CONFIGS:
        log.info("  %-20s  %5.0f MB  (%4.0f%%)", cfg, target_bytes * pct / 1e6, pct * 100)
    log.info("─" * 52)

    corpus_path = OUTPUT_DIR / "corpus.txt"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    seen_hashes: set[int] = set()
    total_bytes_written = 0
    total_docs_written = 0

    start_time = time.time()

    with open(corpus_path, "w", encoding="utf-8") as outf:
        for config_name, proportion in COSMOPEDIA_CONFIGS:
            config_target = int(target_bytes * proportion)
            log.info("")
            try:
                b, d = download_config(config_name, config_target, outf, seen_hashes)
                total_bytes_written += b
                total_docs_written += d
            except Exception as e:
                log.error("  ✗ Config %s failed: %s", config_name, e)

    elapsed = time.time() - start_time

    # Save stats
    stats = compute_statistics(corpus_path, total_docs_written, elapsed)
    stats_path = OUTPUT_DIR / "dataset_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print_statistics(stats)

    log.info("Done! Dataset: %s (%.2f GB)", corpus_path, corpus_path.stat().st_size / 1e9)
    log.info("Stats : %s", stats_path)
    log.info("")
    log.info("Next steps:")
    log.info("  1. python train_tokenizer.py   — train BPE tokenizer on the corpus")
    log.info("  2. python tokenize_dataset.py  — pre-tokenize into train.bin + val.bin")
    log.info("  3. python train.py             — start training")


if __name__ == "__main__":
    main()
