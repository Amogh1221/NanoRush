#!/usr/bin/env python3
"""
train_tokenizer.py  —  Custom BPE Tokenizer Trainer
=====================================================
Trains a BPE tokenizer on data/corpus.txt with custom tokens for:
  - Python indentation spaces (2, 4, 8, 12, 16 spaces)
  - Double-digit numbers: "00" through "99"
  - Space-prefixed double-digit numbers: " 00" through " 99"

Usage:
    python build_dataset.py      # first, download corpus
    python train_tokenizer.py    # then, train tokenizer

Output:
    tokenizer/tokenizer.json     — HuggingFace tokenizers JSON
"""

import logging
import time
from pathlib import Path

from tokenizers import Tokenizer, models, pre_tokenizers, trainers, decoders

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("train_tokenizer")

BASE = Path(__file__).resolve().parent
CORPUS_PATH = BASE / "data" / "corpus.txt"
OUTPUT_DIR = BASE / "tokenizer"
VOCAB_SIZE = 32768
MIN_FREQUENCY = 2

# ──────────────────────────────────────────────────────────────
#  Custom tokens
# ──────────────────────────────────────────────────────────────

# End-of-text separator
EOT_TOKEN = "<|endoftext|>"

# Python indentation spaces: 2, 4, 8, 12, 16 spaces
INDENT_TOKENS = [
    " " * n for n in (2, 4, 8, 12, 16)
]

# Double-digit numbers: "00", "01", ..., "99"
DIGIT_TOKENS = [f"{i:02d}" for i in range(100)]

# Space-prefixed double-digit numbers: " 00", " 01", ..., " 99"
SPACE_DIGIT_TOKENS = [f" {i:02d}" for i in range(100)]

# All special tokens — EOT first (will get ID 0)
SPECIAL_TOKENS = [EOT_TOKEN] + INDENT_TOKENS + DIGIT_TOKENS + SPACE_DIGIT_TOKENS


def main():
    if not CORPUS_PATH.exists():
        log.error("Corpus not found at %s", CORPUS_PATH)
        log.error("Run  python build_dataset.py  first.")
        return

    corpus_size = CORPUS_PATH.stat().st_size

    print()
    print("=" * 56)
    print("  nanorush — Custom BPE Tokenizer Trainer")
    print("=" * 56)
    print()
    log.info("Corpus    : %s  (%.1f GB)", CORPUS_PATH, corpus_size / 1e9)
    log.info("Vocab size: %d", VOCAB_SIZE)
    log.info("")

    # ── List custom tokens ───────────────────────────────────────
    log.info("Custom tokens (%d total):", len(SPECIAL_TOKENS))
    log.info("  EOT           : %r", EOT_TOKEN)
    log.info("  Indent spaces : %d tokens (2, 4, 8, 12, 16 spaces)", len(INDENT_TOKENS))
    log.info("  Digit pairs   : %d tokens (\"00\" .. \"99\")", len(DIGIT_TOKENS))
    log.info("  Space+digits  : %d tokens (\" 00\" .. \" 99\")", len(SPACE_DIGIT_TOKENS))
    log.info("")

    # ── Build tokenizer ──────────────────────────────────────────
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Split(pattern=r'\d', behavior='isolated'),
        pre_tokenizers.ByteLevel(add_prefix_space=False),
    ])
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        min_frequency=MIN_FREQUENCY,
        special_tokens=SPECIAL_TOKENS,
    )

    # ── Train ────────────────────────────────────────────────────
    log.info("Training on full corpus (streaming via Rust) …")
    t0 = time.time()
    tokenizer.train(files=[str(CORPUS_PATH)], trainer=trainer)
    elapsed = time.time() - t0

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / "tokenizer.json"
    tokenizer.save(str(output_path))
    log.info("Saved to %s", output_path)
    log.info("Training took %.1f minutes", elapsed / 60)

    # ── Stats ────────────────────────────────────────────────────
    vocab = tokenizer.get_vocab()
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])

    log.info("")
    log.info("Tokenizer stats:")
    log.info("  Vocab size  : %d", len(vocab))
    log.info("  EOT token   : %s (ID %d)", EOT_TOKEN, vocab.get(EOT_TOKEN, -1))
    log.info("  Training    : %.1f min", elapsed / 60)

    # ── Verify custom tokens ─────────────────────────────────────
    log.info("")
    log.info("Verifying custom tokens …")
    missing = []
    for tok in SPECIAL_TOKENS:
        if tok not in vocab:
            missing.append(tok)
    if missing:
        log.warning("  MISSING %d custom tokens: %s", len(missing), missing[:10])
    else:
        log.info("  ✓ All %d custom tokens present in vocab", len(SPECIAL_TOKENS))

    # ── Show sample vocab ────────────────────────────────────────
    log.info("")
    log.info("Vocabulary samples (first 30 + last 10 tokens):")
    for token, tid in sorted_vocab[:30]:
        log.info("  %5d: %r", tid, token)
    log.info("  ...")
    for token, tid in sorted_vocab[-10:]:
        log.info("  %5d: %r", tid, token)

    # ── Show custom token IDs ────────────────────────────────────
    log.info("")
    log.info("Custom token IDs:")
    log.info("  %-20s → ID %d", repr(EOT_TOKEN), vocab.get(EOT_TOKEN, -1))
    for tok in INDENT_TOKENS:
        log.info("  %-20s → ID %d", repr(tok), vocab.get(tok, -1))
    log.info("  %-20s → ID %d", repr(DIGIT_TOKENS[0]), vocab.get(DIGIT_TOKENS[0], -1))
    log.info("  %-20s → ID %d", repr(DIGIT_TOKENS[-1]), vocab.get(DIGIT_TOKENS[-1], -1))
    log.info("  %-20s → ID %d", repr(SPACE_DIGIT_TOKENS[0]), vocab.get(SPACE_DIGIT_TOKENS[0], -1))
    log.info("  %-20s → ID %d", repr(SPACE_DIGIT_TOKENS[-1]), vocab.get(SPACE_DIGIT_TOKENS[-1], -1))

    # ── Quick round-trip tests ───────────────────────────────────
    log.info("")
    log.info("Quick tests:")
    tests = [
        "def fib(n):\n    return fib(n-1) + fib(n-2)",
        "x = 2024",
        "3.14159 * r**2",
        "    if x > 0:\n        return x\n    else:\n        return -x",
        "values = [00, 01, 42, 99]",
    ]
    for test in tests:
        ids = tokenizer.encode(test).ids
        roundtrip = tokenizer.decode(ids)
        log.info("  %-50s  %4d tokens  ok=%s",
                 repr(test[:50]) if len(test) > 50 else repr(test),
                 len(ids), "yes" if test == roundtrip else "no")

    log.info("")
    log.info("Done.")


if __name__ == "__main__":
    main()
