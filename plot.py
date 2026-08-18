import re
import sys
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── Publication Quality Styling ──────────────────────────────────────────
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 16,
    "font.family": "serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
    "axes.grid": True,
})

LOG_PATH = "log.txt"
OUTPUT_PATH = "training_plot.png"

def parse_log(path):
    steps = []
    evals = []

    step_pat = re.compile(
        r"STEP\s+(\d+)/\d+\s+\|\s+loss=([\d.]+)\s+\|\s+lr=([\d.e+-]+)\s+\|\s+grad_norm=([\d.]+)"
    )
    eval_train_pat = re.compile(r"^\s{2}Train Loss\s+:\s+([\d.]+)")
    eval_val_pat = re.compile(r"^\s{2}Val Loss\s+:\s+([\d.]+)")
    eval_ema_val_pat = re.compile(r"^\s{2}EMA Val Loss\s+:\s+([\d.]+)")
    eval_ppl_pat = re.compile(r"^\s{2}Perplexity\s+:\s+([\d.]+)")
    eval_step_pat = re.compile(r"EVALUATION @ Step (\d+)")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        m = step_pat.search(line)
        if m:
            steps.append((int(m.group(1)), float(m.group(2)), float(m.group(3)), float(m.group(4))))

    eval_step = None
    eval_train = None
    eval_val = None
    eval_ema_val = None
    eval_ppl = None
    
    for line in lines:
        m = eval_step_pat.search(line)
        if m:
            if eval_step is not None and eval_train is not None and eval_val is not None:
                evals.append((eval_step, eval_train, eval_val, eval_ema_val, eval_ppl))
            eval_step = int(m.group(1))
            eval_train = None
            eval_val = None
            eval_ema_val = None
            eval_ppl = None
        m = eval_train_pat.search(line)
        if m:
            eval_train = float(m.group(1))
        m = eval_val_pat.search(line)
        if m:
            eval_val = float(m.group(1))
        m = eval_ema_val_pat.search(line)
        if m:
            eval_ema_val = float(m.group(1))
        m = eval_ppl_pat.search(line)
        if m:
            eval_ppl = float(m.group(1))
            
    if eval_step is not None and eval_train is not None and eval_val is not None:
        evals.append((eval_step, eval_train, eval_val, eval_ema_val, eval_ppl))

    return steps, evals

def plot(steps, evals):
    # Standard academic page proportions (narrower than a wide monitor display)
    fig, axs = plt.subplots(4, 1, figsize=(10, 14), sharex=True)
    ax1, ax2, ax3, ax4 = axs

    step_nums = [s[0] for s in steps]
    step_losses = [s[1] for s in steps]
    step_lrs = [s[2] for s in steps]
    step_gnorms = [s[3] for s in steps]

    # Panel 1: Loss
    ax1.plot(step_nums, step_losses, alpha=0.3, linewidth=0.8, color="gray", label="Raw Step Loss")
    
    # Add a moving average trendline for precision
    if len(step_losses) > 10:
        window_size = max(5, len(step_losses) // 50)
        smoothed = np.convolve(step_losses, np.ones(window_size)/window_size, mode='valid')
        ax1.plot(step_nums[window_size-1:], smoothed, color="black", linewidth=1.5, label=f"Trend (n={window_size})")

    if evals:
        eval_nums = [e[0] for e in evals]
        train_losses = [e[1] for e in evals]
        val_losses = [e[2] for e in evals]
        ema_losses = [e[3] for e in evals]
        ppls = [e[4] for e in evals]

        ax1.plot(eval_nums, train_losses, "o-", color="#2196F3", linewidth=2, markersize=5, label="Eval Train Loss")
        ax1.plot(eval_nums, val_losses, "s-", color="#F44336", linewidth=2, markersize=5, label="Eval Val Loss")
        if any(v is not None for v in ema_losses):
            ax1.plot(eval_nums, ema_losses, "d--", color="#FF9800", linewidth=1.5, markersize=4, label="EMA Val Loss")

    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Zoom in on the relevant loss values by ignoring the massive spikes in the first few steps
    if step_losses:
        sorted_losses = sorted(step_losses)
        # Cap the top of the graph at the 95th percentile + 10% padding
        y_max = sorted_losses[int(len(sorted_losses) * 0.95)] * 1.1
        ax1.set_ylim(top=y_max)

    # Panel 2: Perplexity
    if evals and any(p is not None for p in ppls):
        ax2.plot(eval_nums, ppls, "^-", color="#9C27B0", linewidth=2, markersize=6, label="Validation Perplexity")
        ax2.set_ylabel("Perplexity")
        ax2.set_title("Validation Perplexity (Log Scale)")
        ax2.set_yscale("log")
        ax2.legend(loc="upper right", fontsize=9)
        ax2.grid(True, alpha=0.3, which="both")
    else:
        ax2.text(0.5, 0.5, "No Perplexity Data", ha='center', va='center', transform=ax2.transAxes)

    # Panel 3: Learning Rate
    ax3.plot(step_nums, step_lrs, color="#009688", linewidth=1.5, label="Learning Rate")
    ax3.set_ylabel("Learning Rate")
    ax3.set_title("Learning Rate Schedule")
    ax3.ticklabel_format(axis="y", style="sci", scilimits=(0,0))
    ax3.legend(loc="upper right", fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Panel 4: Gradient Norm
    ax4.plot(step_nums, step_gnorms, color="#795548", alpha=0.7, linewidth=1, label="Gradient Norm")
    ax4.set_ylabel("Grad Norm")
    ax4.set_xlabel("Training Steps")
    ax4.set_title("Gradient Norm (Stability Check)")
    ax4.legend(loc="upper right", fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(bottom=0)

    ax1.set_xlim(left=0)
    if steps:
        ax1.set_xlim(right=steps[-1][0])

    fig.tight_layout()
    # Save at 300 DPI (standard for publications) and crop white space
    fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    print(f"Plot saved to {OUTPUT_PATH}")
    plt.close(fig)

def print_analysis(evals):
    if not evals:
        print("No evaluation data found.")
        return
    print("\n--- Overfitting Analysis ---")
    print(f"{'Step':>7}  {'Train':>8}  {'Val':>8}  {'Gap':>8}  {'Perplexity':>12}")
    print("-" * 50)
    for e in evals:
        step, train_l, val_l, ema_val_l, ppl = e
        gap = val_l - train_l
        ppl_str = f"{ppl:12.2f}" if ppl is not None else "         N/A"
        trend = " <<< WIDENING" if gap > 2 else ""
        print(f"{step:>7}  {train_l:>8.4f}  {val_l:>8.4f}  {gap:>+8.4f}  {ppl_str}{trend}")

def main():
    if not __import__("os").path.exists(LOG_PATH):
        print(f"Error: {LOG_PATH} not found.", file=sys.stderr)
        sys.exit(1)
    steps, evals = parse_log(LOG_PATH)
    if not steps:
        print("No step data found in log.")
        return
    plot(steps, evals)
    print_analysis(evals)

if __name__ == "__main__":
    main()
