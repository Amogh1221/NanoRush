import re
import sys
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

LOG_PATH = "logs/training_log.txt"
OUTPUT_PATH = "logs/training_plot.png"


def parse_log(path):
    steps = []
    evals = []

    step_pat = re.compile(
        r"STEP\s+(\d+)/\d+\s+\|\s+loss=([\d.]+)"
    )
    eval_train_pat = re.compile(r"^\s{2}Train Loss\s+:\s+([\d.]+)")
    eval_val_pat = re.compile(r"^\s{2}Val Loss\s+:\s+([\d.]+)")
    eval_ema_val_pat = re.compile(r"^\s{2}EMA Val Loss\s+:\s+([\d.]+)")
    eval_step_pat = re.compile(r"EVALUATION @ Step (\d+)")

    with open(path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        m = step_pat.search(line)
        if m:
            steps.append((int(m.group(1)), float(m.group(2))))

    eval_step = None
    eval_train = None
    eval_val = None
    eval_ema_val = None
    for line in lines:
        m = eval_step_pat.search(line)
        if m:
            if eval_step is not None and eval_train is not None and eval_val is not None:
                evals.append((eval_step, eval_train, eval_val, eval_ema_val))
            eval_step = int(m.group(1))
            eval_train = None
            eval_val = None
            eval_ema_val = None
        m = eval_train_pat.search(line)
        if m:
            eval_train = float(m.group(1))
        m = eval_val_pat.search(line)
        if m:
            eval_val = float(m.group(1))
        m = eval_ema_val_pat.search(line)
        if m:
            eval_ema_val = float(m.group(1))
    if eval_step is not None and eval_train is not None and eval_val is not None:
        evals.append((eval_step, eval_train, eval_val, eval_ema_val))

    return steps, evals


def plot(steps, evals):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    step_nums = [s[0] for s in steps]
    step_losses = [s[1] for s in steps]

    ax1.plot(step_nums, step_losses, alpha=0.4, linewidth=0.8,
             color="gray", label="Step-level Loss")

    if evals:
        eval_nums = [e[0] for e in evals]
        train_losses = [e[1] for e in evals]
        val_losses = [e[2] for e in evals]
        ema_losses = [e[3] for e in evals]

        ax1.plot(eval_nums, train_losses, "o-", color="#2196F3",
                 linewidth=2, markersize=5, label="Eval Train Loss")
        ax1.plot(eval_nums, val_losses, "s-", color="#F44336",
                 linewidth=2, markersize=5, label="Eval Val Loss")
        if any(v is not None for v in ema_losses):
            ax1.plot(eval_nums, ema_losses, "d--", color="#FF9800",
                     linewidth=1.5, markersize=4, label="EMA Val Loss")

        for i, (x, y) in enumerate(zip(eval_nums, val_losses)):
            if i == 0:
                label = "Gap (Val - Train)"
            else:
                label = None
            gap = val_losses[i] - train_losses[i]
            ax1.vlines(x, train_losses[i], val_losses[i],
                       color="#9C27B0", alpha=0.5, linewidth=1.5,
                       label=label)

    ax1.set_ylabel("Loss")
    ax1.set_title("Training & Validation Loss — Overfitting Check")
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    if evals:
        gaps = [e[2] - e[1] for e in evals]
        ax2.bar([e[0] for e in evals], gaps, width=150,
                color="#9C27B0", alpha=0.7, label="Val - Train Gap")
        ax2.axhline(y=0, color="k", linewidth=0.5)
        ax2.set_xlabel("Step")
        ax2.set_ylabel("Loss Gap")
        ax2.set_title("Generalization Gap (Val Loss - Train Loss)")
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)

    ax1.set_xlim(left=0)
    if steps:
        ax1.set_xlim(right=steps[-1][0])
    if evals:
        ax2.set_xlim(ax1.get_xlim())

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=150)
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
        step, train_l, val_l, _ = e
        gap = val_l - train_l
        ppl = math.exp(min(val_l, 20))
        trend = " <<< WIDENING" if gap > 2 else ""
        print(f"{step:>7}  {train_l:>8.4f}  {val_l:>8.4f}  {gap:>+8.4f}  {ppl:>12.2f}{trend}")

    first_gap = evals[0][2] - evals[0][1]
    last_gap = evals[-1][2] - evals[-1][1]
    gap_change = last_gap - first_gap

    print(f"\n--- Summary ---")
    print(f"Initial gap (step {evals[0][0]}): {first_gap:+.4f}")
    print(f"Final gap   (step {evals[-1][0]}): {last_gap:+.4f}")
    print(f"Gap change  : {gap_change:+.4f}")
    if gap_change > 1:
        print("VERDICT: OVERFITTING — generalization gap is widening significantly.")
    elif gap_change > 0:
        print("CAUTION: Mild overfitting — gap is slowly widening.")
    else:
        print("OK: No clear signs of overfitting (gap is stable or shrinking).")


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
