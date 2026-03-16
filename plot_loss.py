"""
Plot training loss curve from history.json.

Usage:
    python plot_loss.py \
        --history /scratch/gilbreth/$USER/runs/st2cn_baseline_20260228_180438/history.json \
        --label   "Baseline (stride=1)" \
        --out     ~/capstone/eval_output/baseline/loss_curve.png

Multiple runs on one plot:
    python plot_loss.py \
        --history /scratch/gilbreth/$USER/runs/st2cn_baseline_20260228_180438/history.json \
                  /scratch/gilbreth/$USER/runs/st2cn_asym_20260308_130303/history.json \
        --label   "Baseline (stride=1)" "PSPNet (stride=1, in progress)" \
        --out     ~/capstone/eval_output/combined_loss.png
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--history", nargs="+", required=True,
                   help="Path(s) to history.json file(s)")
    p.add_argument("--label", nargs="+", default=None,
                   help="Legend label for each run (same order as --history)")
    p.add_argument("--out", type=str, default="loss_curve.png",
                   help="Output image path")
    return p.parse_args()


def main():
    args = parse_args()
    labels = args.label or [Path(h).parent.name for h in args.history]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#0D9488", "#1A2744", "#D97706", "#DC2626", "#5B21B6"]

    for i, (path, label) in enumerate(zip(args.history, labels)):
        with open(path) as f:
            history = json.load(f)

        epochs   = [h["epoch"]      for h in history]
        tr_loss  = [h["train_loss"] for h in history]
        val_loss = [h["val_loss"]   for h in history]
        miou     = [h["miou"]       for h in history]

        color = colors[i % len(colors)]

        axes[0].plot(epochs, tr_loss,  color=color, lw=2,   label=f"{label} — train")
        axes[0].plot(epochs, val_loss, color=color, lw=2, linestyle="--", label=f"{label} — val")
        axes[1].plot(epochs, miou,     color=color, lw=2,   label=label)

    # Loss plot
    axes[0].set_title("Training & Validation Loss", fontsize=13, fontweight="bold", pad=10)
    axes[0].set_xlabel("Epoch", fontsize=11)
    axes[0].set_ylabel("Cross-Entropy Loss", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[0].spines[["top", "right"]].set_visible(False)

    # mIoU plot (patch-mode from training loop)
    axes[1].set_title("Validation mIoU (Patch-Mode)", fontsize=13, fontweight="bold", pad=10)
    axes[1].set_xlabel("Epoch", fontsize=11)
    axes[1].set_ylabel("mIoU", fontsize=11)
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", color="#E5E7EB", linewidth=0.8)
    axes[1].spines[["top", "right"]].set_visible(False)

    # Annotation on mIoU plot noting shift-mode is the true metric
    axes[1].text(0.98, 0.05,
                 "Note: patch-mode val only.\nShift-mode mIoU (stride=1 baseline) = 0.806",
                 transform=axes[1].transAxes, fontsize=8, color="#6B7280",
                 ha="right", va="bottom", style="italic")

    plt.tight_layout(pad=2.0)
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()