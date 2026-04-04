# Shift-mode evaluation for ST2CN checkpoints
#
# The paper explicitly uses shift-mode inference: the window slides one row at
# a time and always predicts the current (last) row. This is different from
# patch-mode, which evaluates at non-overlapping intervals (stride=30 in the
# training loop's val pass).
#
# This script runs stride=1 evaluation on the val split for any saved checkpoint.
# Use it to compare patch-mode numbers from training logs against shift-mode.
#
# Usage:
#   python evaluate.py \
#       --checkpoint /scratch/gilbreth/$USER/runs/<run>/best_model.pth \
#       --data_root  /scratch/gilbreth/$USER/vivm/prepared \
#       --model      baseline          # or: asym, psp, esp
#       --extra_channels angle         # or: gradient, or both

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

# Allow running from ~/capstone root
sys.path.insert(0, str(Path(__file__).parent))

from data.dataset    import VMVIDataset
from models.st2cn    import ST2CN, NUM_CLASSES, INTERACTION_CLASSES
from utils.metrics   import SegmentationMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Shift-mode evaluation")
    p.add_argument("--checkpoint",     required=True,  help="Path to best_model.pth")
    p.add_argument("--data_root",      required=True,  help="Path to prepared dataset root")
    p.add_argument("--model",          default="baseline",
                   choices=["baseline", "asym", "psp", "esp"],
                   help="Which model architecture was used")
    p.add_argument("--extra_channels", nargs="*", default=[],
                   choices=["angle", "gradient"],
                   help="Extra input channels used during training")
    p.add_argument("--batch_size",     type=int, default=16)
    p.add_argument("--num_workers",    type=int, default=8)
    p.add_argument("--base_filters",   type=int, default=16)
    p.add_argument("--output_dir",     type=str, default=None,
                   help="Directory to save confusion matrix image (optional)")
    return p.parse_args()


def load_model(args, device):
    in_ch = 4 + len(args.extra_channels)

    if args.model == "asym":
        from models.st2cn_asym import ST2CN_Asym
        model = ST2CN_Asym(in_channels=in_ch, num_classes=NUM_CLASSES,
                           base_filters=args.base_filters)
    elif args.model == "psp":
        from models.st2cn_psp import ST2CN_PSP
        model = ST2CN_PSP(in_channels=in_ch, num_classes=NUM_CLASSES,
                          base_filters=args.base_filters)
    elif args.model == "esp":
        from models.st2cn_esp import ST2CN_ESP
        model = ST2CN_ESP(in_channels=in_ch, num_classes=NUM_CLASSES,
                          base_filters=args.base_filters)
    else:
        model = ST2CN(in_channels=in_ch, num_classes=NUM_CLASSES,
                      base_filters=args.base_filters)

    ckpt = torch.load(args.checkpoint, map_location=device)

    # Handle DataParallel-wrapped checkpoints
    state = ckpt.get("model", ckpt)
    if any(k.startswith("module.") for k in state):
        state = {k[len("module."):]: v for k, v in state.items()}

    model.load_state_dict(state)
    model.to(device)
    model.eval()

    epoch = ckpt.get("epoch", "?")
    print(f"Loaded checkpoint (epoch {epoch}): {args.checkpoint}")
    return model


def run_shiftmode_eval(model, data_root, batch_size, num_workers, device,
                       extra_channels=None):
    # Returns (results_dict, metrics_obj) so caller can use print_table
    # stride=1 is shift-mode: every row gets a prediction
    val_ds = VMVIDataset(data_root, split="val", stride=1, augment=False,
                         extra_channels=extra_channels)
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    n_windows = len(val_ds)
    n_clips   = len(val_ds.clips)
    print(f"Val set (shift-mode, stride=1): {n_windows:,} windows from {n_clips} clips  "
          f"({val_ds.in_channels}ch)")
    print()

    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)               # [B, C, 1, W]
            preds  = logits.squeeze(2).argmax(dim=1)   # [B, W]
            metrics.update(preds, y)

            if (i + 1) % 200 == 0:
                pct = 100.0 * (i + 1) / len(val_loader)
                print(f"  [{i+1:5d}/{len(val_loader)}]  {pct:.1f}%", flush=True)

    return metrics.compute(), metrics


def print_results(results, metrics_obj):
    print("=" * 70)
    print("Shift-Mode Evaluation Results")
    print("=" * 70)
    metrics_obj.print_table(results)
    print("=" * 70)
    print(f"\nSummary:  mIoU = {results['mean_iou']:.4f}   mF1 = {results['mean_f1']:.4f}")


def save_confusion_matrix(results, output_dir):
    if not HAS_MPL:
        print("matplotlib not available — skipping confusion matrix plot")
        return

    cm    = results["confusion_matrix"].astype(np.float64)
    names = [c.replace("_", "\n") for c in INTERACTION_CLASSES]

    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    cm_norm  = cm / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm_norm, interpolation="nearest", cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(NUM_CLASSES))
    ax.set_yticks(range(NUM_CLASSES))
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title("Confusion Matrix (row-normalized)\nShift-Mode Evaluation", fontsize=13)

    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            val = cm_norm[i, j]
            if val >= 0.05:
                color = "white" if val > 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=6, color=color)

    plt.tight_layout()
    out_path = Path(output_dir) / "confusion_matrix.png"
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"\nConfusion matrix saved to: {out_path}")


def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    model   = load_model(args, device)
    results, metrics_obj = run_shiftmode_eval(
        model, args.data_root, args.batch_size, args.num_workers, device,
        extra_channels=args.extra_channels,
    )
    print_results(results, metrics_obj)

    if args.output_dir:
        save_confusion_matrix(results, args.output_dir)


if __name__ == "__main__":
    main()