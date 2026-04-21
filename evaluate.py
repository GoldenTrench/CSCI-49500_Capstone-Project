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

from data.dataset    import VMVIDataset, _load_png_channel
from models.st2cn    import ST2CN, NUM_CLASSES, INTERACTION_CLASSES
from utils.metrics   import SegmentationMetrics, compute_tiou_from_traces


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


def load_clip_input(data_root, stem, extra_channels):
    """Load input.npy and any extra channel PNGs for a clip, return (1800, 768, C) float32."""
    input_full = np.load(Path(data_root) / stem / "input.npy", mmap_mode="r")  # (1800, 768, 4)
    if not extra_channels:
        return np.array(input_full, dtype=np.float32)

    parts = [np.array(input_full, dtype=np.float32)]
    clip_dir = Path(data_root) / stem
    for ch in extra_channels:
        folder_name = "angle_mp" if ch == "angle" else "gradient"
        full_ch = _load_png_channel(clip_dir / folder_name).astype(np.float32)  # (1800, 768)
        if ch == "angle":
            full_ch = (full_ch - 127.5) / 127.5
        else:
            full_ch = full_ch / 255.0
        parts.append(full_ch[:, :, np.newaxis])  # (1800, 768, 1)
    return np.concatenate(parts, axis=2)  # (1800, 768, 4+N)


def run_shiftmode_eval(model, data_root, batch_size, num_workers, device,
                       extra_channels=None):
    extra_channels = extra_channels or []
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

    # ── Frame counts per class ────────────────────────────────────────────
    print("Counting frames per class in val split...")
    frame_counts = np.zeros(NUM_CLASSES, dtype=np.int64)
    for stem in val_ds.clips:
        label = np.load(Path(data_root) / stem / "label.npy", mmap_mode="r")
        for c in range(NUM_CLASSES):
            frame_counts[c] += int((label == c).any(axis=1).sum())
    print("Val frame counts per class:")
    for c, name in enumerate(INTERACTION_CLASSES):
        print(f"  {name:20s}: {frame_counts[c]:,}")
    print()

    # ── Main eval loop (ST-IoU / pixel-wise) ─────────────────────────────
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)

    with torch.no_grad():
        for i, (x, y) in enumerate(val_loader):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(x)
            preds  = logits.squeeze(2).argmax(dim=1)
            metrics.update(preds, y)

            if (i + 1) % 200 == 0:
                pct = 100.0 * (i + 1) / len(val_loader)
                print(f"  [{i+1:5d}/{len(val_loader)}]  {pct:.1f}%", flush=True)

    results = metrics.compute()

    # ── Approximate T-IoU (centre-column method, clip by clip) ───────────
    print("\nComputing approximate T-IoU (centre-column method)...")
    T = 256
    W = 768
    tiou_accum = {name: [] for name in INTERACTION_CLASSES[:-1]}

    with torch.no_grad():
        for clip_idx, stem in enumerate(val_ds.clips):
            # Load input with extra channels
            input_full = load_clip_input(data_root, stem, extra_channels)  # (1800, 768, C)
            label_full = np.load(Path(data_root) / stem / "label.npy", mmap_mode="r")
            T_full     = input_full.shape[0]
            C          = input_full.shape[2]

            pred_map = np.zeros((T_full, W), dtype=np.int64)

            windows, rows = [], []
            for end_row in range(T - 1, T_full):
                start_row = end_row - T + 1
                if start_row < 0:
                    pad   = -start_row
                    patch = np.concatenate([
                        np.zeros((pad, W, C), dtype=np.float32),
                        input_full[0:end_row + 1],
                    ], axis=0)
                else:
                    patch = input_full[start_row:end_row + 1]
                windows.append(patch.transpose(2, 0, 1))  # [C, T, W]
                rows.append(end_row)

                if len(windows) == batch_size or end_row == T_full - 1:
                    x_batch = torch.tensor(np.stack(windows),
                                           dtype=torch.float32).to(device)
                    logits  = model(x_batch)
                    preds   = logits.squeeze(2).argmax(dim=1).cpu().numpy()
                    for pred_row, end_r in zip(preds, rows):
                        pred_map[end_r] = pred_row
                    windows, rows = [], []

            clip_tiou = compute_tiou_from_traces(
                pred_map, np.array(label_full), NUM_CLASSES
            )
            for name in INTERACTION_CLASSES[:-1]:
                tiou_accum[name].append(clip_tiou.get(name, 0.0))

            if (clip_idx + 1) % 10 == 0:
                print(f"  T-IoU: {clip_idx+1}/{n_clips} clips done", flush=True)

    tiou_results = {name: float(np.mean(vals))
                    for name, vals in tiou_accum.items()}
    tiou_results["mean"] = float(np.mean(list(tiou_results.values())))

    print("\nApproximate T-IoU per class:")
    for name in INTERACTION_CLASSES[:-1]:
        print(f"  {name:20s}: {tiou_results[name]:.3f}")
    print(f"  {'Mean (no bg)':20s}: {tiou_results['mean']:.3f}")

    results["tiou"]         = tiou_results
    results["frame_counts"] = {INTERACTION_CLASSES[c]: int(frame_counts[c])
                                for c in range(NUM_CLASSES)}
    return results, metrics


def print_results(results, metrics_obj):
    print("=" * 70)
    print("Shift-Mode Evaluation Results")
    print("=" * 70)
    metrics_obj.print_table(results)
    print("=" * 70)
    print(f"\nSummary:  mIoU = {results['mean_iou']:.4f}   mF1 = {results['mean_f1']:.4f}")
    if "tiou" in results:
        print(f"          T-IoU (approx) = {results['tiou']['mean']:.4f}")


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
        import json
        out_path = Path(args.output_dir) / "eval_results.json"
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        results_save = {k: v for k, v in results.items()
                        if k != "confusion_matrix"}
        out_path.write_text(json.dumps(results_save, indent=2))
        print(f"Results saved to: {out_path}")


if __name__ == "__main__":
    main()