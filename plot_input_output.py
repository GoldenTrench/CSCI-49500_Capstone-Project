"""
Visualize a motion profile input alongside the model's predicted output.

Usage:
    python plot_input_output.py \
        --data_dir  /scratch/gilbreth/$USER/vivm/prepared/ \
        --model_path /scratch/gilbreth/$USER/runs/st2cn_baseline_20260228_180438/best_model.pth \
        --clip_name "cutting_in 1" \
        --out       ~/capstone/eval_output/baseline/input_output.png

    # By index:
    python plot_input_output.py ... --clip_idx 0

    # Model variants: baseline (default), asym, psp
    python plot_input_output.py ... --model baseline
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

CLASSES = [
    "CI (cut_in)", "CL (lane_change)", "FA (front_approach)", "FL (front_leaving)",
    "FF (front_following)", "PN (parallel_next)", "PS (passing)", "PD (being_passed)",
    "M (merging)", "O (opposite)", "C (crossing)", "TW (turning_away)",
    "OR (off_road)", "ST (stopping)", "BG (background)",
]
NUM_CLASSES = len(CLASSES)
COLORS = [
    "#FF8C00", "#FFA500", "#FF0000", "#F08080", "#FF6347",
    "#FFFF00", "#FFD700", "#DAA520", "#3CB371", "#800080",
    "#0000FF", "#00FFFF", "#DDDDDD", "#808080", "#111111",
]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--clip_idx", type=int, default=0)
    p.add_argument("--clip_name", type=str, default=None)
    p.add_argument("--model", choices=["baseline", "asym", "psp"], default="baseline")
    p.add_argument("--out", type=str, default="input_output.png")
    return p.parse_args()

def find_clip_dir(data_dir, clip_idx, clip_name):
    root = Path(data_dir)
    clips = sorted([d for d in root.iterdir() if d.is_dir() and (d / "input.npy").exists()])
    if not clips:
        raise FileNotFoundError(f"No clip folders found under {root}")
    print(f"Found {len(clips)} clips")
    if clip_name:
        matches = [c for c in clips if clip_name.lower() in c.name.lower()]
        if not matches:
            raise FileNotFoundError(f"No clip matching '{clip_name}'")
        print(f"Matched: {matches[0].name}")
        return matches[0]
    return clips[clip_idx]

def load_model(model_type, model_path, device):
    sys.path.insert(0, str(Path(__file__).parent))
    if model_type == "baseline":
        from models.st2cn import ST2CN
        model = ST2CN(num_classes=NUM_CLASSES)
    elif model_type == "asym":
        from models.st2cn_asym import ST2CN_Asym
        model = ST2CN_Asym(num_classes=NUM_CLASSES)
    else:
        from models.st2cn_psp import ST2CN_PSP
        model = ST2CN_PSP(num_classes=NUM_CLASSES)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt.get("model_state_dict", ckpt.get("model", ckpt))
    model.load_state_dict(state)
    model.eval()
    return model.to(device)

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clip_dir = find_clip_dir(args.data_dir, args.clip_idx, args.clip_name)
    inp_np = np.load(clip_dir / "input.npy").astype(np.float32)
    lbl_np = np.load(clip_dir / "label.npy")
    print(f"  input: {inp_np.shape}  label: {lbl_np.shape}")

    # Normalise to [C, T, W]
    if inp_np.shape[-1] <= 8:        # [T, W, C]
        inp_np = inp_np.transpose(2, 0, 1)
    elif inp_np.shape[0] <= 8:       # [C, T, W] already
        pass
    else:
        inp_np = inp_np.transpose(2, 1, 0)
    C, T, W = inp_np.shape
    print(f"  normalised [C,T,W]: [{C},{T},{W}]")

    window = 256
    if T < window:
        pad = np.zeros((C, window - T, W), dtype=np.float32)
        inp_np = np.concatenate([inp_np, pad], axis=1)
        T = window

    model = load_model(args.model, args.model_path, device)
    T_eff = T - window + 1
    print(f"Running shift-mode inference ({T_eff} steps)...")

    preds = []
    with torch.no_grad():
        for row in range(T_eff):
            patch = inp_np[:, row:row + window, :]
            x = torch.from_numpy(patch).unsqueeze(0).to(device)
            out = model(x)
            pred = out.argmax(dim=1).squeeze(0).squeeze(0)
            preds.append(pred.cpu().numpy())

    pred_map = np.stack(preds, axis=0)   # [T_eff, W]

    # Ground truth: use the full 2D label map aligned to T_eff
    gt_map = np.array(lbl_np)[:T_eff, :]   # [T_eff, W]

    cmap = ListedColormap(COLORS)
    fig, axes = plt.subplots(1, 4, figsize=(20, 6),
                             gridspec_kw={"width_ratios": [4, 4, 4, 0.35]})
    fig.patch.set_facecolor("#F5F7FA")

    axes[0].imshow(inp_np[0, :T_eff], aspect="auto", cmap="viridis", origin="upper")
    axes[0].set_title("Input — Motion Profile\n(velocity channel)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Spatial (columns)"); axes[0].set_ylabel("Time (rows)")
    axes[0].spines[["top","right"]].set_visible(False)

    axes[1].imshow(pred_map, aspect="auto", cmap=cmap, vmin=0, vmax=NUM_CLASSES-1, origin="upper")
    axes[1].set_title("Predicted Labels\n(baseline model)", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Spatial (columns)"); axes[1].set_ylabel("Time (rows)")
    axes[1].spines[["top","right"]].set_visible(False)

    im = axes[2].imshow(gt_map, aspect="auto", cmap=cmap, vmin=0, vmax=NUM_CLASSES-1, origin="upper")
    axes[2].set_title("Ground Truth Labels", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Spatial (columns)"); axes[2].set_ylabel("Time (rows)")
    axes[2].spines[["top","right"]].set_visible(False)

    cb = fig.colorbar(im, cax=axes[3], ticks=list(range(NUM_CLASSES)))
    cb.ax.set_yticklabels(CLASSES, fontsize=8)

    fig.suptitle(f"Clip: {clip_dir.name}  |  Model: {args.model}", fontsize=12, color="#374151")
    plt.tight_layout()
    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved → {out}")

if __name__ == "__main__":
    main()