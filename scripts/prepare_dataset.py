# prepare_dataset.py
# Converts the raw VMVI 2025 dataset into .npy arrays for ST2CN training.
#
# Per Lin & Zheng (2023) Fig. 8, each clip has three components:
#   MP(x,t) : motion profile   — 1800x2592 RGB  -> resized to 1800x768
#   WP(x,t) : vehicle width    — 1800x2592 gray -> resized to 1800x768
#   GT(x,t) : ground truth     — 1800x2592 RGB  -> resized to 1800x768, mapped to class indices
#
# Output per clip:
#   input.npy  : (1800, 768, 4) float32  — MP RGB + WP grayscale, normalized 0-1
#   label.npy  : (1800, 768)    int64    — class indices
#
# Training patches are extracted in dataset.py at load time, not here.
#
# NOTE: VMVI 2025 has 15 classes vs 13 in the paper. New: off_road (OR).
# Stopping and turning_away are separate here; paper combined them as "ego stop/turning".
#
# GT labels are PNG but have JPEG compression artifacts from the annotation pipeline,
# so exact color matching fails. We use nearest-color L2 matching instead.
#
# Usage:
#   python scripts/prepare_dataset.py \
#     --raw_dir /scratch/gilbreth/$USER/vivm/raw/10_4231_6WPT-RZ29 \
#     --out_dir /scratch/gilbreth/$USER/vivm/prepared
#
# Also extracted a zip bomb false positive during dataset extraction — resolved with:
#   UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o data.zip
# (large datasets with overlapping zip entries trigger this warning incorrectly)

import argparse
import glob
import os
import sys
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

# Class definitions from VMVI 2025 Table 1
CLASS_NAMES = [
    "CI",   #  0  Cut-in                  (255, 140,   0)
    "CL",   #  1  Lane-changing           (255, 165,   0)
    "FA",   #  2  Front approaching       (255,   0,   0)
    "FL",   #  3  Front leaving           (240, 128, 128)
    "FF",   #  4  Front following         (255,  99,  71)
    "PN",   #  5  Parallel next lane      (255, 255,   0)
    "PS",   #  6  Passing                 (255, 215,   0)
    "PD",   #  7  Being passed            (218, 165,  32)
    "M",    #  8  Merging                 ( 60, 179, 113)
    "O",    #  9  Opposite                (128,   0, 128)
    "C",    # 10  Crossing                (  0,   0, 255)
    "TW",   # 11  Turning away            (  0, 255, 255)
    "OR",   # 12  Off-road                (255, 255, 255)
    "ST",   # 13  Stopping                (128, 128, 128)
    "BG",   # 14  Background              (  0,   0,   0)
]

CLASS_COLORS_RGB = np.array([
    [255, 140,   0],
    [255, 165,   0],
    [255,   0,   0],
    [240, 128, 128],
    [255,  99,  71],
    [255, 255,   0],
    [255, 215,   0],
    [218, 165,  32],
    [ 60, 179, 113],
    [128,   0, 128],
    [  0,   0, 255],
    [  0, 255, 255],
    [255, 255, 255],
    [128, 128, 128],
    [  0,   0,   0],
], dtype=np.float32)

NUM_CLASSES = len(CLASS_NAMES)  # 15


def color_gt_to_label(img_bgr: np.ndarray) -> np.ndarray:
    # Nearest-color L2 matching — needed because GT PNGs have JPEG artifacts
    # baked in from an earlier stage of the annotation pipeline, so exact
    # color lookup fails near class boundaries.
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    H, W, _ = img_rgb.shape
    pixels  = img_rgb.reshape(-1, 3)
    dists   = np.sum((pixels[:, None, :] - CLASS_COLORS_RGB[None, :, :]) ** 2, axis=2)
    return dists.argmin(axis=1).reshape(H, W).astype(np.int64)


def mp_to_stem(mp_path: str) -> str:
    # MP filenames use parentheses: "{class} ({n})_m.png" -> stem "{class} {n}"
    name = os.path.basename(mp_path).replace("_m.png", "")
    name = name.replace("(", "").replace(")", "")
    return " ".join(name.split())


def gt_to_stem(gt_path: str) -> str:
    # GT filenames use spaces: "{class} {n}_g_refined_add.png" -> stem "{class} {n}"
    name = os.path.basename(gt_path).replace("_g_refined_add.png", "")
    return " ".join(name.split())


def build_pairs(raw_dir: str) -> list:
    mp_root = os.path.join(raw_dir, "MPTV800", "AutomaticLabelSource800")
    gt_root = os.path.join(raw_dir, "Label400")

    mp_files = sorted(glob.glob(os.path.join(mp_root, "*_m.png")))
    w_files  = sorted(glob.glob(os.path.join(mp_root, "*_w.png")))
    gt_files = sorted(glob.glob(os.path.join(gt_root, "*_g_refined_add.png")))

    mp_map = {mp_to_stem(f): f for f in mp_files}
    w_map  = {mp_to_stem(f.replace("_w.png", "_m.png")): f for f in w_files}
    gt_map = {gt_to_stem(f): f for f in gt_files}

    valid_stems = set(mp_map) & set(w_map) & set(gt_map)
    pairs = [{"stem": s, "mp": mp_map[s], "w": w_map[s], "gt": gt_map[s]}
             for s in sorted(valid_stems)]

    skipped = len(mp_map) - len(valid_stems)
    if skipped:
        print(f"  Skipped {skipped} clips with no matching GT or W file")

    return pairs


def process_clip(args: tuple) -> Optional[str]:
    pair, out_dir, target_width = args
    stem       = pair["stem"]
    out        = os.path.join(out_dir, stem)
    inp_path   = os.path.join(out, "input.npy")
    label_path = os.path.join(out, "label.npy")

    if os.path.exists(inp_path) and os.path.exists(label_path):
        return stem  # already processed

    os.makedirs(out, exist_ok=True)

    mp_bgr = cv2.imread(pair["mp"])
    w_bgr  = cv2.imread(pair["w"])
    gt_bgr = cv2.imread(pair["gt"])

    if mp_bgr is None or w_bgr is None or gt_bgr is None:
        print(f"  [WARN] Could not read files for: {stem}")
        return None

    H = mp_bgr.shape[0]  # 1800 — keep full temporal height
    mp_resized = cv2.resize(mp_bgr, (target_width, H), interpolation=cv2.INTER_LINEAR)
    w_resized  = cv2.resize(w_bgr,  (target_width, H), interpolation=cv2.INTER_LINEAR)
    gt_resized = cv2.resize(gt_bgr, (target_width, H), interpolation=cv2.INTER_NEAREST)
    # INTER_NEAREST for GT — must not blend class colors

    mp_rgb    = cv2.cvtColor(mp_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    w_gray    = cv2.cvtColor(w_resized,  cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    w_gray    = w_gray[:, :, np.newaxis]
    input_4ch = np.concatenate([mp_rgb, w_gray], axis=2)  # (H, W, 4)

    label = color_gt_to_label(gt_resized)  # (H, W) int64

    np.save(inp_path,   input_4ch)
    np.save(label_path, label)
    return stem


def write_splits(stems: list, out_dir: str, train_frac: float):
    np.random.seed(42)
    idx        = np.random.permutation(len(stems))
    n_train    = int(len(stems) * train_frac)
    train_stems = [stems[i] for i in idx[:n_train]]
    val_stems   = [stems[i] for i in idx[n_train:]]

    with open(os.path.join(out_dir, "train.txt"), "w") as f:
        f.write("\n".join(train_stems) + "\n")
    with open(os.path.join(out_dir, "val.txt"), "w") as f:
        f.write("\n".join(val_stems) + "\n")

    print(f"  Train clips: {len(train_stems)}")
    print(f"  Val   clips: {len(val_stems)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare VMVI 2025 dataset for ST2CN training.")
    parser.add_argument("--raw_dir", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--width",   type=int,   default=768)
    parser.add_argument("--workers", type=int,   default=4)
    parser.add_argument("--split",   type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"[ERROR] raw_dir not found: {args.raw_dir}")
        sys.exit(1)

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nVMVI 2025 -> ST2CN Preprocessing")
    print(f"  raw_dir : {args.raw_dir}")
    print(f"  out_dir : {args.out_dir}")
    print(f"  width   : {args.width}  ({NUM_CLASSES} classes)")

    print("\nBuilding file pairs...")
    pairs = build_pairs(args.raw_dir)
    print(f"  Found {len(pairs)} matched clips")

    if not pairs:
        print("[ERROR] No matched clips. Check --raw_dir path.")
        sys.exit(1)

    print(f"\nProcessing with {args.workers} workers...")
    job_args = [(p, args.out_dir, args.width) for p in pairs]

    if args.workers > 1:
        with Pool(args.workers) as pool:
            results = []
            for i, stem in enumerate(pool.imap_unordered(process_clip, job_args), 1):
                results.append(stem)
                if i % 20 == 0 or i == len(pairs):
                    print(f"  {i}/{len(pairs)}  ({sum(1 for r in results if r)} succeeded)")
    else:
        results = [process_clip(j) for j in job_args]

    succeeded = [r for r in results if r is not None]
    print(f"\nCompleted: {len(succeeded)}/{len(pairs)} clips")

    print("\nWriting train/val splits...")
    write_splits(succeeded, args.out_dir, args.split)

    meta_path = os.path.join(args.out_dir, "classes.txt")
    with open(meta_path, "w") as f:
        for i, name in enumerate(CLASS_NAMES):
            r, g, b = CLASS_COLORS_RGB[i].astype(int)
            f.write(f"{i:2d}  {name:<6}  RGB ({r:3d}, {g:3d}, {b:3d})\n")

    print(f"\nDone. Class map -> {meta_path}")
    print("Next: run train.py")


if __name__ == "__main__":
    main()