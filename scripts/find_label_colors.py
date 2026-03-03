# find_label_colors.py
# One-time diagnostic to discover the true class colors in VMVI label PNGs.
#
# The GT label images have JPEG compression artifacts baked in, so the pixel
# colors don't exactly match the color table in the documentation. K-means
# clustering across all label files collapses the smeared colors back to their
# true centers.
#
# Already ran this — output is hardcoded into prepare_dataset.py as CLASS_COLORS_RGB.
# Keeping this here in case the dataset changes or colors need re-verification.
#
# Usage:
#   python scripts/find_label_colors.py \
#       --raw_dir /scratch/gilbreth/$USER/vivm/raw/10_4231_6WPT-RZ29 \
#       --n_classes 15

import argparse
import glob
import os
import sys

import cv2
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir",    required=True)
    parser.add_argument("--n_classes",  type=int, default=15)
    parser.add_argument("--ext",        default="png", choices=["png", "jpg"])
    parser.add_argument("--max_pixels", type=int, default=500_000)
    return parser.parse_args()


def find_label_files(raw_dir: str, ext: str) -> list:
    pattern = os.path.join(raw_dir, "**", f"*.label.{ext}")
    return sorted(glob.glob(pattern, recursive=True))


def collect_unique_colors(label_files: list, max_pixels: int) -> np.ndarray:
    all_pixels      = []
    pixels_per_file = max(1, max_pixels // len(label_files))

    print(f"Scanning {len(label_files)} label files ({pixels_per_file} px/file max)...")

    for i, fpath in enumerate(label_files):
        img = cv2.imread(fpath)
        if img is None:
            print(f"  [WARN] Could not read: {fpath}")
            continue
        pixels = img.reshape(-1, 3)
        if len(pixels) > pixels_per_file:
            idx    = np.random.choice(len(pixels), pixels_per_file, replace=False)
            pixels = pixels[idx]
        all_pixels.append(pixels)
        if (i + 1) % 20 == 0 or (i + 1) == len(label_files):
            print(f"  {i + 1}/{len(label_files)} files")

    if not all_pixels:
        print("[ERROR] No pixels collected. Check --raw_dir and --ext.")
        sys.exit(1)

    combined = np.vstack(all_pixels).astype(np.float32)
    unique   = np.unique(combined, axis=0)
    print(f"Total pixels: {len(combined):,}  |  Unique colors: {len(unique):,}")
    return unique


def cluster_colors(unique_colors: np.ndarray, n_classes: int) -> np.ndarray:
    from sklearn.cluster import KMeans
    print(f"\nRunning k-means with k={n_classes}...")
    kmeans  = KMeans(n_clusters=n_classes, random_state=42, n_init=20, max_iter=500)
    kmeans.fit(unique_colors)
    centers = np.clip(np.round(kmeans.cluster_centers_).astype(int), 0, 255)
    return centers[np.argsort(centers.sum(axis=1))]  # sort darkest first


def print_results(centers: np.ndarray):
    print("\n" + "=" * 60)
    print("DISCOVERED CLASS COLORS")
    print("=" * 60)
    for i, bgr in enumerate(centers):
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        print(f"  [{i:>2}]  BGR ({b:>3}, {g:>3}, {r:>3})  ->  RGB ({r:>3}, {g:>3}, {b:>3})")

    print("\nPaste into prepare_dataset.py as CLASS_COLORS_RGB:\n")
    print("CLASS_COLORS_RGB = np.array([")
    for i, bgr in enumerate(centers):
        b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
        print(f"    [{r:>3}, {g:>3}, {b:>3}],   # class {i}")
    print("], dtype=np.float32)")
    print("\nCross-reference class names with Table 1 in VMVI2025.4.pdf")


def main():
    args = parse_args()

    if not os.path.isdir(args.raw_dir):
        print(f"[ERROR] raw_dir not found: {args.raw_dir}")
        sys.exit(1)

    label_files = find_label_files(args.raw_dir, args.ext)
    if not label_files:
        print(f"[ERROR] No *.label.{args.ext} files found under {args.raw_dir}")
        sys.exit(1)

    unique_colors = collect_unique_colors(label_files, args.max_pixels)

    if len(unique_colors) < args.n_classes:
        print(f"[WARN] Only {len(unique_colors)} unique colors found, expected {args.n_classes}")
        args.n_classes = len(unique_colors)

    centers = cluster_colors(unique_colors, args.n_classes)
    print_results(centers)


if __name__ == "__main__":
    np.random.seed(42)
    main()