# VMVI Dataset Loader
# Loads the .npy arrays produced by prepare_dataset.py
#
# Each clip has:
#   input.npy  : (1800, 768, 4) float32  — MP RGB + vehicle width trace, normalized 0-1
#   label.npy  : (1800, 768)    int64    — class index per pixel
#
# At load time, a sliding T=256 window extracts patches.
# The label for each patch is the last row only (shift-mode, per paper).

import random
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from models.st2cn import NUM_CLASSES

T_PATCH = 256
W_PATCH = 768


class VMVIDataset(Dataset):
    def __init__(
        self,
        root:    str,
        split:   str  = "train",
        T:       int  = T_PATCH,
        stride:  int  = 1,
        augment: bool = True,
    ):
        self.root    = Path(root)
        self.split   = split
        self.T       = T
        self.stride  = stride
        self.augment = augment and (split == "train")

        split_file = self.root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        self.clips = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

        if not self.clips:
            raise ValueError(f"No clips found in {split_file}")

        self.index = self._build_index()

    def _build_index(self):
        # Build list of (clip_stem, end_row) for every valid window position
        index = []
        for stem in self.clips:
            label_path = self.root / stem / "label.npy"
            if not label_path.exists():
                print(f"  [WARN] Missing label.npy for {stem} — skipping")
                continue
            label  = np.load(label_path, mmap_mode="r")
            T_full = label.shape[0]  # 1800
            for end_row in range(self.T - 1, T_full, self.stride):
                index.append((stem, end_row))
        return index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        stem, end_row = self.index[idx]
        start_row     = end_row - self.T + 1

        input_full = np.load(self.root / stem / "input.npy", mmap_mode="r")  # (1800, 768, 4)
        label_full = np.load(self.root / stem / "label.npy", mmap_mode="r")  # (1800, 768)

        if start_row < 0:
            # Pad the front with zeros if window extends before clip start
            pad       = -start_row
            inp_patch = np.concatenate([
                np.zeros((pad, input_full.shape[1], 4), dtype=np.float32),
                np.array(input_full[0 : end_row + 1]),
            ], axis=0)
        else:
            inp_patch = np.array(input_full[start_row : end_row + 1])  # (T, 768, 4)

        target = np.array(label_full[end_row])  # (768,) — last row only

        if self.augment and random.random() < 0.5:
            inp_patch = inp_patch[:, ::-1, :].copy()
            target    = target[::-1].copy()

        # (T, W, 4) -> (4, T, W) for Conv2d
        x = inp_patch.transpose(2, 0, 1).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(target.astype(np.int64))


def get_dataloaders(
    root:         str,
    batch_size:   int = 4,
    num_workers:  int = 4,
    stride_train: int = 1,
    stride_val:   int = 30,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = VMVIDataset(root, split="train", stride=stride_train, augment=True)
    val_ds   = VMVIDataset(root, split="val",   stride=stride_val,   augment=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds):,} patches from {len(train_ds.clips)} clips")
    print(f"Val:   {len(val_ds):,} patches from {len(val_ds.clips)} clips")

    return train_loader, val_loader


def compute_class_weights(root: str, num_classes: int = NUM_CLASSES, max_clips: int = 50) -> torch.Tensor:
    # Inverse-frequency weights for handling class imbalance
    counts    = np.zeros(num_classes, dtype=np.int64)
    clips     = [l.strip() for l in (Path(root) / "train.txt").read_text().splitlines()
                 if l.strip()][:max_clips]

    for stem in clips:
        label_path = Path(root) / stem / "label.npy"
        if not label_path.exists():
            continue
        gt = np.load(label_path, mmap_mode="r")
        for c in range(num_classes):
            counts[c] += int((gt == c).sum())

    counts  = np.maximum(counts.astype(np.float64), 1)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python data/dataset.py <prepared_root>")
        sys.exit(1)

    train_loader, val_loader = get_dataloaders(sys.argv[1], batch_size=2,
                                               num_workers=0, stride_train=300, stride_val=300)
    x, y = next(iter(train_loader))
    print(f"x: {x.shape}  (expect [2, 4, 256, 768])")
    print(f"y: {y.shape}  (expect [2, 768])")
    print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"y range: [{y.min()}, {y.max()}]")
    print("Sanity check passed")