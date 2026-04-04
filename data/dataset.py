# VMVI Dataset Loader
# Loads the .npy arrays produced by prepare_dataset.py
#
# Each clip has:
#   input.npy       : (1800, 768, 4) float32  — MP RGB + vehicle width trace, normalized 0-1
#   label.npy       : (1800, 768)    int64    — class index per pixel
#   angle_mp/<f>.png: (1800, 2592)   uint8    — trajectory orientation, encoded (angle+90)/180*255
#   gradient/<f>.png: (1800, 2592)   uint8    — high-contrast trace points, sparse
#
# At load time, a sliding T=256 window extracts patches.
# The label for each patch is the last row only (shift-mode, per paper).
#
# Ablation channels:
#   extra_channels=[]                  → 4ch baseline (default)
#   extra_channels=["angle"]           → 5ch
#   extra_channels=["gradient"]        → 5ch
#   extra_channels=["angle","gradient"]→ 6ch
#
# Angle normalization : (pixel - 127.5) / 127.5  → [-1, 1]
#   Encoding: (angle_deg + 90) / 180 * 255, so 127.5 = 0 degrees
# Gradient normalization: pixel / 255.0  → [0, 1]
#   Sparse channel (mean ~7/255), mostly zero

import random
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).parent.parent))
from models.st2cn import NUM_CLASSES

T_PATCH = 256
W_PATCH = 768

# Target spatial size after resizing PNGs (matches input.npy width)
_PNG_TARGET_H = 1800
_PNG_TARGET_W = 768


def _load_png_channel(folder: Path, target_h: int = _PNG_TARGET_H,
                      target_w: int = _PNG_TARGET_W) -> np.ndarray:
    """
    Load a single PNG from `folder`, resize to (target_h, target_w),
    return as float32 array shape (target_h, target_w).

    The folder contains exactly one PNG per clip (e.g. angle_mp/ or gradient/).
    We grab whichever file is there rather than hard-coding names.
    """
    from PIL import Image

    pngs = list(folder.glob("*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG found in {folder}")

    img = Image.open(pngs[0]).convert("L")   # grayscale uint8
    if img.size != (target_w, target_h):     # PIL size = (W, H)
        img = img.resize((target_w, target_h), Image.Resampling.BILINEAR)

    return np.array(img, dtype=np.float32)   # (H, W)


class VMVIDataset(Dataset):
    def __init__(
        self,
        root:           str,
        split:          str        = "train",
        T:              int        = T_PATCH,
        stride:         int        = 1,
        augment:        bool       = True,
        extra_channels: Optional[List[str]]  = None,
    ):
        """
        Args:
            extra_channels: list of additional channels to load alongside input.npy.
                            Supported values: "angle", "gradient".
                            Default None → [] (4-channel baseline).
        """
        self.root           = Path(root)
        self.split          = split
        self.T              = T
        self.stride         = stride
        self.augment        = augment and (split == "train")
        self.extra_channels = extra_channels or []

        # Validate
        for ch in self.extra_channels:
            if ch not in ("angle", "gradient"):
                raise ValueError(f"Unknown extra channel '{ch}'. "
                                 f"Choose from 'angle', 'gradient'.")

        split_file = self.root / f"{split}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")
        self.clips = [l.strip() for l in split_file.read_text().splitlines() if l.strip()]

        if not self.clips:
            raise ValueError(f"No clips found in {split_file}")

        self.index = self._build_index()

    @property
    def in_channels(self) -> int:
        """Total input channels: 4 base + number of extra channels."""
        return 4 + len(self.extra_channels)

    def _build_index(self):
        index = []
        for stem in self.clips:
            label_path = self.root / stem / "label.npy"
            if not label_path.exists():
                print(f"  [WARN] Missing label.npy for {stem} — skipping")
                continue
            label  = np.load(label_path, mmap_mode="r")
            T_full = label.shape[0]
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
            pad       = -start_row
            inp_patch = np.concatenate([
                np.zeros((pad, input_full.shape[1], 4), dtype=np.float32),
                np.array(input_full[0 : end_row + 1]),
            ], axis=0)                                                         # (T, 768, 4)
        else:
            inp_patch = np.array(input_full[start_row : end_row + 1])         # (T, 768, 4)

        # ── Extra channels ────────────────────────────────────────────────
        if self.extra_channels:
            extra_parts = []
            clip_dir = self.root / stem

            for ch in self.extra_channels:
                folder_name = "angle_mp" if ch == "angle" else "gradient"
                full_ch = _load_png_channel(clip_dir / folder_name)           # (1800, 768)

                if start_row < 0:
                    pad    = -start_row
                    ch_patch = np.concatenate([
                        np.zeros((pad, full_ch.shape[1]), dtype=np.float32),
                        full_ch[0 : end_row + 1],
                    ], axis=0)                                                 # (T, 768)
                else:
                    ch_patch = full_ch[start_row : end_row + 1]               # (T, 768)

                # Normalize
                if ch == "angle":
                    ch_patch = (ch_patch - 127.5) / 127.5   # → [-1, 1]
                else:  # gradient
                    ch_patch = ch_patch / 255.0              # → [0, 1]

                extra_parts.append(ch_patch[:, :, np.newaxis])                # (T, 768, 1)

            inp_patch = np.concatenate([inp_patch] + extra_parts, axis=2)     # (T, 768, 4+N)

        target = np.array(label_full[end_row])   # (768,)

        # ── Augmentation (horizontal flip) ────────────────────────────────
        if self.augment and random.random() < 0.5:
            inp_patch = inp_patch[:, ::-1, :].copy()
            target    = target[::-1].copy()

        # (T, W, C) -> (C, T, W) for Conv2d
        x = inp_patch.transpose(2, 0, 1).astype(np.float32)

        return torch.from_numpy(x), torch.from_numpy(target.astype(np.int64))


def get_dataloaders(
    root:           str,
    batch_size:     int       = 4,
    num_workers:    int       = 4,
    stride_train:   int       = 1,
    stride_val:     int       = 30,
    extra_channels: Optional[List[str]] = None,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = VMVIDataset(root, split="train", stride=stride_train,
                           augment=True,  extra_channels=extra_channels)
    val_ds   = VMVIDataset(root, split="val",   stride=stride_val,
                           augment=False, extra_channels=extra_channels)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    print(f"Train: {len(train_ds):,} patches from {len(train_ds.clips)} clips  "
          f"({train_ds.in_channels}ch)")
    print(f"Val:   {len(val_ds):,} patches from {len(val_ds.clips)} clips  "
          f"({val_ds.in_channels}ch)")

    return train_loader, val_loader


def compute_class_weights(root: str, num_classes: int = NUM_CLASSES,
                          max_clips: int = 50) -> torch.Tensor:
    counts = np.zeros(num_classes, dtype=np.int64)
    clips  = [l.strip() for l in (Path(root) / "train.txt").read_text().splitlines()
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
        print("Usage: python data/dataset.py <prepared_root> [angle] [gradient]")
        sys.exit(1)

    extra = sys.argv[2:] if len(sys.argv) > 2 else []
    print(f"Extra channels: {extra or 'none (4ch baseline)'}")

    train_loader, val_loader = get_dataloaders(
        sys.argv[1], batch_size=2, num_workers=0,
        stride_train=300, stride_val=300, extra_channels=extra,
    )
    x, y = next(iter(train_loader))
    print(f"x: {x.shape}   (expect [2, {4+len(extra)}, 256, 768])")
    print(f"y: {y.shape}   (expect [2, 768])")
    print(f"x range: [{x.min():.3f}, {x.max():.3f}]")
    print(f"y range: [{y.min()}, {y.max()}]")
    print("Sanity check passed")