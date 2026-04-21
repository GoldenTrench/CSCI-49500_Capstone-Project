"""
Fine-tune ST2CN baseline with angle+gradient channels (6ch).

Strategy:
- Load baseline checkpoint (4ch, 0.806 shift-mode mIoU)
- Extend first conv from 4->6 channels
  - Copy existing 4ch weights exactly
  - Initialize new 2 channels with small random values (0.01 std)
    so they contribute near-zero initially, preserving temporal encoding
- Freeze ALL layers except encoders.0.conv (the first conv)
- Train for 20 epochs at lr=1e-4 (10x lower than baseline)
- Evaluate in patch-mode every 5 epochs as usual

Rationale:
  Training angle+gradient from scratch disrupts the temporal encoding
  that produces the 0.806 shift-mode gain. Fine-tuning from the baseline
  preserves that encoding while allowing the network to learn how to use
  the new channels.

Usage:
    python train_finetune.py \
        --data_root    /scratch/gilbreth/$USER/vivm/prepared \
        --output_dir   /scratch/gilbreth/$USER/runs/st2cn_finetune_angle_gradient \
        --baseline_ckpt /scratch/gilbreth/$USER/runs/st2cn_baseline_20260228_180438/best_model.pth
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from models.st2cn  import ST2CN, ST2CNLoss, NUM_CLASSES
from data.dataset  import get_dataloaders
from utils.metrics import SegmentationMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune ST2CN with angle+gradient channels")
    p.add_argument("--data_root",     type=str, required=True)
    p.add_argument("--output_dir",    type=str, required=True)
    p.add_argument("--baseline_ckpt", type=str, required=True,
                   help="Path to baseline best_model.pth (4ch)")
    p.add_argument("--epochs",        type=int,   default=20)
    p.add_argument("--batch_size",    type=int,   default=4)
    p.add_argument("--lr",            type=float, default=1e-4,
                   help="Low LR to avoid disrupting temporal encoding")
    p.add_argument("--momentum",      type=float, default=0.9)
    p.add_argument("--weight_decay",  type=float, default=1e-4)
    p.add_argument("--num_workers",   type=int,   default=20)
    p.add_argument("--base_filters",  type=int,   default=16)
    p.add_argument("--amp",           action="store_true", default=True)
    p.add_argument("--eval_every",    type=int,   default=5)
    p.add_argument("--seed",          type=int,   default=42)
    return p.parse_args()


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, max_iter, power=0.9, last_epoch=-1):
        self.max_iter = max_iter
        self.power    = power
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        factor = (1.0 - min(self.last_epoch / self.max_iter, 1.0)) ** self.power
        return [base_lr * factor for base_lr in self.base_lrs]


def set_seed(seed):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(state, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def build_6ch_model(baseline_ckpt_path, base_filters, device):
    """
    Load baseline 4ch weights and extend first conv to 6ch.
    New channels initialized with small random weights.
    """
    # Load baseline state dict
    ckpt  = torch.load(baseline_ckpt_path, map_location="cpu")
    state = ckpt.get("model", ckpt)

    # Build 6ch model
    model = ST2CN(in_channels=6, num_classes=NUM_CLASSES,
                  base_filters=base_filters)

    # Copy all weights except first conv
    model_state = model.state_dict()
    for k, v in state.items():
        if k == "encoders.0.conv.weight":
            # Old: [16, 4, 3, 3] -> New: [16, 6, 3, 3]
            new_weight = torch.zeros(v.shape[0], 6, v.shape[2], v.shape[3])
            new_weight[:, :4, :, :] = v          # copy existing 4ch weights exactly
            new_weight[:, 4:, :, :] = torch.randn(v.shape[0], 2,
                                                    v.shape[2], v.shape[3]) * 0.01
            model_state[k] = new_weight
            print(f"Extended first conv: {v.shape} -> {new_weight.shape}")
        elif k in model_state:
            model_state[k] = v

    model.load_state_dict(model_state)
    print(f"Loaded baseline weights from epoch {ckpt.get('epoch', '?')}")
    print(f"Baseline mIoU: {ckpt.get('miou', '?')}")
    return model.to(device)


def freeze_except_first_conv(model):
    """Freeze all parameters except encoders.0.conv"""
    frozen = 0
    trainable = 0
    for name, param in model.named_parameters():
        if name.startswith("encoders.0.conv"):
            param.requires_grad = True
            trainable += param.numel()
        else:
            param.requires_grad = False
            frozen += param.numel()
    print(f"Frozen params:    {frozen:,}")
    print(f"Trainable params: {trainable:,}  (first conv only)")
    return model


def train_one_epoch(model, loader, criterion, optimizer, scheduler,
                    scaler, device, epoch, use_amp):
    model.train()
    total_loss = 0.0
    t0 = time.time()

    for step, (x, target) in enumerate(loader):
        x      = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast(enabled=use_amp):
            logits = model(x)
            loss   = criterion(logits, target)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        total_loss += loss.item()

        if step % 200 == 0:
            lr  = scheduler.get_last_lr()[0]
            eta = (time.time() - t0) / (step + 1) * (len(loader) - step - 1)
            print(f"  E{epoch:03d} [{step:5d}/{len(loader)}]  "
                  f"loss={loss.item():.4f}  lr={lr:.6f}  eta={eta/60:.1f}min")

    return total_loss / len(loader)


@torch.no_grad()
def validate(model, loader, criterion, device, use_amp):
    model.eval()
    metrics    = SegmentationMetrics(num_classes=NUM_CLASSES)
    total_loss = 0.0

    for x, target in loader:
        x      = x.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with autocast(enabled=use_amp):
            logits = model(x)
            loss   = criterion(logits, target)

        total_loss += loss.item()
        metrics.update(logits.squeeze(2).argmax(dim=1), target)

    return total_loss / len(loader), metrics.compute()


def main():
    args   = parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  |  GPUs: {torch.cuda.device_count()}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(vars(args), indent=2))

    # Data — 6ch with angle+gradient
    train_loader, val_loader = get_dataloaders(
        root            = args.data_root,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        stride_train    = 1,
        extra_channels  = ["angle", "gradient"],
    )

    # Model — extend baseline to 6ch, freeze all but first conv
    model = build_6ch_model(args.baseline_ckpt, args.base_filters, device)
    model = freeze_except_first_conv(model)

    criterion = ST2CNLoss()
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay
    )
    max_iter  = args.epochs * len(train_loader)
    scheduler = PolyLR(optimizer, max_iter=max_iter, power=0.9)
    scaler    = GradScaler(enabled=args.amp)

    best_miou = 0.0
    history   = []

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scheduler, scaler, device, epoch + 1, args.amp)
        print(f"  Train loss: {train_loss:.4f}")

        ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch+1:03d}.pth"
        save_checkpoint({
            "epoch":      epoch,
            "model":      model.state_dict(),
            "train_loss": train_loss,
        }, str(ckpt_path))

        if (epoch + 1) % args.eval_every == 0:
            val_loss, val_results = validate(model, val_loader, criterion,
                                             device, args.amp)
            miou = val_results["mean_iou"]
            mf1  = val_results["mean_f1"]
            print(f"  Val loss: {val_loss:.4f}  mIoU: {miou:.4f}  mF1: {mf1:.4f}")

            SegmentationMetrics().print_table(val_results)

            history.append({"epoch": epoch + 1, "train_loss": train_loss,
                            "val_loss": val_loss, "miou": miou, "mf1": mf1})
            (out_dir / "history.json").write_text(json.dumps(history, indent=2))

            if miou > best_miou:
                best_miou = miou
                save_checkpoint({
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "miou":  miou,
                }, str(out_dir / "best_model.pth"))
                print(f"  New best mIoU: {best_miou:.4f}")

    print(f"\nDone. Best patch-mode mIoU: {best_miou:.4f}")
    print(f"Outputs saved to: {out_dir}")
    print("Run evaluate.py with --model baseline --extra_channels angle gradient")
    print("to get shift-mode results.")


if __name__ == "__main__":
    main()