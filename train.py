# ST2CN Training Script
# Reproduces the training setup from Lin & Zheng (2023):
#   Optimizer : SGD, lr=0.001, momentum=0.9, weight_decay=1e-4
#   LR policy : Poly decay (DeepLab-v2 style)
#   Epochs    : 100
#   Loss      : Cross-Entropy on last temporal line only
#
# Usage:
#   python train.py --data_root /scratch/gilbreth/$USER/vivm/prepared --output_dir runs/baseline
#   See scripts/train.slurm for cluster submission

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast

sys.path.insert(0, str(Path(__file__).parent))
from models.st2cn  import ST2CN, ST2CNLoss, NUM_CLASSES
from data.dataset  import get_dataloaders, compute_class_weights
from utils.metrics import SegmentationMetrics


def parse_args():
    p = argparse.ArgumentParser(description="Train ST2CN baseline")
    p.add_argument("--data_root",    type=str, required=True)
    p.add_argument("--output_dir",   type=str, default="runs/baseline")
    p.add_argument("--epochs",       type=int,   default=100)
    p.add_argument("--batch_size",   type=int,   default=4)
    p.add_argument("--lr",           type=float, default=1e-3)
    p.add_argument("--momentum",     type=float, default=0.9)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--num_workers",  type=int,   default=4)
    p.add_argument("--stride_train", type=int,   default=1,
                   help="Temporal stride for training patches. "
                        "stride=1 is dense (paper default). "
                        "stride=15 is ~15x faster for debugging but starves rare classes.")
    p.add_argument("--base_filters", type=int,   default=16)
    p.add_argument("--num_classes",  type=int,   default=NUM_CLASSES)
    p.add_argument("--model",        type=str,   default="baseline",
                   choices=["baseline", "asym", "psp", "esp"],
                   help="baseline=original ST2CN, asym=asymmetric pooling variant, psp=pyramid pooling variant")
    p.add_argument("--extra_channels", nargs="*", default=[], choices=["angle", "gradient"], help="Additional input channels for ablation")
    p.add_argument("--class_weights", action="store_true",
                   help="Use inverse-frequency class weights (didn't help in run 2)")
    p.add_argument("--amp",          action="store_true", default=True)
    p.add_argument("--resume",       type=str,   default=None,
                   help="Path to checkpoint to resume from")
    p.add_argument("--eval_every",   type=int,   default=5)
    p.add_argument("--seed",         type=int,   default=42)
    p.add_argument("--stratified", action="store_true",
               help="Use WeightedRandomSampler to oversample rare classes")
    return p.parse_args()


class PolyLR(torch.optim.lr_scheduler._LRScheduler):
    # lr = base_lr * (1 - iter/max_iter)^power, applied per step (DeepLab-v2 style)
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


def load_checkpoint(path, model, optimizer=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    start_epoch = ckpt.get("epoch", 0) + 1
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    print(f"Resumed from {path}  (epoch {ckpt.get('epoch', 0)})")
    return start_epoch


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, device, epoch, use_amp):
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
            print(f"  E{epoch:03d} [{step:5d}/{len(loader)}]  loss={loss.item():.4f}  lr={lr:.6f}  eta={eta/60:.1f}min")

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

    train_loader, val_loader = get_dataloaders(
        root            = args.data_root,
        batch_size      = args.batch_size,
        num_workers     = args.num_workers,
        stride_train    = args.stride_train,
        extra_channels  = args.extra_channels,
        stratified      = args.stratified,
    )

    if args.model == "asym":
        from models.st2cn_asym import ST2CN_Asym
        model = ST2CN_Asym(in_channels=4 + len(args.extra_channels), num_classes=args.num_classes,
                           base_filters=args.base_filters).to(device)
        print("Model: ST2CN_Asym (asymmetric pooling)")
    elif args.model == "psp":
        from models.st2cn_psp import ST2CN_PSP
        model = ST2CN_PSP(in_channels=4 + len(args.extra_channels), num_classes=args.num_classes,
                          base_filters=args.base_filters).to(device)
        print("Model: ST2CN_PSP (pyramid pooling)")
    elif args.model == "esp":
        from models.st2cn_esp import ST2CN_ESP
        model = ST2CN_ESP(in_channels=4 + len(args.extra_channels), num_classes=args.num_classes,
                          base_filters=args.base_filters).to(device)
        print("Model: ST2CN_ESP (ESPpyramid pooling)")
    else:
        model = ST2CN(in_channels=4 + len(args.extra_channels), num_classes=args.num_classes,
                      base_filters=args.base_filters).to(device)
        print("Model: ST2CN baseline")

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    if args.class_weights:
        print("Computing class weights...")
        weights   = compute_class_weights(args.data_root, args.num_classes)
        criterion = ST2CNLoss(class_weights=weights.to(device))
    else:
        criterion = ST2CNLoss()

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay)
    max_iter  = args.epochs * len(train_loader)
    scheduler = PolyLR(optimizer, max_iter=max_iter, power=0.9)
    scaler    = GradScaler(enabled=args.amp)

    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(args.resume, model, optimizer, scaler)

    best_miou = 0.0
    history   = []

    for epoch in range(start_epoch, args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scheduler, scaler, device, epoch + 1, args.amp)
        print(f"  Train loss: {train_loss:.4f}")

        ckpt_path = out_dir / "checkpoints" / f"epoch_{epoch+1:03d}.pth"
        save_checkpoint({
            "epoch":      epoch,
            "model":      (model.module if hasattr(model, "module") else model).state_dict(),
            "optimizer":  optimizer.state_dict(),
            "scaler":     scaler.state_dict(),
            "train_loss": train_loss,
        }, str(ckpt_path))

        if (epoch + 1) % args.eval_every == 0:
            val_loss, val_results = validate(model, val_loader, criterion, device, args.amp)
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
                    "model": (model.module if hasattr(model, "module") else model).state_dict(),
                    "miou":  miou,
                }, str(out_dir / "best_model.pth"))
                print(f"  New best mIoU: {best_miou:.4f}")

    # Final eval using best checkpoint
    print("\n" + "="*60)
    print("Final evaluation on val split")
    print("="*60)
    best_path = out_dir / "best_model.pth"
    if best_path.exists():
        load_checkpoint(str(best_path), model)

    _, final_results = validate(model, val_loader, criterion, device, args.amp)
    SegmentationMetrics().print_table(final_results)
    (out_dir / "final_results.json").write_text(json.dumps(final_results, indent=2, default=str))
    print(f"\nFinal val mIoU: {final_results['mean_iou']:.4f}")
    print(f"Final val mF1:  {final_results['mean_f1']:.4f}")
    print(f"Outputs saved to: {out_dir}")


if __name__ == "__main__":
    main()