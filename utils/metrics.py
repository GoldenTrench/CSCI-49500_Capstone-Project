# Evaluation metrics for ST2CN
# Implements the metrics from Table II of Lin & Zheng (2023):
#
#   ST-IoU : pixel-wise IoU over the full trajectory region (related to vehicle size/depth)
#   T-IoU  : vehicle-instance-wise IoU along the centre trace (more robust, less boundary noise)
#   Precision, Recall, F1 : per class, computed along centre traces
#
# In practice we use the pixel-wise SegmentationMetrics for training/validation,
# since we don't have trajectory instance IDs at inference time.

import numpy as np
import torch
from typing import Dict, Optional, Union


INTERACTION_CLASSES = [
    "cut_in",           #  0
    "lane_change",      #  1
    "front_approach",   #  2
    "front_leaving",    #  3
    "front_following",  #  4
    "parallel_next",    #  5
    "passing",          #  6
    "being_passed",     #  7
    "merging",          #  8
    "opposite",         #  9
    "crossing",         # 10
    "turning_away",     # 11
    "off_road",         # 12
    "stopping",         # 13
    "background",       # 14
]
NUM_CLASSES = len(INTERACTION_CLASSES)  # 15


class SegmentationMetrics:
    """
    Accumulates a confusion matrix over batches, then computes
    per-class IoU, precision, recall, and F1.

    Background (index 14) is excluded from averages to match the paper.
    """

    def __init__(self, num_classes: int = NUM_CLASSES, ignore_index: int = 255):
        self.num_classes  = num_classes
        self.ignore_index = ignore_index
        self.conf_matrix  = np.zeros((num_classes, num_classes), dtype=np.int64)

    def reset(self):
        self.conf_matrix.fill(0)

    def update(self, pred: Union[torch.Tensor, np.ndarray],
                     target: Union[torch.Tensor, np.ndarray]):
        if isinstance(pred,   torch.Tensor): pred   = pred.cpu().numpy()
        if isinstance(target, torch.Tensor): target = target.cpu().numpy()

        pred   = pred.flatten().astype(np.int64)
        target = target.flatten().astype(np.int64)

        mask   = target != self.ignore_index
        pred   = np.clip(pred[mask],   0, self.num_classes - 1)
        target = np.clip(target[mask], 0, self.num_classes - 1)

        np.add.at(self.conf_matrix, (target, pred), 1)

    def compute(self) -> dict:
        cm = self.conf_matrix.astype(np.float64)
        tp = np.diag(cm)
        fp = cm.sum(axis=0) - tp
        fn = cm.sum(axis=1) - tp

        iou       = tp / np.maximum(tp + fp + fn, 1)
        precision = tp / np.maximum(tp + fp, 1)
        recall    = tp / np.maximum(tp + fn, 1)
        f1        = 2 * precision * recall / np.maximum(precision + recall, 1e-9)

        valid = list(range(NUM_CLASSES - 1))  # exclude background

        return {
            "per_class": {
                cls: {
                    "iou":       float(iou[i]),
                    "precision": float(precision[i]),
                    "recall":    float(recall[i]),
                    "f1":        float(f1[i]),
                }
                for i, cls in enumerate(INTERACTION_CLASSES)
            },
            "mean_iou":       float(iou[valid].mean()),
            "mean_precision": float(precision[valid].mean()),
            "mean_recall":    float(recall[valid].mean()),
            "mean_f1":        float(f1[valid].mean()),
            "confusion_matrix": cm.astype(np.int64),
        }

    def print_table(self, results: Optional[dict] = None):
        if results is None:
            results = self.compute()
        header = f"{'Class':<22} {'IoU':>6} {'Prec':>6} {'Rec':>6} {'F1':>6}"
        print(header)
        print("-" * len(header))
        for cls in INTERACTION_CLASSES:
            m = results["per_class"][cls]
            print(f"{cls:<22} {m['iou']:6.3f} {m['precision']:6.3f} {m['recall']:6.3f} {m['f1']:6.3f}")
        print("-" * len(header))
        print(f"{'Average (no bg)':<22} {results['mean_iou']:6.3f} "
              f"{results['mean_precision']:6.3f} {results['mean_recall']:6.3f} "
              f"{results['mean_f1']:6.3f}")


def compute_tiou_from_traces(
    pred_map:    np.ndarray,  # [T, W] predicted labels
    gt_map:      np.ndarray,  # [T, W] ground truth labels
    num_classes: int = NUM_CLASSES,
    ignore_index: int = 255,
) -> Dict[str, float]:
    """
    Approximation of T-IoU from the paper.
    For each class, restricts evaluation to the centre column of each
    GT trajectory region (vehicle-instance-wise, per paper Section V-B).
    Full T-IoU would need instance trajectory IDs — this is the best
    we can do without them.
    """
    results = {}
    for c in range(num_classes - 1):
        gt_mask   = (gt_map == c) & (gt_map != ignore_index)
        pred_mask = (pred_map == c)

        centre_mask = np.zeros_like(gt_mask)
        for t in range(gt_mask.shape[0]):
            cols = np.where(gt_mask[t])[0]
            if len(cols) > 0:
                centre_mask[t, (cols[0] + cols[-1]) // 2] = True

        tp = int((pred_mask & centre_mask).sum())
        fp = int((pred_mask & ~gt_mask & centre_mask).sum())
        fn = int((gt_mask   & ~pred_mask & centre_mask).sum())
        results[INTERACTION_CLASSES[c]] = tp / max(tp + fp + fn, 1)

    results["mean"] = float(np.mean(list(results.values())))
    return results


if __name__ == "__main__":
    metrics = SegmentationMetrics(num_classes=NUM_CLASSES)
    W = 768

    # Perfect predictions -> IoU should be 1.0
    pred = np.random.randint(0, NUM_CLASSES, (10, W), dtype=np.int64)
    for i in range(10):
        metrics.update(pred[i], pred[i])
    results = metrics.compute()
    assert abs(results["mean_iou"] - 1.0) < 1e-6
    print("Perfect-prediction check passed")

    # Random predictions
    metrics.reset()
    for _ in range(100):
        p = np.random.randint(0, NUM_CLASSES, W, dtype=np.int64)
        t = np.random.randint(0, NUM_CLASSES, W, dtype=np.int64)
        metrics.update(p, t)
    metrics.print_table()