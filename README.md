# ST2CN — Vehicle Interaction Recognition

CSCI-49500 Capstone Project — Purdue Computer Science  
Duncan Stephenson | Advisor: Dr. Zheng

Replication and improvement of the ST2CN network from:
> Lin & Zheng (2023), *"Understanding Vehicle Interaction in Driving Video with Spatial-temporal Deep Learning Network"*, IEEE ITSC 2023

---

## What it does

ST2CN takes a dashcam video clip and classifies vehicle interactions frame by frame. Instead of processing raw video directly, it encodes each clip as a 2D **motion profile image** — time runs vertically, lane position runs horizontally — and runs a U-Net CNN over that image to predict the interaction class at each spatial position for the current moment.

The output is one prediction per pixel column at the latest frame: which interaction class (cut-in, lane change, approaching, etc.) is happening at each horizontal position in the scene.

---

## Dataset

Using **VMVI 2025** (downloaded from Purdue Research Repository, `10_4231_6WPT-RZ29`).

- 394 matched clips after filtering
- 15 interaction classes (paper had 13 — new: off_road; stopping and turning_away are separate here vs combined in paper)
- Raw images: 1800×2592px → resized to 1800×768 for training

The ground truth PNGs have JPEG compression artifacts from the annotation pipeline, so color-to-class mapping uses nearest-color L2 matching instead of exact lookup.

---

## Project structure

```
capstone/
    models/
        st2cn.py          # U-Net encoder-decoder, 8+8 blocks, 15-class head
    data/
        dataset.py        # VMVIDataset — sliding window over .npy arrays
    utils/
        metrics.py        # ST-IoU, T-IoU, F1 (per Lin & Zheng Table II)
    scripts/
        prepare_dataset.py  # raw PNG -> .npy arrays
        train.slurm         # SLURM job script for Purdue RCAC Gilbreth
    motion_profile.py     # MP/WP generation from raw video (inference)
    train.py              # training loop
```

---

## Setup (on Gilbreth)

```bash
cd ~/capstone
module load cuda/12.6.0
module load cudnn/9.2.0.82-12
python -m venv venv
source venv/bin/activate
pip install torch torchvision opencv-python numpy
```

---

## Usage

**1. Prepare dataset**
```bash
python scripts/prepare_dataset.py \
    --raw_dir /scratch/gilbreth/$USER/vivm/raw/10_4231_6WPT-RZ29 \
    --out_dir /scratch/gilbreth/$USER/vivm/prepared
```

If you hit a zip bomb warning extracting the dataset:
```bash
UNZIP_DISABLE_ZIPBOMB_DETECTION=TRUE unzip -o data.zip
```
This is a false positive — large datasets with overlapping zip entries trigger it incorrectly.

**2. Sanity check model**
```bash
python models/st2cn.py
```

**3. Submit training job**
```bash
sbatch scripts/train.slurm
```

**4. Monitor**
```bash
squeue -u $USER
tail -f ~/capstone/logs/{JOBID}_train.out
```

**5. Resume from checkpoint**

Add `--resume /path/to/checkpoint.pth` to the python command in `train.slurm`.

---

## Results so far

| Run | Stride | Class weights | mIoU | mF1 |
|-----|--------|--------------|------|-----|
| Run 1 | 15 | No  | 0.373 | 0.501 |
| Run 2 | 15 | Yes | 0.346 | 0.470 |
| Run 3 | 1  | No  | in progress | — |

**Key finding:** stride=15 was the wrong tradeoff. With a 256-row window, stride=15 generates ~32k patches/epoch. At stride=1 it's ~487k. Rare classes like merging (35 IoU) and cut_in (19 IoU) barely appear per epoch at stride=15 — not enough to learn from. Class weighting can't fix what data density breaks.

Paper target metrics (different dataset): ST-IoU 0.85, T-IoU 0.89, F1 0.85

---

## Phase 2 (planned)

1. **PSPNet pyramid pooling** at the bottleneck — multi-scale spatial context
2. **Temporal consistency loss** — penalize frame-to-frame prediction changes
3. **Ego-turn detection** — detect ego-vehicle turns from the slanted trace pattern
4. **Ablation study** comparing each component against the stride=1 baseline