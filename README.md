ST2CN — Vehicle Interaction Recognition
CSCI-49500 Capstone Project — Purdue Computer Science
Duncan Stephenson | Advisor: Dr. Jiang Yu Zheng
Replication and extension of the ST2CN network from:

Lin & Zheng (2023), "Understanding Vehicle Interaction in Driving Video with Spatial-temporal Deep Learning Network", IEEE ITSC 2023


What it does
ST2CN takes a dashcam video clip and classifies vehicle interactions frame by frame. Instead of processing raw video directly, it encodes each clip as a 2D motion profile image — time runs vertically, lane position runs horizontally — and runs a U-Net CNN over that image to predict the interaction class at each spatial position for the current moment.
The output is one prediction per pixel column at the latest frame: which interaction class (cut-in, lane change, merging, etc.) is happening at each horizontal position in the scene.
Shift-mode inference slides the 256-row window one row at a time through the full 1800-row clip, producing a dense prediction at every temporal position with no look-ahead bias.

Dataset
VMVI 2025 — Zheng & Li (2025), Purdue Research Repository
DOI: 10.4231/6WPT-RZ29

394 usable clips after filtering (400 total, 6 excluded due to missing files)
308 train / 86 validation (80/20 split, stratified by class prefix, seed=42)
15 interaction classes
Motion profiles: 1800×2592px → resized to 1800×768 for training


Results
All results use shift-mode evaluation (stride=1) on the validation set.
ModelChannelsmIoUmF1T-IoUBaseline ST2CN4ch0.8060.8910.297PSPNet bottleneck4ch0.3660.4790.196ESP bottleneck4ch0.3630.4790.194+angle (scratch)5ch0.3800.4940.197+gradient (scratch)5ch0.3790.4910.198Focal loss4ch0.3570.4670.192Stratified sampling4ch0.3730.4880.194Fine-tune +angle+gradient6ch0.7780.8720.295
Key finding: The 1×3 bottleneck combined with stride=1 training produces a temporally-sensitive encoding that collapses to ~0.36 mIoU under any modification when trained from scratch. Fine-tuning from the baseline checkpoint preserves this encoding — the 6ch fine-tune recovers 96% of baseline performance.

Project structure
capstone/
    models/
        st2cn.py            # Baseline ST2CN — U-Net encoder-decoder, 15-class head
        st2cn_psp.py        # PSPNet pyramid pooling bottleneck variant
        st2cn_esp.py        # ESP dilated convolution bottleneck variant
    data/
        dataset.py          # VMVIDataset — sliding window, extra channels, stratified sampling
    utils/
        metrics.py          # ST-IoU, approximate T-IoU, F1 
    scripts/
        train.slurm             # Baseline training job
        train_stratified.slurm  # Stratified sampling training job
        train_finetune.slurm    # Fine-tune from baseline checkpoint
        eval_all.slurm          # Evaluate all models
    train.py                # Main training loop
    train_finetune.py       # Fine-tune script — extends first conv, freezes all other layers
    evaluate.py             # Shift-mode eval with T-IoU and frame counts
    plot_input_output.py    # Visualization — motion profile, predictions, ground truth

Setup (on Gilbreth cluster)
bashcd ~/capstone
module load cuda/12.6.0
module load cudnn/9.2.0.82-12
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install numpy pillow matplotlib

Usage
1. Prepare dataset
bashpython data/prepare_dataset.py \
    --raw_dir /scratch/gilbreth/$USER/vivm/raw \
    --out_dir /scratch/gilbreth/$USER/vivm/prepared \
    --seed    42
2. Train baseline
bashsbatch scripts/train.slurm
Training takes ~5 days on a single NVIDIA A30 at stride=1. Key flags:

--model — baseline, psp, esp
--extra_channels angle gradient — add trajectory feature channels
--stratified — enable WeightedRandomSampler for rare class oversampling
--stride_train 1 — required for shift-mode gain (stride=15 produces ~0.37 regardless)

3. Fine-tune from baseline checkpoint
bashsbatch scripts/train_finetune.slurm
Extends first conv from 4→6 channels, freezes all other layers, trains for 20 epochs at lr=1e-4.
4. Evaluate (shift-mode)
bashpython evaluate.py \
    --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_baseline_20260228_180438/best_model.pth \
    --data_root  /scratch/gilbreth/ddstephe/vivm/prepared \
    --model      baseline \
    --output_dir /scratch/gilbreth/$USER/eval_results/baseline
Expected: mIoU = 0.806, mF1 = 0.891. Saves eval_results.json and confusion_matrix.png.
5. Visualize predictions
bash# Requires GPU — get interactive node first
sinteractive -A zheng826 -p a30 --gres=gpu:1 --mem=60G -t 00:30:00

python plot_input_output.py \
    --data_dir   /scratch/gilbreth/ddstephe/vivm/prepared \
    --model_path /scratch/gilbreth/ddstephe/runs/st2cn_baseline_20260228_180438/best_model.pth \
    --model      baseline \
    --clip_name  "cutting_in 6" \
    --out        output.png
6. Monitor training
bashsqueue -u $USER
tail -f ~/capstone/logs/$(ls -t ~/capstone/logs/ | head -1)

Pre-trained checkpoints
All checkpoints are on Gilbreth at /scratch/gilbreth/ddstephe/runs/:
RunPathBaselinest2cn_baseline_20260228_180438/best_model.pthPSPNetst2cn_psp_20260315_093413/best_model.pthESPst2cn_esp_20260318_125917/best_model.pth+anglest2cn_angle_20260329_131127/best_model.pth+gradientst2cn_gradient_20260323_153648/best_model.pthFocal lossst2cn_focal_20260404_151017/best_model.pthStratifiedst2cn_stratified_20260414_160350/best_model.pthFine-tunest2cn_finetune_20260419_122625/best_model.pth

References
Lin, L., & Zheng, J. Y. (2023). Understanding vehicle interaction in driving video with spatial-temporal deep learning network. IEEE ITSC 2023. https://doi.org/10.1109/ITSC57777.2023.10421903
Zheng, J. Y., & Li, Z. (2025). VMVI: Vehicle motion and vehicle interaction dataset. Purdue University Research Repository. https://doi.org/10.4231/6WPT-RZ29