#!/bin/bash
#SBATCH --job-name=st2cn_angle
#SBATCH --account=zheng826
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/%j_train_angle.out
#SBATCH --error=logs/%j_train_angle.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddstephe@purdue.edu

# ---- Environment -----------------------------------------------------------
module purge
module load cuda/12.6.0
module load cudnn/9.2.0.82-12

cd ~/capstone
source venv/bin/activate

# ---- Paths -----------------------------------------------------------------
DATA_ROOT=/scratch/gilbreth/$USER/vivm/prepared
OUTPUT_DIR=/scratch/gilbreth/$USER/runs/st2cn_angle_$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUTPUT_DIR"

# ---- Run -------------------------------------------------------------------
python train.py \
    --data_root    "$DATA_ROOT" \
    --output_dir   "$OUTPUT_DIR" \
    --epochs       100 \
    --batch_size   4 \
    --lr           1e-3 \
    --momentum     0.9 \
    --weight_decay 1e-4 \
    --num_workers  20 \
    --stride_train 1 \
    --base_filters 16 \
    --amp \
    --eval_every   5 \
    --extra_channels angle
echo "Done. Results in $OUTPUT_DIR"

