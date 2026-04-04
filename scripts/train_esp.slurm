#!/bin/bash
#SBATCH --job-name=st2cn_esp
#SBATCH --account=zheng826
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=logs/%j_train_esp.out
#SBATCH --error=logs/%j_train_esp.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddstephe@purdue.edu

# Experiment: ESPNet pyramid pooling module at the bottleneck
# Replaces the 1x1 bottleneck conv with multi-scale adaptive pooling

module purge
module load cuda/12.6.0
module load cudnn/9.2.0.82-12

cd ~/capstone
source venv/bin/activate

DATA_ROOT=/scratch/gilbreth/$USER/vivm/prepared
OUTPUT_DIR=/scratch/gilbreth/$USER/runs/st2cn_esp_$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUTPUT_DIR"

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
    --model        esp \
    --amp \
    --eval_every   5 \
    --resume       /scratch/gilbreth/ddstephe/runs/st2cn_esp_20260316_232546/checkpoints/epoch_020.pth
 

echo "Done. Results in $OUTPUT_DIR"
