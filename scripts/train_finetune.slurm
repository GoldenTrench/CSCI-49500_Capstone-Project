#!/bin/bash
#SBATCH --job-name=st2cn_finetune
#SBATCH --account=zheng826
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=20
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%j_train_finetune.out
#SBATCH --error=logs/%j_train_finetune.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddstephe@purdue.edu

module purge
module load cuda/12.6.0
module load cudnn/9.2.0.82-12

cd ~/capstone
source venv/bin/activate

DATA_ROOT=/scratch/gilbreth/$USER/vivm/prepared
OUTPUT_DIR=/scratch/gilbreth/$USER/runs/st2cn_finetune_$(date +%Y%m%d_%H%M%S)
mkdir -p logs "$OUTPUT_DIR"

python fine_tune.py \
    --data_root     "$DATA_ROOT" \
    --output_dir    "$OUTPUT_DIR" \
    --baseline_ckpt /scratch/gilbreth/ddstephe/runs/st2cn_baseline_20260228_180438/best_model.pth \
    --epochs        20 \
    --batch_size    4 \
    --lr            1e-4 \
    --num_workers   20 \
    --amp \
    --eval_every    5

echo "Done. Results in $OUTPUT_DIR"
