#!/bin/bash
#SBATCH -A zheng826
#SBATCH -p a30
#SBATCH --gres=gpu:1
#SBATCH --mem=60G
#SBATCH -t 00:20:00
#SBATCH -o ~/capstone/logs/viz_%j.out

cd ~/capstone
source venv/bin/activate
python plot_input_output.py \
    --data_dir  /scratch/gilbreth/$USER/vivm/prepared/ \
    --model_path /scratch/gilbreth/$USER/runs/st2cn_baseline_20260228_180438/best_model.pth \
    --clip_name "cutting_in 1" \
    --out       ~/capstone/eval_output/baseline/input_output.png
