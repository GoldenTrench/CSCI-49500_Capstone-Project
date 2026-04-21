#!/bin/bash
#SBATCH --job-name=eval_all
#SBATCH --account=zheng826
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%j_eval_all.out
#SBATCH --error=logs/%j_eval_all.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddstephe@purdue.edu

module purge
module load cuda/12.6.0
module load cudnn/9.2.0.82-12

cd ~/capstone
source venv/bin/activate

DATA=/scratch/gilbreth/ddstephe/vivm/prepared
OUT=/scratch/gilbreth/ddstephe/eval_results
mkdir -p $OUT

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_baseline_20260228_180438/best_model.pth --data_root $DATA --model baseline --output_dir $OUT/baseline

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_psp_20260315_093413/best_model.pth --data_root $DATA --model psp --output_dir $OUT/psp

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_esp_20260318_125917/best_model.pth --data_root $DATA --model esp --output_dir $OUT/esp

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_angle_20260329_131127/best_model.pth --data_root $DATA --model baseline --extra_channels angle --output_dir $OUT/angle

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_gradient_20260323_153648/best_model.pth --data_root $DATA --model baseline --extra_channels gradient --output_dir $OUT/gradient

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_focal_20260404_151017/best_model.pth --data_root $DATA --model baseline --output_dir $OUT/focal

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_stratified_20260414_160350/best_model.pth --data_root $DATA --model baseline --output_dir $OUT/stratified

echo "All evals done"
