#!/bin/bash
#SBATCH --job-name=eval_ag
#SBATCH --account=zheng826
#SBATCH --partition=a30
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%j_eval_ag.out
#SBATCH --error=logs/%j_eval_ag.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ddstephe@purdue.edu

module purge
module load cuda/12.6.0
module load cudnn/9.2.0.82-12

cd ~/capstone
source venv/bin/activate

DATA=/scratch/gilbreth/ddstephe/vivm/prepared
OUT=/scratch/gilbreth/ddstephe/eval_results

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_angle_20260329_131127/best_model.pth --data_root $DATA --model baseline --extra_channels angle --output_dir $OUT/angle

python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_gradient_20260323_153648/best_model.pth --data_root $DATA --model baseline --extra_channels gradient --output_dir $OUT/gradient

echo "Done"
python evaluate.py --checkpoint /scratch/gilbreth/ddstephe/runs/st2cn_finetune_20260419_122625/best_model.pth --data_root $DATA --model baseline --extra_channels angle gradient --output_dir $OUT/finetune
