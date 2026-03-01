#!/bin/bash
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:4
#SBATCH --mem=256G
#SBATCH --cpus-per-task=128
#SBATCH --time=04:00:00
#SBATCH --job-name=wan_cf_5b_full
#SBATCH --chdir=/scratch/zh2025/finetune_action/action_cond
#SBATCH --output=/scratch/zh2025/finetune_action/action_cond/logs/%x_%j.out
#SBATCH --error=/scratch/zh2025/finetune_action/action_cond/logs/%x_%j.err

set -euo pipefail

mkdir -p logs

export PATH="/scratch/zh2025/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/scratch/zh2025/miniforge3/lib:${LD_LIBRARY_PATH:-}"
source /scratch/zh2025/miniforge3/etc/profile.d/conda.sh
conda activate diffsynth

bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sh
