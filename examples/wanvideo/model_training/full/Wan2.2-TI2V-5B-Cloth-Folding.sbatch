#!/bin/bash
#SBATCH --account=system
#SBATCH --partition=a100_short
#SBATCH --nodes=1
#SBATCH --gres=gpu:a100:2
#SBATCH --mem=256G
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --job-name=wan_cf_5b_full
#SBATCH --chdir=/gpfs/scratch/zh2025/DiffSynth-Studio
#SBATCH --output=/gpfs/scratch/zh2025/DiffSynth-Studio/logs/%x_%j.out
#SBATCH --error=/gpfs/scratch/zh2025/DiffSynth-Studio/logs/%x_%j.err

set -euo pipefail

mkdir -p logs models/train

export DIFFSYNTH_MODEL_BASE_PATH="/gpfs/scratch/zh2025/DiffSynth-Studio/models"
export PATH="/gpfs/data/shenlab/zh2025/miniforge3/bin:$PATH"
export LD_LIBRARY_PATH="/gpfs/data/shenlab/zh2025/miniforge3/lib:${LD_LIBRARY_PATH:-}"
source /gpfs/data/shenlab/zh2025/miniforge3/etc/profile.d/conda.sh
conda activate diffsynth

bash examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sh
