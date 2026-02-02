#!/usr/bin/env bash
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --constraint=l40s
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --job-name=wan22-infer
#SBATCH --output=slurm-infer-%x-%j.out
#SBATCH --error=slurm-infer-%x-%j.err

set -euo pipefail

if [[ -z "${CKPT:-}" ]]; then
  echo "ERROR: CKPT env var not set (path to .safetensors checkpoint)" >&2
  exit 1
fi

cd /scratch/zh2025/finetune_action/DiffSynth-Studio

export DIFFSYNTH_MODEL_BASE_PATH=/scratch/zh2025/DiffSynth-Studio/models
export DIFFSYNTH_SKIP_DOWNLOAD=true

MAX_SAMPLES="${MAX_SAMPLES:-8}"
OUTPUT_BASE="${OUTPUT_BASE:-./inference_gifs}"

ckpt_name="$(basename "${CKPT}" .safetensors)"
out_dir="${OUTPUT_BASE}/${ckpt_name}"

python examples/wanvideo/model_inference/infer_ti2v_action_tokens_gif.py \
  --checkpoint_path "${CKPT}" \
  --output_dir "${out_dir}" \
  --max_samples "${MAX_SAMPLES}"
