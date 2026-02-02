#!/usr/bin/env bash
set -euo pipefail

CKPT_DIR="${1:-/scratch/zh2025/finetune_action/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_full}"
MAX_SAMPLES="${MAX_SAMPLES:-8}"
OUTPUT_BASE="${OUTPUT_BASE:-./inference_gifs}"

shopt -s nullglob
ckpts=("${CKPT_DIR}"/step-*.safetensors)
if [[ ${#ckpts[@]} -eq 0 ]]; then
  echo "No step-*.safetensors found in ${CKPT_DIR}" >&2
  exit 1
fi

for ckpt in "${ckpts[@]}"; do
  name="$(basename "${ckpt}" .safetensors)"
  sbatch \
    --job-name="wan22-infer-${name}" \
    --export=ALL,CKPT="${ckpt}",MAX_SAMPLES="${MAX_SAMPLES}",OUTPUT_BASE="${OUTPUT_BASE}" \
    run_infer_one_ckpt.sbatch
done
