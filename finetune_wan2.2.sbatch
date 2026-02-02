#!/usr/bin/env bash
#SBATCH --account=torch_pr_147_courant
#SBATCH --nodes=1
#SBATCH --gres=gpu:h200:4
#SBATCH --mem=1000GB
#SBATCH --partition=h200
#SBATCH --comment="preemption=yes;preemption_partitions_only=yes;requeue=true"

#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH -t 48:00:00

#SBATCH --job-name=wan22-ti2v-train
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

set -euo pipefail

cd /scratch/zh2025/finetune_action/DiffSynth-Studio

export DIFFSYNTH_MODEL_BASE_PATH=/scratch/zh2025/DiffSynth-Studio/models
export DIFFSYNTH_SKIP_DOWNLOAD=true

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path /scratch/zh2025/finetune_action/DiffSynth-Studio/data/lerobot_clothes_folding_25f_100k \
  --dataset_metadata_path /scratch/zh2025/finetune_action/DiffSynth-Studio/data/lerobot_clothes_folding_25f_100k/metadata.csv \
  --data_file_keys "video,action_tokens" \
  --height 256 \
  --width 256 \
  --num_frames 25 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" \
  --model_paths '[
    "/scratch/zh2025/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  ]' \
  --action_token_dim 256 \
  --gradient_accumulation_steps 16 \
  --save_steps 5000 \
  --resume_from_checkpoint /scratch/zh2025/finetune_action/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_full/step-22540.safetensors \
  --resume_num_steps 22540 \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit,text_encoder" \
  --extra_inputs "input_image"
