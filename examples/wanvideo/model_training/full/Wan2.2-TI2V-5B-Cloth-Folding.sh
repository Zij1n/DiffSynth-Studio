#!/bin/bash
set -euo pipefail

OUTPUT_PATH="/gpfs/scratch/zh2025/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_cloth_folding_full"
WAN_NUM_EPOCHS="${WAN_NUM_EPOCHS:-5}"
WAN_ENABLE_TORCH_PROFILER="${WAN_ENABLE_TORCH_PROFILER:-1}"
WAN_PROFILER_TRACE_DIR="${WAN_PROFILER_TRACE_DIR:-${OUTPUT_PATH}/profiler}"
WAN_PROFILER_ACTIVITIES="${WAN_PROFILER_ACTIVITIES:-cpu,cuda}"
WAN_PROFILER_WAIT_STEPS="${WAN_PROFILER_WAIT_STEPS:-1}"
WAN_PROFILER_WARMUP_STEPS="${WAN_PROFILER_WARMUP_STEPS:-1}"
WAN_PROFILER_ACTIVE_STEPS="${WAN_PROFILER_ACTIVE_STEPS:-4}"
WAN_PROFILER_REPEAT="${WAN_PROFILER_REPEAT:-1}"
WAN_PROFILER_ALL_PROCESSES="${WAN_PROFILER_ALL_PROCESSES:-1}"
WAN_PROFILER_RECORD_SHAPES="${WAN_PROFILER_RECORD_SHAPES:-0}"
WAN_PROFILER_PROFILE_MEMORY="${WAN_PROFILER_PROFILE_MEMORY:-0}"
WAN_PROFILER_WITH_STACK="${WAN_PROFILER_WITH_STACK:-0}"
WAN_PROFILER_WITH_FLOPS="${WAN_PROFILER_WITH_FLOPS:-0}"

profiler_args=()
if [[ "${WAN_ENABLE_TORCH_PROFILER}" == "1" ]]; then
  profiler_args+=(
    --enable_profiler
    --profiler_trace_dir "${WAN_PROFILER_TRACE_DIR}"
    --profiler_activities "${WAN_PROFILER_ACTIVITIES}"
    --profiler_wait_steps "${WAN_PROFILER_WAIT_STEPS}"
    --profiler_warmup_steps "${WAN_PROFILER_WARMUP_STEPS}"
    --profiler_active_steps "${WAN_PROFILER_ACTIVE_STEPS}"
    --profiler_repeat "${WAN_PROFILER_REPEAT}"
  )
  if [[ "${WAN_PROFILER_ALL_PROCESSES}" == "1" ]]; then
    profiler_args+=(--profiler_all_processes)
  fi
  if [[ "${WAN_PROFILER_RECORD_SHAPES}" == "1" ]]; then
    profiler_args+=(--profiler_record_shapes)
  fi
  if [[ "${WAN_PROFILER_PROFILE_MEMORY}" == "1" ]]; then
    profiler_args+=(--profiler_profile_memory)
  fi
  if [[ "${WAN_PROFILER_WITH_STACK}" == "1" ]]; then
    profiler_args+=(--profiler_with_stack)
  fi
  if [[ "${WAN_PROFILER_WITH_FLOPS}" == "1" ]]; then
    profiler_args+=(--profiler_with_flops)
  fi
fi

accelerate launch --config_file examples/wanvideo/model_training/full/accelerate_config_5B_2gpu.yaml examples/wanvideo/model_training/train.py \
  --dataset_base_path /gpfs/scratch/zh2025/DiffSynth-Studio/cloth_folding \
  --dataset_type cloth_folding_action \
  --action_dataset_split train \
  --action_dataset_sequence_interval 1 \
  --action_dataset_val_start_frame_interval 1 \
  --action_dataset_prompt "cloth folding" \
  --action_feature_dim 20 \
  --height 256 \
  --width 256 \
  --num_frames 13 \
  --dataset_repeat 1 \
  --dataset_num_workers 4 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.2-TI2V-5B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.2-TI2V-5B:Wan2.2_VAE.pth" \
  --learning_rate 1e-5 \
  --num_epochs "${WAN_NUM_EPOCHS}" \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "${OUTPUT_PATH}" \
  --trainable_models "dit" \
  --extra_inputs "input_image" \
  --use_gradient_checkpointing_offload \
  "${profiler_args[@]}"
