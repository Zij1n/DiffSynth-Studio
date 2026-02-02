# Development Note: WAN2.2 TI2V (Action Tokens) Training

## Finetuning (current setup)

### Where to run

- Slurm script: `finetune_wan2.2.sbatch`
- Working dir: `/scratch/zh2025/finetune_action/DiffSynth-Studio`
- Model cache:
  - `DIFFSYNTH_MODEL_BASE_PATH=/scratch/zh2025/DiffSynth-Studio/models`
  - `DIFFSYNTH_SKIP_DOWNLOAD=true`

### Training command (from sbatch)

- Key settings:
  - Dataset: `data/lerobot_clothes_folding_25f_100k` (via absolute path in sbatch)
  - Shape: `256x256`, `25` frames
  - Trainables: `dit,text_encoder`
  - Extra inputs: `input_image`
  - Action tokens: `--action_token_dim 256`
  - Checkpoint cadence: `--save_steps 5000`
  - Output: `./models/train/Wan2.2-TI2V-5B_full`

### Resume + fast-forward

- Resume weights only (no optimizer/scheduler restore):
  - `--resume_from_checkpoint /scratch/zh2025/finetune_action/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_full/step-15000.safetensors`
  - `--resume_num_steps 15000`
- Fast-forward behavior:
  - The dataloader skips the first `resume_num_steps` batches before training.
  - Progress bar in `.err` will still start at 0% because it is per-epoch.
  - Expect next checkpoint at `step-(resume_num_steps + save_steps)`.

### What is (and is not) restored

- Restored: trainable weights in the checkpoint.
- Not restored: optimizer state, LR scheduler, RNG state, dataloader position beyond fast-forward.

## Inference (side-by-side GIF)

- Script: `examples/wanvideo/model_inference/infer_ti2v_action_tokens_gif.py`
- Output: ground truth on the left, prediction on the right (one GIF per sample).

### Example

```bash
python examples/wanvideo/model_inference/infer_ti2v_action_tokens_gif.py \
  --checkpoint_path /scratch/zh2025/finetune_action/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_full/step-15000.safetensors \
  --output_dir ./inference_gifs
```

### Cluster submission (one job per checkpoint)

- Slurm script: `run_infer_one_ckpt.sbatch`
- Submit helper: `submit_infer_ckpts.sh`

```bash
cd /scratch/zh2025/finetune_action/DiffSynth-Studio
bash submit_infer_ckpts.sh
```

Optional overrides:

```bash
MAX_SAMPLES=8 OUTPUT_BASE=./inference_gifs bash submit_infer_ckpts.sh \
  /scratch/zh2025/finetune_action/DiffSynth-Studio/models/train/Wan2.2-TI2V-5B_full
```

### Useful flags

- Limit samples: `--max_samples 8` (use `0` for all)
- Skip to an index: `--start_index 100`
- GIF FPS: `--fps 8`
- CFG: `--cfg_scale 5.0` (default), `--cfg_merge` optional
- Steps: `--num_inference_steps 50` (default)
## Dataset expectations

- `metadata.csv` must contain `video` and `action_tokens` columns.
- `action_tokens` should point to a `.pth` file or be inline numeric data.

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/scratch/zh2025/DiffSynth-Studio/models
export DIFFSYNTH_SKIP_DOWNLOAD=true

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --data_file_keys "video,action_tokens" \
  --height 256 \
  --width 256 \
  --num_frames 25 \
  --dataset_repeat 1 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" \
  --model_paths '[
    "/scratch/zh2025/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  ]' \
  --action_token_dim 256 \
  --learning_rate 1e-5 \
  --num_epochs 1 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit,text_encoder" \
  --extra_inputs "input_image"
```
