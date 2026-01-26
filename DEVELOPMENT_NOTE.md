# Development Note: WAN2.2 TI2V (Action Tokens) Training

```bash
export DIFFSYNTH_MODEL_BASE_PATH=/scratch/zh2025/DiffSynth-Studio/models
export DIFFSYNTH_SKIP_DOWNLOAD=true

accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/example_video_dataset \
  --dataset_metadata_path data/example_video_dataset/metadata.csv \
  --data_file_keys "video,action_tokens" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors" \
  --model_paths '[
    "/scratch/zh2025/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"
  ]' \
  --action_token_dim 128 \
  --learning_rate 1e-5 \
  --num_epochs 2 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path "./models/train/Wan2.2-TI2V-5B_full" \
  --trainable_models "dit,text_encoder" \
  --extra_inputs "input_image"
```
