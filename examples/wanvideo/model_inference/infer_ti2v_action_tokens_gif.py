#!/usr/bin/env python3
import argparse
import json
import os
from typing import List, Optional, Tuple

import imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from diffsynth.core import UnifiedDataset, load_state_dict
from diffsynth.core.data.operators import ToAbsolutePath, LoadTorchPickle
from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig


def parse_model_configs(model_paths_json: Optional[str], model_id_with_origin_paths: Optional[str]) -> List[ModelConfig]:
    model_configs: List[ModelConfig] = []
    if model_paths_json:
        model_paths = json.loads(model_paths_json)
        for path in model_paths:
            model_configs.append(ModelConfig(path=path))
    if model_id_with_origin_paths:
        for item in model_id_with_origin_paths.split(","):
            item = item.strip()
            if not item:
                continue
            if ":" not in item:
                raise ValueError(f"Invalid model_id_with_origin_paths entry: {item}")
            split_id = item.rfind(":")
            model_id = item[:split_id]
            origin_file_pattern = item[split_id + 1 :]
            model_configs.append(ModelConfig(model_id=model_id, origin_file_pattern=origin_file_pattern))
    return model_configs


def load_checkpoint_into_pipe(pipe: WanVideoPipeline, checkpoint_path: str, remove_prefix_in_ckpt: Optional[str]):
    state_dict = load_state_dict(checkpoint_path, torch_dtype=pipe.torch_dtype, device="cpu")
    if remove_prefix_in_ckpt:
        fixed_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("pipe."):
                fixed_state_dict[key] = value
            else:
                fixed_state_dict[remove_prefix_in_ckpt + key] = value
        state_dict = fixed_state_dict
    load_result = pipe.load_state_dict(state_dict, strict=False)
    if load_result.missing_keys:
        print(f"Checkpoint missing keys (first 20): {load_result.missing_keys[:20]}")
    if load_result.unexpected_keys:
        print(f"Checkpoint unexpected keys (first 20): {load_result.unexpected_keys[:20]}")


@torch.no_grad()
def infer_video_with_action_tokens(
    pipe: WanVideoPipeline,
    input_image: Image.Image,
    action_tokens,
    height: int,
    width: int,
    num_frames: int,
    seed: Optional[int],
    num_inference_steps: int,
    cfg_scale: float,
    cfg_merge: bool,
    sigma_shift: float,
    tiled: bool,
    tile_size: Tuple[int, int],
    tile_stride: Tuple[int, int],
    progress_bar: bool,
):
    pipe.scheduler.set_timesteps(num_inference_steps, denoising_strength=1.0, shift=sigma_shift)

    inputs_posi = {
        "prompt": None,
        "action_tokens": action_tokens,
        "vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_nega = {
        "negative_prompt": "",
        "action_tokens": action_tokens,
        "negative_vap_prompt": " ",
        "tea_cache_l1_thresh": None,
        "tea_cache_model_id": "",
        "num_inference_steps": num_inference_steps,
    }
    inputs_shared = {
        "input_image": input_image,
        "end_image": None,
        "input_video": None,
        "denoising_strength": 1.0,
        "control_video": None,
        "reference_image": None,
        "camera_control_direction": None,
        "camera_control_speed": 1 / 54,
        "camera_control_origin": (0, 0.532139961, 0.946026558, 0.5, 0.5, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0),
        "vace_video": None,
        "vace_video_mask": None,
        "vace_reference_image": None,
        "vace_scale": 1.0,
        "seed": seed,
        "rand_device": "cpu",
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "cfg_scale": cfg_scale,
        "cfg_merge": cfg_merge,
        "sigma_shift": sigma_shift,
        "motion_bucket_id": None,
        "longcat_video": None,
        "tiled": tiled,
        "tile_size": tile_size,
        "tile_stride": tile_stride,
        "sliding_window_size": None,
        "sliding_window_stride": None,
        "input_audio": None,
        "audio_sample_rate": 16000,
        "s2v_pose_video": None,
        "audio_embeds": None,
        "s2v_pose_latents": None,
        "motion_video": None,
        "animate_pose_video": None,
        "animate_face_video": None,
        "animate_inpaint_video": None,
        "animate_mask_video": None,
        "vap_video": None,
    }

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    progress_iter = tqdm if progress_bar else (lambda x: x)
    for progress_id, timestep in enumerate(progress_iter(pipe.scheduler.timesteps)):
        if timestep.item() < 0.875 * 1000 and pipe.dit2 is not None and not models["dit"] is pipe.dit2:
            pipe.load_models_to_device(pipe.in_iteration_models_2)
            models["dit"] = pipe.dit2
            models["vace"] = pipe.vace2

        timestep = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred_posi = pipe.model_fn(**models, **inputs_shared, **inputs_posi, timestep=timestep)
        if cfg_scale != 1.0:
            if cfg_merge:
                noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
            else:
                noise_pred_nega = pipe.model_fn(**models, **inputs_shared, **inputs_nega, timestep=timestep)
            noise_pred = noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)
        else:
            noise_pred = noise_pred_posi

        inputs_shared["latents"] = pipe.scheduler.step(noise_pred, pipe.scheduler.timesteps[progress_id], inputs_shared["latents"])
        if "first_frame_latents" in inputs_shared:
            inputs_shared["latents"][:, :, 0:1] = inputs_shared["first_frame_latents"]

    for unit in pipe.post_units:
        inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(inputs_shared["latents"], device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride)
    video = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    return video


def make_side_by_side_gif(gt_frames: List[Image.Image], pred_frames: List[Image.Image], output_path: str, fps: int):
    frame_count = min(len(gt_frames), len(pred_frames))
    merged_frames = []
    for i in range(frame_count):
        left = gt_frames[i].convert("RGB")
        right = pred_frames[i].convert("RGB")
        if right.size != left.size:
            right = right.resize(left.size, Image.BICUBIC)
        canvas = Image.new("RGB", (left.width + right.width, max(left.height, right.height)))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.width, 0))
        merged_frames.append(np.array(canvas))
    imageio.mimsave(output_path, merged_frames, duration=1 / fps)


def main():
    parser = argparse.ArgumentParser(description="Run TI2V inference with action tokens and save side-by-side GIFs.")
    parser.add_argument("--dataset_base_path", type=str, default="/scratch/zh2025/finetune_action/DiffSynth-Studio/data/lerobot_clothes_folding_25f_100k")
    parser.add_argument("--dataset_metadata_path", type=str, default="/scratch/zh2025/finetune_action/DiffSynth-Studio/data/lerobot_clothes_folding_25f_100k/metadata.csv")
    parser.add_argument("--data_file_keys", type=str, default="video,action_tokens")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--num_frames", type=int, default=25)
    parser.add_argument("--model_id_with_origin_paths", type=str, default="Wan-AI/Wan2.2-TI2V-5B:diffusion_pytorch_model*.safetensors")
    parser.add_argument("--model_paths", type=str, default='["/scratch/zh2025/DiffSynth-Studio/models/Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"]')
    parser.add_argument("--action_token_dim", type=int, default=256)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--remove_prefix_in_ckpt", type=str, default="pipe.dit.")
    parser.add_argument("--output_dir", type=str, default="./inference_gifs")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=8)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--cfg_merge", action="store_true")
    parser.add_argument("--sigma_shift", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--no_progress_bar", action="store_true")
    args = parser.parse_args()

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model_configs = parse_model_configs(args.model_paths, args.model_id_with_origin_paths)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch_dtype,
        device=device,
        model_configs=model_configs,
        tokenizer_config=None,
        action_token_dim=args.action_token_dim,
    )
    load_checkpoint_into_pipe(pipe, args.checkpoint_path, args.remove_prefix_in_ckpt)

    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=[k.strip() for k in args.data_file_keys.split(",")],
        main_data_operator=UnifiedDataset.default_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.height * args.width,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
            time_division_remainder=1,
        ),
        special_operator_map={
            "action_tokens": ToAbsolutePath(args.dataset_base_path) >> LoadTorchPickle(),
        },
    )

    os.makedirs(args.output_dir, exist_ok=True)
    total = len(dataset)
    end_index = min(args.start_index + args.max_samples, total) if args.max_samples > 0 else total

    for idx in tqdm(range(args.start_index, end_index), desc="Samples"):
        data = dataset[idx]
        gt_video = data.get("video")
        if gt_video is None:
            raise ValueError("Missing 'video' key in dataset item.")
        action_tokens = data.get("action_tokens")
        if action_tokens is None:
            raise ValueError("Missing 'action_tokens' key in dataset item.")
        if isinstance(action_tokens, str):
            action_tokens = torch.load(action_tokens, map_location="cpu")
        elif not isinstance(action_tokens, torch.Tensor):
            action_tokens = torch.tensor(action_tokens)

        width, height = gt_video[0].size
        num_frames = len(gt_video)
        pred_video = infer_video_with_action_tokens(
            pipe=pipe,
            input_image=gt_video[0],
            action_tokens=action_tokens,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=args.seed,
            num_inference_steps=args.num_inference_steps,
            cfg_scale=args.cfg_scale,
            cfg_merge=args.cfg_merge,
            sigma_shift=args.sigma_shift,
            tiled=True,
            tile_size=(30, 52),
            tile_stride=(15, 26),
            progress_bar=not args.no_progress_bar,
        )

        output_path = os.path.join(args.output_dir, f"sample_{idx:06d}.gif")
        make_side_by_side_gif(gt_video, pred_video, output_path, fps=args.fps)


if __name__ == "__main__":
    main()
