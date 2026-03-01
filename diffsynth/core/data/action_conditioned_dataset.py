import json
import os
import random
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

import imageio
import numpy as np
import torch
from PIL import Image


class WanActionConditionedDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        base_path,
        split="train",
        repeat=1,
        num_frames=13,
        sequence_interval=1,
        frame_processor=lambda x: x,
        prompt_fallback="cloth folding",
        camera_id=0,
        use_precomputed_action=True,
        precomputed_action_key="action",
        precomputed_action_scale=None,
        val_start_frame_interval=1,
        max_data_items=None,
    ):
        self.base_path = base_path
        self.split = split
        self.repeat = repeat
        self.sequence_length = num_frames
        self.sequence_interval = sequence_interval
        self.frame_processor = frame_processor
        self.prompt_fallback = prompt_fallback
        self.camera_id = camera_id
        self.use_precomputed_action = use_precomputed_action
        self.precomputed_action_key = precomputed_action_key
        self.precomputed_action_scale = precomputed_action_scale
        self.val_start_frame_interval = val_start_frame_interval
        self.max_data_items = max_data_items
        self.load_from_cache = False
        if self.use_precomputed_action and self.sequence_interval != 1:
            raise NotImplementedError(
                "WanActionConditionedDataset currently supports sequence_interval=1 for precomputed actions."
            )

        self.annotation_dir = os.path.join(base_path, "annotation", split)
        if not os.path.isdir(self.annotation_dir):
            raise FileNotFoundError(f"Missing annotation directory: {self.annotation_dir}")

        self.start_frame_interval = 1 if split == "train" else val_start_frame_interval
        self.ann_files = self._init_annotations()
        self.samples = self._init_sequences(self.ann_files)
        self.samples = sorted(self.samples, key=lambda sample: (sample["annotation_file"], sample["frame_ids"][0]))
        if len(self.samples) == 0:
            raise RuntimeError(
                f"No valid action-conditioned samples found under {self.annotation_dir}. "
                "Check num_frames, sequence_interval, and annotation contents."
            )

    def _init_annotations(self):
        ann_files = [
            os.path.join(self.annotation_dir, file_name)
            for file_name in os.listdir(self.annotation_dir)
            if file_name.endswith(".json")
        ]
        ann_files.sort()
        return ann_files

    def _build_samples_from_annotation(self, annotation_file):
        with open(annotation_file, "r") as file:
            annotation = json.load(file)

        num_available_frames = self._get_num_available_frames(annotation)
        samples = []
        for start_frame in range(0, num_available_frames, self.start_frame_interval):
            frame_ids = []
            current_frame = start_frame
            while current_frame < num_available_frames and len(frame_ids) < self.sequence_length:
                frame_ids.append(current_frame)
                current_frame += self.sequence_interval
            if len(frame_ids) == self.sequence_length:
                samples.append(
                    {
                        "annotation_file": annotation_file,
                        "frame_ids": frame_ids,
                    }
                )
        return samples

    def _get_num_available_frames(self, annotation):
        states = annotation.get("state")
        if states is not None:
            return len(states)

        if not self.use_precomputed_action:
            raise KeyError("Annotation is missing 'state' and precomputed actions are disabled.")
        if self.precomputed_action_key not in annotation:
            raise KeyError(
                f"Annotation is missing both 'state' and precomputed action key '{self.precomputed_action_key}'."
            )

        actions = annotation[self.precomputed_action_key]
        return len(actions) + 1

    def _init_sequences(self, ann_files):
        if len(ann_files) == 0:
            return []
        samples = []
        max_workers = min(32, len(ann_files))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(self._build_samples_from_annotation, ann_file) for ann_file in ann_files]
            for future in as_completed(futures):
                samples.extend(future.result())
        return samples

    def _select_video_path(self, annotation):
        videos = annotation.get("videos", [])
        if len(videos) == 0:
            raise KeyError("Annotation is missing the 'videos' field.")
        if self.camera_id is None:
            video_index = random.randrange(len(videos))
        else:
            video_index = self.camera_id
        if video_index >= len(videos):
            raise IndexError(f"Camera index {video_index} is out of range for annotation videos.")
        return os.path.join(self.base_path, videos[video_index]["video_path"])

    def _load_video_frames(self, video_path, frame_ids):
        reader = imageio.get_reader(video_path)
        frames = []
        try:
            for frame_id in frame_ids:
                frame = Image.fromarray(reader.get_data(int(frame_id))).convert("RGB")
                frames.append(self.frame_processor(frame))
        finally:
            reader.close()
        return frames

    def _load_action(self, annotation, frame_ids):
        if not self.use_precomputed_action:
            raise NotImplementedError("WanActionConditionedDataset currently supports only precomputed actions.")
        if self.precomputed_action_key not in annotation:
            raise KeyError(
                f"Missing precomputed action key '{self.precomputed_action_key}' in annotation."
            )
        actions = np.asarray(annotation[self.precomputed_action_key], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[:, None]

        action_indices = np.asarray(frame_ids[:-1], dtype=np.int64)
        if len(action_indices) > 0 and actions.shape[0] <= int(action_indices.max()):
            raise IndexError(
                f"Precomputed action length {actions.shape[0]} is too short for index {int(action_indices.max())}."
            )

        action_tensor = torch.from_numpy(actions[action_indices].copy())
        if self.precomputed_action_scale is not None:
            scale = torch.as_tensor(self.precomputed_action_scale, dtype=action_tensor.dtype)
            action_tensor = action_tensor * scale
        return action_tensor

    def _resolve_prompt(self, annotation):
        for text in annotation.get("texts", []):
            if isinstance(text, str) and text.strip() != "":
                return text.strip()
        task = annotation.get("task", "")
        if isinstance(task, str) and task.strip() not in ("", "robot_trajectory_prediction"):
            return task.replace("_", " ").strip()
        return self.prompt_fallback

    def __getitem__(self, index):
        if len(self.samples) == 0:
            raise RuntimeError("WanActionConditionedDataset has no samples to load.")

        num_samples = len(self.samples)
        start_index = index % num_samples
        sample_order = list(range(start_index, num_samples)) + list(range(0, start_index))
        if len(sample_order) > 1:
            tail = sample_order[1:]
            random.shuffle(tail)
            sample_order[1:] = tail

        errors = []
        for sample_index in sample_order:
            sample = self.samples[sample_index]
            try:
                with open(sample["annotation_file"], "r") as file:
                    annotation = json.load(file)

                video_path = self._select_video_path(annotation)
                frames = self._load_video_frames(video_path, sample["frame_ids"])
                action = self._load_action(annotation, sample["frame_ids"]).float()

                key = annotation.get("episode_id")
                if key is None:
                    key = os.path.splitext(os.path.basename(sample["annotation_file"]))[0]

                return {
                    "prompt": self._resolve_prompt(annotation),
                    "video": frames,
                    "action": action,
                    "annotation_file": sample["annotation_file"],
                    "video_path": video_path,
                    "frame_ids": sample["frame_ids"],
                    "__key__": str(key),
                }
            except Exception as error:
                warnings.warn(
                    f"Invalid action-conditioned sample {sample['annotation_file']} skipped: {error}",
                    stacklevel=2,
                )
                errors.append(f"{sample['annotation_file']}: {error}")

        raise RuntimeError(
            "Failed to load any action-conditioned sample. "
            + " | ".join(errors[:5])
        )

    def __len__(self):
        if self.max_data_items is not None:
            return self.max_data_items
        return len(self.samples) * self.repeat
