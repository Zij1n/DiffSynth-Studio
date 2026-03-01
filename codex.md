# Codex Handoff

## Current State

- Goal: train Wan on Cosmos cloth-folding with action conditioning injected only through the Wan timestep embedding path.
- Current target model: full fine-tuning of `Wan2.2-TI2V-5B`, not LoRA.
- Active Slurm job: `3045802`
- Current job state at handoff: `PENDING` with reason `QOSGrpGRES`

## Code Changes Landed

- `063e324` `Add Wan cloth-folding action conditioning path`
  - Added Wan-side cloth-folding dataset at [diffsynth/core/data/action_conditioned_dataset.py](/scratch/zh2025/finetune_action/action_cond/diffsynth/core/data/action_conditioned_dataset.py)
  - Wired dataset selection into [examples/wanvideo/model_training/train.py](/scratch/zh2025/finetune_action/action_cond/examples/wanvideo/model_training/train.py)
  - Plumbed `action` through [diffsynth/pipelines/wan_video.py](/scratch/zh2025/finetune_action/action_cond/diffsynth/pipelines/wan_video.py), [diffsynth/models/wan_video_dit.py](/scratch/zh2025/finetune_action/action_cond/diffsynth/models/wan_video_dit.py), and [diffsynth/utils/xfuser/xdit_context_parallel.py](/scratch/zh2025/finetune_action/action_cond/diffsynth/utils/xfuser/xdit_context_parallel.py)

- `47f0e24` `Fix Wan cloth-folding dataset edge cases`
  - Fixed action-only annotations to treat `len(action) + 1` as available frame count when `state` is absent
  - Replaced recursive invalid-sample retry with bounded iteration and explicit failure
  - Rejected `sequence_interval != 1` for precomputed cloth-folding actions instead of silently misaligning actions to skipped-frame clips

- `4047f62` `Switch cloth-folding launcher to full 5B training`
  - Removed mistaken LoRA cloth-folding launcher path
  - Added full-training launcher at [examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sh](/scratch/zh2025/finetune_action/action_cond/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sh)
  - Added Slurm wrapper at [examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sbatch](/scratch/zh2025/finetune_action/action_cond/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sbatch)
  - Added 4-GPU accelerate config at [examples/wanvideo/model_training/full/accelerate_config_5B_4gpu.yaml](/scratch/zh2025/finetune_action/action_cond/examples/wanvideo/model_training/full/accelerate_config_5B_4gpu.yaml)

## Validation Already Done

- Real cloth-folding dataset load in `conda activate diffsynth`
  - Dataset returns `13` frames and action shape `(12, 20)`
- Synthetic edge-case checks
  - Action-only annotation format now works
  - Fully broken sample set now raises a single `RuntimeError` instead of recursing forever
  - `sequence_interval=2` now fails immediately for precomputed actions
- Slurm validation
  - `sbatch --test-only examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sbatch` succeeded

## Active Training Process

- Launcher script:
  - [examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sh](/scratch/zh2025/finetune_action/action_cond/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sh)
- Slurm script:
  - [examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sbatch](/scratch/zh2025/finetune_action/action_cond/examples/wanvideo/model_training/full/Wan2.2-TI2V-5B-Cloth-Folding.sbatch)
- Submitted job:
  - `3045802`
- Scheduler details at handoff:
  - Account: `torch_pr_147_courant`
  - Constraint: `l40s`
  - GPUs: `4`
  - CPUs: `128`
  - Memory: `256G`
  - Time limit: `04:00:00`

## What To Check When Coming Back

- Slurm status:
  - `scontrol show job 3045802`
- Stdout log:
  - `/scratch/zh2025/finetune_action/action_cond/logs/wan_cf_5b_full_3045802.out`
- Stderr log:
  - `/scratch/zh2025/finetune_action/action_cond/logs/wan_cf_5b_full_3045802.err`
- Live tail commands:
  - `tail -f /scratch/zh2025/finetune_action/action_cond/logs/wan_cf_5b_full_3045802.out`
  - `tail -f /scratch/zh2025/finetune_action/action_cond/logs/wan_cf_5b_full_3045802.err`

## Older Job Worth Remembering

- Cancelled LoRA smoke job:
  - `3045446`
- Existing logs from that cancelled job:
  - `/scratch/zh2025/finetune_action/action_cond/logs/wan_cf_5b_smoke_3045446.out`
  - `/scratch/zh2025/finetune_action/action_cond/logs/wan_cf_5b_smoke_3045446.err`

## Current Worktree Note

- `logs/` is kept in the tree via [logs/.gitignore](/scratch/zh2025/finetune_action/action_cond/logs/.gitignore) so Slurm can open stdout/stderr paths on a clean checkout.
- Actual log files under `logs/` remain untracked.
