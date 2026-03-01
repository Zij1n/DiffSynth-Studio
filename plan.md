# Wan Cloth-Folding Action Conditioning Plan

## Target

Replace the current Wan raw-action sinusoidal shortcut with a Cosmos-style learned action embedding path, while keeping action injection limited to the timestep embedding path.

## Desired Semantics

- Dataset keeps returning clip-aligned actions with shape `[num_frames - 1, D_actual]`.
- User provides `action_feature_dim`.
- Model infers `action_steps = num_frames - 1`.
- Model pads or clips only the per-step action feature dimension:
  - `[B, action_steps, D_actual] -> [B, action_steps, action_feature_dim]`
- Model flattens action to:
  - `[B, action_steps * action_feature_dim]`
- Model applies a learned MLP action embedder:
  - `[B, flat_action_dim] -> [B, dim]`
- Model adds learned action embedding to the normal Wan timestep embedding.
- Model applies a post-sum timestep normalization.
- Model does not add a separate direct action-to-modulation branch.

## Implementation Steps

1. Add an action-conditioning module to `WanModel` that can be configured at construction time from runtime model config overrides, while still restoring from checkpoint metadata on reload.
2. Replace the current `normalize_wan_action` / raw sinusoidal action logic with:
   - structural action preparation
   - learned action MLP embedding
   - post-sum timestep normalization
3. Route all Wan call paths through the same conditioned timestep helper:
   - direct `WanModel.forward`
   - `model_fn_wan_video`
   - unified sequence parallel forward
4. Add training CLI/config support for `--action_feature_dim`.
5. Configure Wan action conditioning from training args before `from_pretrained()` using model-config overrides:
   - `action_steps = num_frames - 1`
   - `action_feature_dim = args.action_feature_dim`
6. Persist action-conditioning metadata in exported checkpoints so reloads remain self-describing.
7. Update the cloth-folding full-training launcher to pass `--action_feature_dim`.
8. Run shape/load/smoke verification in the `diffsynth` conda env.
9. Review the final diff, then squash all local implementation commits into one commit on top of the base branch.

## Constraints

- Do not hardcode `12`, `20`, or `240`.
- Do not pad or clip the time dimension.
- Do not add a Cosmos-style direct AdaLN action branch.
- Preserve pretrained Wan checkpoint loading.
- Keep VRAM-managed / disk-offload model construction compatible by creating action modules before wrapping.
