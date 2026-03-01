import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Optional
from einops import rearrange
from .wan_video_camera_controller import SimpleAdapter

try:
    import flash_attn_interface
    FLASH_ATTN_3_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_3_AVAILABLE = False

try:
    import flash_attn
    FLASH_ATTN_2_AVAILABLE = True
except ModuleNotFoundError:
    FLASH_ATTN_2_AVAILABLE = False

try:
    from sageattention import sageattn
    SAGE_ATTN_AVAILABLE = True
except ModuleNotFoundError:
    SAGE_ATTN_AVAILABLE = False
    
    
def flash_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, num_heads: int, compatibility_mode=False):
    if compatibility_mode:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_3_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn_interface.flash_attn_func(q, k, v)
        if isinstance(x,tuple):
            x = x[0]
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif FLASH_ATTN_2_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b s n d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b s n d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b s n d", n=num_heads)
        x = flash_attn.flash_attn_func(q, k, v)
        x = rearrange(x, "b s n d -> b s (n d)", n=num_heads)
    elif SAGE_ATTN_AVAILABLE:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = sageattn(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    else:
        q = rearrange(q, "b s (n d) -> b n s d", n=num_heads)
        k = rearrange(k, "b s (n d) -> b n s d", n=num_heads)
        v = rearrange(v, "b s (n d) -> b n s d", n=num_heads)
        x = F.scaled_dot_product_attention(q, k, v)
        x = rearrange(x, "b n s d -> b s (n d)", n=num_heads)
    return x


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor):
    return (x * (1 + scale) + shift)


def sinusoidal_embedding_1d(dim, position):
    sinusoid = torch.outer(position.type(torch.float64), torch.pow(
        10000, -torch.arange(dim//2, dtype=torch.float64, device=position.device).div(dim//2)))
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x.to(position.dtype)


def build_wan_time_embedding(
    time_embedding: nn.Module,
    freq_dim: int,
    timestep: torch.Tensor,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    expand_to_sequence: bool = False,
):
    device = timestep.device if device is None else device
    if dtype is None:
        dtype = timestep.dtype if torch.is_floating_point(timestep) else torch.float32

    timestep = timestep.to(device=device, dtype=dtype)
    timestep_freqs = sinusoidal_embedding_1d(freq_dim, timestep)
    if expand_to_sequence:
        timestep_freqs = timestep_freqs.unsqueeze(0)
    return time_embedding(timestep_freqs)


ACTION_STEPS_STATE_KEY = "_action_conditioning.action_steps"
ACTION_FEATURE_DIM_STATE_KEY = "_action_conditioning.action_feature_dim"
ACTION_STATE_PREFIXES = ("action_embedder.", "action_t_embedding_norm.")


def precompute_freqs_cis_3d(dim: int, end: int = 1024, theta: float = 10000.0):
    # 3d rope precompute
    f_freqs_cis = precompute_freqs_cis(dim - 2 * (dim // 3), end, theta)
    h_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    w_freqs_cis = precompute_freqs_cis(dim // 3, end, theta)
    return f_freqs_cis, h_freqs_cis, w_freqs_cis


def precompute_freqs_cis(dim: int, end: int = 1024, theta: float = 10000.0):
    # 1d rope precompute
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)
                   [: (dim // 2)].double() / dim))
    freqs = torch.outer(torch.arange(end, device=freqs.device), freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def rope_apply(x, freqs, num_heads):
    x = rearrange(x, "b s (n d) -> b s n d", n=num_heads)
    x_out = torch.view_as_complex(x.to(torch.float64).reshape(
        x.shape[0], x.shape[1], x.shape[2], -1, 2))
    freqs = freqs.to(torch.complex64) if freqs.device == "npu" else freqs
    x_out = torch.view_as_real(x_out * freqs).flatten(2)
    return x_out.to(x.dtype)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        dtype = x.dtype
        return self.norm(x.float()).to(dtype) * self.weight


class AttentionModule(nn.Module):
    def __init__(self, num_heads):
        super().__init__()
        self.num_heads = num_heads
        
    def forward(self, q, k, v):
        x = flash_attention(q=q, k=k, v=v, num_heads=self.num_heads)
        return x


class SelfAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x, freqs):
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(x))
        v = self.v(x)
        q = rope_apply(q, freqs, self.num_heads)
        k = rope_apply(k, freqs, self.num_heads)
        x = self.attn(q, k, v)
        return self.o(x)


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, eps: float = 1e-6, has_image_input: bool = False):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = RMSNorm(dim, eps=eps)
        self.norm_k = RMSNorm(dim, eps=eps)
        self.has_image_input = has_image_input
        if has_image_input:
            self.k_img = nn.Linear(dim, dim)
            self.v_img = nn.Linear(dim, dim)
            self.norm_k_img = RMSNorm(dim, eps=eps)
            
        self.attn = AttentionModule(self.num_heads)

    def forward(self, x: torch.Tensor, y: torch.Tensor):
        if self.has_image_input:
            img = y[:, :257]
            ctx = y[:, 257:]
        else:
            ctx = y
        q = self.norm_q(self.q(x))
        k = self.norm_k(self.k(ctx))
        v = self.v(ctx)
        x = self.attn(q, k, v)
        if self.has_image_input:
            k_img = self.norm_k_img(self.k_img(img))
            v_img = self.v_img(img)
            y = flash_attention(q, k_img, v_img, num_heads=self.num_heads)
            x = x + y
        return self.o(x)


class GateModule(nn.Module):
    def __init__(self,):
        super().__init__()

    def forward(self, x, gate, residual):
        return x + gate * residual

class DiTBlock(nn.Module):
    def __init__(self, has_image_input: bool, dim: int, num_heads: int, ffn_dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.ffn_dim = ffn_dim

        self.self_attn = SelfAttention(dim, num_heads, eps)
        self.cross_attn = CrossAttention(
            dim, num_heads, eps, has_image_input=has_image_input)
        self.norm1 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.norm3 = nn.LayerNorm(dim, eps=eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(
            approximate='tanh'), nn.Linear(ffn_dim, dim))
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)
        self.gate = GateModule()

    def forward(self, x, context, t_mod, freqs):
        has_seq = len(t_mod.shape) == 4
        chunk_dim = 2 if has_seq else 1
        # msa: multi-head self-attention  mlp: multi-layer perceptron
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
        if has_seq:
            shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
                shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
                shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
            )
        input_x = modulate(self.norm1(x), shift_msa, scale_msa)
        x = self.gate(x, gate_msa, self.self_attn(input_x, freqs))
        x = x + self.cross_attn(self.norm3(x), context)
        input_x = modulate(self.norm2(x), shift_mlp, scale_mlp)
        x = self.gate(x, gate_mlp, self.ffn(input_x))
        return x


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, has_pos_emb=False):
        super().__init__()
        self.proj = torch.nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, in_dim),
            nn.GELU(),
            nn.Linear(in_dim, out_dim),
            nn.LayerNorm(out_dim)
        )
        self.has_pos_emb = has_pos_emb
        if has_pos_emb:
            self.emb_pos = torch.nn.Parameter(torch.zeros((1, 514, 1280)))

    def forward(self, x):
        if self.has_pos_emb:
            x = x + self.emb_pos.to(dtype=x.dtype, device=x.device)
        return self.proj(x)


class ActionEmbeddingMLP(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.activation = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        return x


class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        super().__init__()
        self.dim = dim
        self.patch_size = patch_size
        self.norm = nn.LayerNorm(dim, eps=eps, elementwise_affine=False)
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, t_mod):
        if len(t_mod.shape) == 3:
            shift, scale = (self.modulation.unsqueeze(0).to(dtype=t_mod.dtype, device=t_mod.device) + t_mod.unsqueeze(2)).chunk(2, dim=2)
            x = (self.head(self.norm(x) * (1 + scale.squeeze(2)) + shift.squeeze(2)))
        else:
            shift, scale = (self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(2, dim=1)
            x = (self.head(self.norm(x) * (1 + scale) + shift))
        return x


class WanModel(torch.nn.Module):
    def __init__(
        self,
        dim: int,
        in_dim: int,
        ffn_dim: int,
        out_dim: int,
        text_dim: int,
        freq_dim: int,
        eps: float,
        patch_size: Tuple[int, int, int],
        num_heads: int,
        num_layers: int,
        has_image_input: bool,
        has_image_pos_emb: bool = False,
        has_ref_conv: bool = False,
        add_control_adapter: bool = False,
        in_dim_control_adapter: int = 24,
        seperated_timestep: bool = False,
        require_vae_embedding: bool = True,
        require_clip_embedding: bool = True,
        fuse_vae_embedding_in_latents: bool = False,
        action_steps: Optional[int] = None,
        action_feature_dim: Optional[int] = None,
    ):
        super().__init__()
        self.dim = dim
        self.in_dim = in_dim
        self.freq_dim = freq_dim
        self.has_image_input = has_image_input
        self.patch_size = patch_size
        self.seperated_timestep = seperated_timestep
        self.require_vae_embedding = require_vae_embedding
        self.require_clip_embedding = require_clip_embedding
        self.fuse_vae_embedding_in_latents = fuse_vae_embedding_in_latents
        self.action_steps: Optional[int] = None
        self.action_feature_dim: Optional[int] = None
        self.action_flat_dim: Optional[int] = None
        self.action_embedder: Optional[ActionEmbeddingMLP] = None
        self.action_t_embedding_norm: Optional[RMSNorm] = None

        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size)
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim),
            nn.GELU(approximate='tanh'),
            nn.Linear(dim, dim)
        )
        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(
            nn.SiLU(), nn.Linear(dim, dim * 6))
        self.blocks = nn.ModuleList([
            DiTBlock(has_image_input, dim, num_heads, ffn_dim, eps)
            for _ in range(num_layers)
        ])
        self.head = Head(dim, out_dim, patch_size, eps)
        head_dim = dim // num_heads
        self.freqs = precompute_freqs_cis_3d(head_dim)

        if has_image_input:
            self.img_emb = MLP(1280, dim, has_pos_emb=has_image_pos_emb)  # clip_feature_dim = 1280
        if has_ref_conv:
            self.ref_conv = nn.Conv2d(16, dim, kernel_size=(2, 2), stride=(2, 2))
        self.has_image_pos_emb = has_image_pos_emb
        self.has_ref_conv = has_ref_conv
        if add_control_adapter:
            self.control_adapter = SimpleAdapter(in_dim_control_adapter, dim, kernel_size=patch_size[1:], stride=patch_size[1:])
        else:
            self.control_adapter = None
        if (action_steps is None) != (action_feature_dim is None):
            raise ValueError("action_steps and action_feature_dim must be provided together.")
        if action_steps is not None and action_feature_dim is not None:
            self.configure_action_conditioning(action_steps, action_feature_dim)

    def configure_action_conditioning(self, action_steps: int, action_feature_dim: int):
        if action_steps <= 0:
            raise ValueError(f"action_steps must be positive, got {action_steps}")
        if action_feature_dim <= 0:
            raise ValueError(f"action_feature_dim must be positive, got {action_feature_dim}")

        action_flat_dim = action_steps * action_feature_dim
        if (
            self.action_embedder is not None
            and self.action_flat_dim == action_flat_dim
        ):
            self.action_steps = action_steps
            self.action_feature_dim = action_feature_dim
            return

        self.action_steps = action_steps
        self.action_feature_dim = action_feature_dim
        self.action_flat_dim = action_flat_dim

        device = self.time_embedding[0].weight.device
        dtype = self.time_embedding[0].weight.dtype
        self.action_embedder = ActionEmbeddingMLP(
            in_features=action_flat_dim,
            hidden_features=self.dim * 4,
            out_features=self.dim,
        ).to(device=device, dtype=dtype)
        self.action_t_embedding_norm = RMSNorm(self.dim, eps=1e-6).to(device=device, dtype=dtype)

    def initialize_missing_action_conditioning(self):
        if self.action_steps is None or self.action_feature_dim is None:
            raise RuntimeError("Cannot initialize action conditioning without action_steps and action_feature_dim.")
        if self.action_embedder is None or self.action_t_embedding_norm is None:
            self.configure_action_conditioning(self.action_steps, self.action_feature_dim)
        if not any(param.is_meta for param in self.action_embedder.parameters()) and not any(param.is_meta for param in self.action_t_embedding_norm.parameters()):
            return

        device = self.time_embedding[0].weight.device
        dtype = self.time_embedding[0].weight.dtype
        action_steps = self.action_steps
        action_feature_dim = self.action_feature_dim
        self.action_embedder = None
        self.action_t_embedding_norm = None
        self.action_flat_dim = None
        self.configure_action_conditioning(action_steps, action_feature_dim)

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        if len(args) > 3:
            raise TypeError(f"state_dict() received too many positional arguments: expected at most 3, got {len(args)}")
        if len(args) >= 1:
            destination = args[0]
        if len(args) >= 2:
            prefix = args[1]
        if len(args) >= 3:
            keep_vars = args[2]
        state_dict = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.action_steps is not None and self.action_feature_dim is not None:
            state_dict[prefix + ACTION_STEPS_STATE_KEY] = torch.tensor(self.action_steps, dtype=torch.int64)
            state_dict[prefix + ACTION_FEATURE_DIM_STATE_KEY] = torch.tensor(self.action_feature_dim, dtype=torch.int64)
        return state_dict

    def extra_trainable_state_dict_keys(self):
        if self.action_steps is None or self.action_feature_dim is None:
            return []
        return [ACTION_STEPS_STATE_KEY, ACTION_FEATURE_DIM_STATE_KEY]

    def configure_from_state_dict_metadata(self, state_dict):
        has_action_steps = ACTION_STEPS_STATE_KEY in state_dict
        has_action_feature_dim = ACTION_FEATURE_DIM_STATE_KEY in state_dict
        if not has_action_steps and not has_action_feature_dim:
            return
        if has_action_steps != has_action_feature_dim:
            raise RuntimeError("Action-conditioning checkpoint metadata is incomplete.")
        self.configure_action_conditioning(
            int(state_dict[ACTION_STEPS_STATE_KEY].item()),
            int(state_dict[ACTION_FEATURE_DIM_STATE_KEY].item()),
        )

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        state_dict = dict(state_dict)
        self.configure_from_state_dict_metadata(state_dict)
        action_steps = state_dict.pop(ACTION_STEPS_STATE_KEY, None)
        action_feature_dim = state_dict.pop(ACTION_FEATURE_DIM_STATE_KEY, None)
        if action_steps is not None or action_feature_dim is not None:
            assert action_steps is not None and action_feature_dim is not None
        has_action_weights = any(key.startswith(ACTION_STATE_PREFIXES) for key in state_dict)
        if has_action_weights and self.action_embedder is None:
            raise RuntimeError(
                "Action-conditioning weights were found, but action_steps/action_feature_dim metadata is missing. "
                "Reload with checkpoint metadata or provide constructor overrides before loading."
            )
        allow_missing_action_weights = self.action_embedder is not None and not has_action_weights
        incompatible_keys = super().load_state_dict(state_dict, strict=False, assign=assign)
        if allow_missing_action_weights:
            self.initialize_missing_action_conditioning()
        if strict:
            missing_keys = list(incompatible_keys.missing_keys)
            unexpected_keys = list(incompatible_keys.unexpected_keys)
            if allow_missing_action_weights:
                missing_keys = [
                    key for key in missing_keys
                    if not key.startswith(ACTION_STATE_PREFIXES)
                ]
            if missing_keys or unexpected_keys:
                error_messages = []
                if unexpected_keys:
                    error_messages.append(
                        "Unexpected key(s) in state_dict: {}.".format(
                            ", ".join(f'"{key}"' for key in unexpected_keys)
                        )
                    )
                if missing_keys:
                    error_messages.append(
                        "Missing key(s) in state_dict: {}.".format(
                            ", ".join(f'"{key}"' for key in missing_keys)
                        )
                    )
                raise RuntimeError(
                    f"Error(s) in loading state_dict for {self.__class__.__name__}:\n\t" + "\n\t".join(error_messages)
                )
        return incompatible_keys

    def _prepare_action_input(
        self,
        action: Optional[torch.Tensor],
        batch_size: int,
        device: torch.device,
    ):
        if action is None:
            return None
        if self.action_embedder is None or self.action_steps is None or self.action_feature_dim is None or self.action_flat_dim is None:
            raise RuntimeError(
                "Wan action conditioning is not configured. "
                "Call configure_action_conditioning(action_steps, action_feature_dim) before using action inputs."
            )

        if not torch.is_tensor(action):
            action = torch.as_tensor(action, device=device)
        else:
            action = action.to(device=device)
        if action.numel() == 0:
            return None

        if action.ndim == 1:
            if action.shape[0] != self.action_flat_dim:
                raise ValueError(
                    f"Expected flattened action dim {self.action_flat_dim}, got {tuple(action.shape)}"
                )
            action = action.reshape(1, self.action_flat_dim)
        elif action.ndim == 2:
            if action.shape[0] == self.action_steps:
                action = action.unsqueeze(0)
            elif action.shape[1] != self.action_flat_dim:
                raise ValueError(
                    f"Expected action shape [T, D] with T={self.action_steps} or flattened dim {self.action_flat_dim}, "
                    f"got {tuple(action.shape)}"
                )
        elif action.ndim != 3:
            raise ValueError(f"Expected action to have 1, 2, or 3 dims, got shape {tuple(action.shape)}")

        if action.ndim == 3:
            if action.shape[1] != self.action_steps:
                raise ValueError(
                    f"Expected action time dimension {self.action_steps}, got {action.shape[1]}"
                )
            if action.shape[2] < self.action_feature_dim:
                action = F.pad(action, (0, self.action_feature_dim - action.shape[2]))
            elif action.shape[2] > self.action_feature_dim:
                action = action[:, :, :self.action_feature_dim]
            action = action.reshape(action.shape[0], self.action_flat_dim)

        if action.shape[0] == 1 and batch_size != 1:
            action = action.expand(batch_size, -1)
        elif action.shape[0] != batch_size:
            raise ValueError(f"Expected action batch dimension 1 or {batch_size}, got shape {tuple(action.shape)}")

        return action.contiguous()

    def build_conditioned_time_embedding(
        self,
        timestep: torch.Tensor,
        action: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        expand_to_sequence: bool = False,
    ):
        t = build_wan_time_embedding(
            self.time_embedding,
            self.freq_dim,
            timestep,
            device=device,
            dtype=dtype,
            expand_to_sequence=expand_to_sequence,
        )
        if action is None:
            return t

        device = t.device if device is None else device
        action_input = self._prepare_action_input(action, batch_size=t.shape[0], device=device)
        if action_input is None:
            return t

        action_embed = self.action_embedder(action_input.to(dtype=t.dtype))
        if t.ndim == 3:
            action_embed = action_embed.unsqueeze(1).expand(-1, t.shape[1], -1)
        return self.action_t_embedding_norm(t + action_embed.to(dtype=t.dtype))

    def patchify(self, x: torch.Tensor, control_camera_latents_input: Optional[torch.Tensor] = None):
        x = self.patch_embedding(x)
        if self.control_adapter is not None and control_camera_latents_input is not None:
            y_camera = self.control_adapter(control_camera_latents_input)
            x = [u + v for u, v in zip(x, y_camera)]
            x = x[0].unsqueeze(0)
        return x

    def unpatchify(self, x: torch.Tensor, grid_size: torch.Tensor):
        return rearrange(
            x, 'b (f h w) (x y z c) -> b c (f x) (h y) (w z)',
            f=grid_size[0], h=grid_size[1], w=grid_size[2], 
            x=self.patch_size[0], y=self.patch_size[1], z=self.patch_size[2]
        )

    def forward(self,
                x: torch.Tensor,
                timestep: torch.Tensor,
                context: torch.Tensor,
                clip_feature: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None,
                action: Optional[torch.Tensor] = None,
                use_gradient_checkpointing: bool = False,
                use_gradient_checkpointing_offload: bool = False,
                **kwargs,
                ):
        t = self.build_conditioned_time_embedding(
            timestep,
            action=action,
            device=x.device,
            dtype=x.dtype,
        )
        t_mod = self.time_projection(t).unflatten(1, (6, self.dim))
        context = self.text_embedding(context)
        
        if self.has_image_input:
            x = torch.cat([x, y], dim=1)  # (b, c_x + c_y, f, h, w)
            clip_embdding = self.img_emb(clip_feature)
            context = torch.cat([clip_embdding, context], dim=1)
        
        x, (f, h, w) = self.patchify(x)
        
        freqs = torch.cat([
            self.freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
            self.freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
            self.freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1)
        ], dim=-1).reshape(f * h * w, 1, -1).to(x.device)
        
        def create_custom_forward(module):
            def custom_forward(*inputs):
                return module(*inputs)
            return custom_forward

        for block in self.blocks:
            if self.training and use_gradient_checkpointing:
                if use_gradient_checkpointing_offload:
                    with torch.autograd.graph.save_on_cpu():
                        x = torch.utils.checkpoint.checkpoint(
                            create_custom_forward(block),
                            x, context, t_mod, freqs,
                            use_reentrant=False,
                        )
                else:
                    x = torch.utils.checkpoint.checkpoint(
                        create_custom_forward(block),
                        x, context, t_mod, freqs,
                        use_reentrant=False,
                    )
            else:
                x = block(x, context, t_mod, freqs)

        x = self.head(x, t)
        x = self.unpatchify(x, (f, h, w))
        return x
