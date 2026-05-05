from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
from torch import nn

from .configs import DiTConfig


def modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos: np.ndarray) -> np.ndarray:
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.0
    omega = 1.0 / 10000**omega
    out = np.einsum("m,d->md", pos.reshape(-1), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int) -> np.ndarray:
    assert embed_dim % 2 == 0
    grid_h = np.arange(grid_size, dtype=np.float64)
    grid_w = np.arange(grid_size, dtype=np.float64)
    grid = np.meshgrid(grid_w, grid_h)  # w, h
    grid = np.stack(grid, axis=0).reshape([2, 1, grid_size, grid_size])
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])
    return np.concatenate([emb_h, emb_w], axis=1)


def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=timesteps.device) / half)
    args = timesteps.float()[:, None] * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class TimestepEmbedder(nn.Module):
    def __init__(self, hidden_size: int, frequency_embedding_size: int = 256) -> None:
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return self.mlp(timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Class embedding with a learned unconditional token for classifier-free guidance."""

    def __init__(self, num_classes: int, hidden_size: int, dropout_prob: float) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.embedding_table = nn.Embedding(num_classes + 1, hidden_size)
        self.uncond_id = num_classes

    def token_drop(self, labels: torch.Tensor, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids.bool().to(labels.device)
        labels = torch.where(drop_ids, torch.full_like(labels, self.uncond_id), labels)
        return labels

    def forward(self, labels: torch.Tensor, train: bool, force_drop_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        labels = labels.long()
        if train and self.dropout_prob > 0 or force_drop_ids is not None:
            labels = self.token_drop(labels, force_drop_ids)
        return self.embedding_table(labels)


class Mlp(nn.Module):
    def __init__(self, in_features: int, hidden_features: int, out_features: Optional[int] = None) -> None:
        super().__init__()
        out_features = out_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate="tanh")
        self.fc2 = nn.Linear(hidden_features, out_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class DiTBlock(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int, mlp_ratio: float = 4.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        self.mlp = Mlp(hidden_size, mlp_hidden_dim)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size))

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + gate_msa.unsqueeze(1) * h
        h = self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        x = x + gate_mlp.unsqueeze(1) * h
        return x


class FinalLayer(nn.Module):
    def __init__(self, hidden_size: int, patch_size: int, out_channels: int) -> None:
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        return self.linear(x)


class DiT(nn.Module):
    """Diffusion Transformer for latent-space image generation.

    This is a compact PyTorch implementation of the paper's adaLN-Zero DiT block.
    It is intentionally notebook-friendly for a compute-limited term project.
    """

    def __init__(self, cfg: DiTConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_size = cfg.input_size
        self.patch_size = cfg.patch_size
        self.in_channels = cfg.in_channels
        self.out_channels = cfg.out_channels
        self.num_classes = cfg.num_classes
        self.num_tokens = cfg.num_tokens
        self.hidden_size = cfg.hidden_size
        self.x_embedder = nn.Conv2d(cfg.in_channels, cfg.hidden_size, kernel_size=cfg.patch_size, stride=cfg.patch_size)
        self.t_embedder = TimestepEmbedder(cfg.hidden_size)
        self.y_embedder = LabelEmbedder(cfg.num_classes, cfg.hidden_size, cfg.class_dropout_prob)
        self.blocks = nn.ModuleList([
            DiTBlock(cfg.hidden_size, cfg.num_heads, cfg.mlp_ratio) for _ in range(cfg.depth)
        ])
        self.final_layer = FinalLayer(cfg.hidden_size, cfg.patch_size, cfg.out_channels)
        grid_size = cfg.input_size // cfg.patch_size
        pos_embed = get_2d_sincos_pos_embed(cfg.hidden_size, grid_size)
        self.register_buffer("pos_embed", torch.from_numpy(pos_embed).float().unsqueeze(0), persistent=False)
        self.initialize_weights()

    def initialize_weights(self) -> None:
        def _basic_init(module: nn.Module) -> None:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)
        # Conv patchify weight is initialized like a linear projection.
        w = self.x_embedder.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.bias, 0)
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        # adaLN-Zero: each block starts as an identity residual block.
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    def unpatchify(self, x: torch.Tensor) -> torch.Tensor:
        c = self.out_channels
        p = self.patch_size
        h = w = self.input_size // p
        x = x.reshape(x.shape[0], h, w, p, p, c)
        x = torch.einsum("nhwpqc->nchpwq", x)
        return x.reshape(x.shape[0], c, h * p, w * p)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        force_drop_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.x_embedder(x).flatten(2).transpose(1, 2)  # B, T, D
        x = x + self.pos_embed.to(dtype=x.dtype, device=x.device)
        t_emb = self.t_embedder(t)
        y_emb = self.y_embedder(y, self.training, force_drop_ids)
        c = t_emb + y_emb
        for block in self.blocks:
            x = block(x, c)
        x = self.final_layer(x, c)
        return self.unpatchify(x)

    @torch.no_grad()
    def forward_with_cfg(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float) -> torch.Tensor:
        if cfg_scale == 1.0:
            return self.forward(x, t, y)
        x_in = torch.cat([x, x], dim=0)
        t_in = torch.cat([t, t], dim=0)
        y_uncond = torch.full_like(y, self.num_classes)
        y_in = torch.cat([y, y_uncond], dim=0)
        model_out = self.forward(x_in, t_in, y_in)
        cond, uncond = model_out.chunk(2, dim=0)
        if self.cfg.learn_sigma:
            cond_eps, cond_rest = cond[:, : self.in_channels], cond[:, self.in_channels :]
            uncond_eps, _ = uncond[:, : self.in_channels], uncond[:, self.in_channels :]
            guided_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
            return torch.cat([guided_eps, cond_rest], dim=1)
        return uncond + cfg_scale * (cond - uncond)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def make_model(cfg: DiTConfig) -> DiT:
    return DiT(cfg)
