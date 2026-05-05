from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, Any


DIT_ARCHES: Dict[str, Dict[str, int]] = {
    # Tiny is intentionally smaller than the paper's DiT-S and is used for the 1-day patch-size sweep.
    "DiT-Tiny": {"depth": 6, "hidden_size": 192, "num_heads": 3},
    # Paper-style configs from Peebles & Xie. DiT-S/4 is the main 1-day stronger run.
    "DiT-S": {"depth": 12, "hidden_size": 384, "num_heads": 6},
    "DiT-B": {"depth": 12, "hidden_size": 768, "num_heads": 12},
    "DiT-L": {"depth": 24, "hidden_size": 1024, "num_heads": 16},
    "DiT-XL": {"depth": 28, "hidden_size": 1152, "num_heads": 16},
}


@dataclass
class DiTConfig:
    model_name: str = "DiT-S"
    input_size: int = 32          # 256x256 image -> 32x32 latent with Stable-Diffusion VAE
    in_channels: int = 4
    patch_size: int = 4
    num_classes: int = 10
    class_dropout_prob: float = 0.10
    learn_sigma: bool = False     # stable default; exact ADM learned-variance objective is omitted in this project
    depth: int = 12
    hidden_size: int = 384
    num_heads: int = 6
    mlp_ratio: float = 4.0
    conditioning: str = "adaln_zero"

    @property
    def out_channels(self) -> int:
        return self.in_channels * (2 if self.learn_sigma else 1)

    @property
    def num_tokens(self) -> int:
        return (self.input_size // self.patch_size) ** 2

    @property
    def run_id(self) -> str:
        return f"{self.model_name}-{self.patch_size}".replace("/", "-")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class TrainConfig:
    run_mode: str = "real_1day"
    train_steps: int = 10_000
    batch_size: int = 32
    grad_accum_steps: int = 2
    learning_rate: float = 1e-4
    weight_decay: float = 0.0
    ema_decay: float = 0.999
    diffusion_steps: int = 1000
    val_every: int = 500
    sample_every: int = 2_000
    save_every: int = 2_000
    log_every: int = 50
    num_workers: int = 4
    mixed_precision: bool = True
    grad_clip: float = 1.0
    seed: int = 42

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def make_dit_config(model_name: str = "DiT-S", patch_size: int = 4, **kwargs: Any) -> DiTConfig:
    if model_name not in DIT_ARCHES:
        raise ValueError(f"Unknown model_name={model_name}. Available: {sorted(DIT_ARCHES)}")
    arch = DIT_ARCHES[model_name]
    cfg = DiTConfig(model_name=model_name, patch_size=patch_size, **arch)
    for key, value in kwargs.items():
        if not hasattr(cfg, key):
            raise ValueError(f"DiTConfig has no field {key}")
        setattr(cfg, key, value)
    if cfg.input_size % cfg.patch_size != 0:
        raise ValueError(f"input_size={cfg.input_size} must be divisible by patch_size={cfg.patch_size}")
    return cfg


# The 1-day experiment plan. Tiny runs test patch-size scaling; S/4 is the stronger paper-style run.
RUNS = [
    {"model_name": "DiT-Tiny", "patch_size": 8, "target_steps": 10_000, "group": "patch_size"},
    {"model_name": "DiT-Tiny", "patch_size": 4, "target_steps": 10_000, "group": "patch_size"},
    {"model_name": "DiT-Tiny", "patch_size": 2, "target_steps": 10_000, "group": "patch_size"},
    {"model_name": "DiT-S", "patch_size": 4, "target_steps": 20_000, "group": "model_size"},
]
