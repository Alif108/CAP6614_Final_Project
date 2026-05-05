from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
from torchvision.utils import save_image

from .latent import decode_latents_to_images


@torch.no_grad()
def sample_latents(
    model,
    diffusion,
    labels: torch.Tensor,
    ddim_steps: int = 100,
    cfg_scale: float = 1.5,
    device: torch.device | str = "cuda",
    use_ddpm: bool = False,
    progress: bool = True,
) -> torch.Tensor:
    model.eval()
    labels = labels.to(device).long()
    shape = (labels.shape[0], model.cfg.in_channels, model.cfg.input_size, model.cfg.input_size)
    if use_ddpm:
        return diffusion.ddpm_sample_loop(model, shape, labels, cfg_scale=cfg_scale, device=device, progress=progress)
    return diffusion.ddim_sample_loop(model, shape, labels, steps=ddim_steps, cfg_scale=cfg_scale, device=device, progress=progress)


@torch.no_grad()
def sample_images(
    model,
    diffusion,
    vae,
    labels: torch.Tensor,
    ddim_steps: int = 100,
    cfg_scale: float = 1.5,
    device: torch.device | str = "cuda",
    progress: bool = True,
) -> torch.Tensor:
    latents = sample_latents(model, diffusion, labels, ddim_steps, cfg_scale, device, progress=progress)
    return decode_latents_to_images(vae, latents)


@torch.no_grad()
def save_class_grid(
    model,
    diffusion,
    vae,
    output_path: str | Path,
    num_classes: int = 10,
    samples_per_class: int = 4,
    ddim_steps: int = 100,
    cfg_scale: float = 1.5,
    device: torch.device | str = "cuda",
    progress: bool = True,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    labels = torch.arange(num_classes, device=device).repeat_interleave(samples_per_class)
    images = sample_images(model, diffusion, vae, labels, ddim_steps, cfg_scale, device, progress=progress)
    save_image(images, output_path, nrow=samples_per_class)
    return output_path


def save_tensor_images(images: torch.Tensor, output_dir: str | Path, start_index: int = 0) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(images):
        save_image(img, output_dir / f"sample_{start_index + i:06d}.png")
