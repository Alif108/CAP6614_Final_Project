from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional

import torch
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from tqdm.auto import tqdm

VAE_SCALE = 0.18215


def load_vae(device: torch.device | str = "cuda", model_id: str = "stabilityai/sd-vae-ft-mse"):
    try:
        from diffusers import AutoencoderKL
    except ImportError as exc:
        raise ImportError("diffusers is required for latent DiT. Install it with `%pip install diffusers transformers accelerate safetensors`." ) from exc
    vae = AutoencoderKL.from_pretrained(model_id)
    vae.to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad_(False)
    return vae


@torch.no_grad()
def encode_images_to_latents(vae, images: torch.Tensor, sample: bool = True) -> torch.Tensor:
    posterior = vae.encode(images).latent_dist
    latents = posterior.sample() if sample else posterior.mode()
    return latents * VAE_SCALE


@torch.no_grad()
def decode_latents_to_images(vae, latents: torch.Tensor) -> torch.Tensor:
    latents = latents / VAE_SCALE
    images = vae.decode(latents).sample
    images = (images.clamp(-1, 1) + 1) / 2
    return images.clamp(0, 1)


@torch.no_grad()
def cache_latents(
    vae,
    dataset,
    output_dir: str | Path,
    batch_size: int = 32,
    num_workers: int = 4,
    device: torch.device | str = "cuda",
    shard_size: int = 512,
    dtype: torch.dtype = torch.float16,
    overwrite: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(output_dir.glob("shard_*.pt"))
    if existing and not overwrite:
        print(f"Found {len(existing)} latent shards in {output_dir}; set overwrite=True to recache.")
        return
    for old in existing:
        old.unlink()

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    shard_latents = []
    shard_labels = []
    shard_paths = []
    shard_idx = 0
    sample_offset = 0
    for images, labels in tqdm(loader, desc=f"Caching latents -> {output_dir.name}"):
        images = images.to(device, non_blocking=True)
        labels = labels.cpu()
        latents = encode_images_to_latents(vae, images).detach().cpu().to(dtype)
        for i in range(latents.shape[0]):
            shard_latents.append(latents[i])
            shard_labels.append(labels[i])
            path = dataset.samples[sample_offset + i][0] if hasattr(dataset, "samples") else ""
            shard_paths.append(path)
            if len(shard_latents) >= shard_size:
                _save_shard(output_dir, shard_idx, shard_latents, shard_labels, shard_paths)
                shard_idx += 1
                shard_latents, shard_labels, shard_paths = [], [], []
        sample_offset += latents.shape[0]
    if shard_latents:
        _save_shard(output_dir, shard_idx, shard_latents, shard_labels, shard_paths)


def _save_shard(output_dir: Path, shard_idx: int, latents, labels, paths) -> None:
    payload = {
        "latents": torch.stack(latents, dim=0),
        "labels": torch.stack(labels, dim=0).long(),
        "paths": list(paths),
    }
    torch.save(payload, output_dir / f"shard_{shard_idx:05d}.pt")


@torch.no_grad()
def save_reconstruction_grid(vae, dataset, output_path: str | Path, device: torch.device | str = "cuda", n: int = 16) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    images = torch.stack([dataset[i][0] for i in range(min(n, len(dataset)))], dim=0).to(device)
    recons = decode_latents_to_images(vae, encode_images_to_latents(vae, images, sample=False))
    originals = (images.clamp(-1, 1) + 1) / 2
    grid = torch.cat([originals, recons], dim=0)
    save_image(grid, output_path, nrow=min(n, 8))
