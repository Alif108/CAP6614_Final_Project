from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


IMAGENETTE_URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
IMAGENETTE_CLASSES = {
    "n01440764": "tench",
    "n02102040": "English springer",
    "n02979186": "cassette player",
    "n03000684": "chain saw",
    "n03028079": "church",
    "n03394916": "French horn",
    "n03417042": "garbage truck",
    "n03425413": "gas pump",
    "n03445777": "golf ball",
    "n03888257": "parachute",
}


def image_transform(image_size: int = 256) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def eval_image_transform(image_size: int = 256) -> transforms.Compose:
    return image_transform(image_size)


def get_imagenette_dataset(root: str | Path, split: str, image_size: int = 256) -> datasets.ImageFolder:
    root = Path(root)
    folder = root / "imagenette2-320" / split
    if not folder.exists():
        raise FileNotFoundError(f"Expected Imagenette split at {folder}. Run 00_setup_and_data.ipynb first.")
    return datasets.ImageFolder(folder, transform=image_transform(image_size))


def save_class_mapping(dataset: datasets.ImageFolder, output_path: str | Path) -> Dict[str, int]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mapping = {cls: idx for cls, idx in dataset.class_to_idx.items()}
    with output_path.open("w") as f:
        json.dump(mapping, f, indent=2)
    return mapping


class LatentTensorDataset(Dataset):
    """Loads cached VAE latents from shard_*.pt files.

    Each shard stores a dict with latents, labels, and paths. Imagenette latents fit comfortably
    in memory on most machines when saved as float16, so we concatenate them on init for speed.
    """

    def __init__(self, cache_dir: str | Path, dtype: torch.dtype = torch.float32) -> None:
        self.cache_dir = Path(cache_dir)
        self.shards = sorted(self.cache_dir.glob("shard_*.pt"))
        if not self.shards:
            raise FileNotFoundError(f"No latent shards found in {self.cache_dir}")
        latent_list = []
        label_list = []
        self.paths = []
        for shard in self.shards:
            data = torch.load(shard, map_location="cpu")
            latent_list.append(data["latents"].to(dtype=dtype))
            label_list.append(data["labels"].long())
            self.paths.extend(data.get("paths", [""] * len(data["labels"])))
        self.latents = torch.cat(latent_list, dim=0)
        self.labels = torch.cat(label_list, dim=0)

    def __len__(self) -> int:
        return self.labels.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.latents[idx], self.labels[idx]


def latent_stats(cache_dir: str | Path) -> Dict[str, object]:
    ds = LatentTensorDataset(cache_dir)
    return {
        "num_items": len(ds),
        "latent_shape": tuple(ds.latents.shape[1:]),
        "num_classes": int(ds.labels.max().item() + 1),
        "mean": float(ds.latents.mean().item()),
        "std": float(ds.latents.std().item()),
    }
