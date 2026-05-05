from __future__ import annotations

import csv
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def append_csv(path: str | Path, row: Dict[str, Any]) -> None:
    """Append a row to CSV while preserving/expanding the header.

    Training and validation rows sometimes have different fields. This helper rewrites
    the file with a union header if new columns appear, preventing malformed CSVs.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    row = dict(row)
    if not path.exists():
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            writer.writeheader()
            writer.writerow(row)
        return

    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    new_fields = [k for k in row.keys() if k not in fieldnames]
    if new_fields:
        fieldnames.extend(new_fields)
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for old_row in existing_rows:
                writer.writerow(old_row)
            writer.writerow(row)
    else:
        with path.open("a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow(row)


def save_json(path: str | Path, obj: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(obj, f, indent=2)


def save_checkpoint(
    path: str | Path,
    model,
    optimizer,
    scaler,
    ema,
    step: int,
    dit_config: Dict[str, Any],
    train_config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "ema": ema.state_dict() if ema is not None else None,
        "dit_config": dit_config,
        "train_config": train_config,
        "extra": extra or {},
    }
    torch.save(payload, path)


def load_checkpoint(path: str | Path, model, optimizer=None, scaler=None, ema=None, device="cpu") -> Dict[str, Any]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scaler is not None and ckpt.get("scaler") is not None:
        scaler.load_state_dict(ckpt["scaler"])
    if ema is not None and ckpt.get("ema") is not None:
        ema.load_state_dict(ckpt["ema"])
    return ckpt


@torch.no_grad()
def validation_loss(model, diffusion, loader, device, max_batches: int = 50) -> Dict[str, float]:
    model.eval()
    losses = []
    bins = {
        "t_000_100_loss": [],
        "t_100_300_loss": [],
        "t_300_600_loss": [],
        "t_600_1000_loss": [],
    }
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x = x.to(device, non_blocking=True).float()
        y = y.to(device, non_blocking=True).long()
        out = diffusion.training_losses(model, x, y)
        per_sample = out["per_sample_loss"].detach().cpu()
        t = out["t"].detach().cpu()
        losses.append(float(per_sample.mean().item()))
        masks = {
            "t_000_100_loss": (t >= 0) & (t < 100),
            "t_100_300_loss": (t >= 100) & (t < 300),
            "t_300_600_loss": (t >= 300) & (t < 600),
            "t_600_1000_loss": (t >= 600) & (t < 1000),
        }
        for name, mask in masks.items():
            if mask.any():
                bins[name].append(float(per_sample[mask].mean().item()))
    result = {"val_loss": float(np.mean(losses)) if losses else float("nan")}
    for name, values in bins.items():
        result[name] = float(np.mean(values)) if values else float("nan")
    model.train()
    return result


def make_loader(dataset, batch_size: int, shuffle: bool, num_workers: int, seed: int = 42) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=shuffle,
        worker_init_fn=seed_worker,
        generator=generator,
    )
