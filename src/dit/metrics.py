from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional

import torch


def compute_clean_fid(real_dir: str | Path, fake_dir: str | Path, mode: str = "clean", num_workers: int = 0) -> float:
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise ImportError("clean-fid is required. Install with `%pip install clean-fid`.") from exc
    return float(fid.compute_fid(str(fake_dir), str(real_dir), mode=mode, num_workers=num_workers))


def compute_torch_fidelity(real_dir: str | Path, fake_dir: str | Path, kid: bool = True, fid: bool = True) -> Dict[str, float]:
    try:
        import torch_fidelity
    except ImportError as exc:
        raise ImportError("torch-fidelity is required. Install with `%pip install torch-fidelity`.") from exc
    metrics = torch_fidelity.calculate_metrics(
        input1=str(fake_dir),
        input2=str(real_dir),
        cuda=torch.cuda.is_available(),
        isc=False,
        fid=fid,
        kid=kid,
        verbose=False,
    )
    return {k: float(v) for k, v in metrics.items()}


def safe_metric(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs), "ok"
    except Exception as exc:  # noqa: BLE001
        return float("nan"), f"metric_error: {type(exc).__name__}: {exc}"
