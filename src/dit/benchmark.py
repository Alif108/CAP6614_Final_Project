from __future__ import annotations

import time
from typing import Dict, Tuple

import torch

from .models import count_parameters


def estimate_flops_fvcore(model, input_shape: Tuple[int, int, int, int], device) -> float:
    try:
        from fvcore.nn import FlopCountAnalysis
    except Exception:
        return float("nan")
    model.eval()
    x = torch.randn(input_shape, device=device)
    t = torch.randint(0, 1000, (input_shape[0],), device=device)
    y = torch.randint(0, model.cfg.num_classes, (input_shape[0],), device=device)
    try:
        flops = FlopCountAnalysis(model, (x, t, y)).total()
        return float(flops)
    except Exception:
        return float("nan")


def benchmark_forward(model, batch_size: int = 8, device: str | torch.device = "cuda", num_warmup: int = 10, num_iters: int = 50) -> Dict[str, object]:
    device = torch.device(device)
    model = model.to(device).eval()
    x = torch.randn(batch_size, model.cfg.in_channels, model.cfg.input_size, model.cfg.input_size, device=device)
    t = torch.randint(0, 1000, (batch_size,), device=device)
    y = torch.randint(0, model.cfg.num_classes, (batch_size,), device=device)
    status = "ok"
    peak_mem = float("nan")
    latency_ms = float("nan")
    try:
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = model(x, t, y)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)
                starter = torch.cuda.Event(enable_timing=True)
                ender = torch.cuda.Event(enable_timing=True)
                torch.cuda.synchronize(device)
                starter.record()
                for _ in range(num_iters):
                    _ = model(x, t, y)
                ender.record()
                torch.cuda.synchronize(device)
                latency_ms = starter.elapsed_time(ender) / num_iters
                peak_mem = torch.cuda.max_memory_allocated(device) / 1024**2
            else:
                start = time.perf_counter()
                for _ in range(num_iters):
                    _ = model(x, t, y)
                latency_ms = (time.perf_counter() - start) * 1000 / num_iters
                status = "cpu_no_cuda_memory_stats"
    except RuntimeError as exc:
        status = f"runtime_error: {exc}"
        if device.type == "cuda":
            torch.cuda.empty_cache()
    steps_per_sec = 1000.0 / latency_ms if latency_ms and latency_ms == latency_ms else float("nan")
    return {
        "model_size": model.cfg.model_name,
        "conditioning": model.cfg.conditioning,
        "parameters": count_parameters(model),
        "input_size": model.cfg.input_size,
        "in_channels": model.cfg.in_channels,
        "patch_size": model.cfg.patch_size,
        "tokens": model.cfg.num_tokens,
        "depth": model.cfg.depth,
        "hidden_size": model.cfg.hidden_size,
        "num_heads": model.cfg.num_heads,
        "estimated_flops": estimate_flops_fvcore(model, tuple(x.shape), device),
        "batch_size": batch_size,
        "dtype": str(next(model.parameters()).dtype),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "forward_latency_ms": latency_ms,
        "forward_steps_per_sec": steps_per_sec,
        "peak_gpu_mem_mb": peak_mem,
        "status": status,
    }
