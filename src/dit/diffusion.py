from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def _extract(a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]) -> torch.Tensor:
    out = a.gather(0, t)
    return out.reshape(t.shape[0], *((1,) * (len(x_shape) - 1)))


class GaussianDiffusion:
    """DDPM/DDIM utilities for epsilon-prediction diffusion in latent space."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 1e-4,
        beta_end: float = 2e-2,
        device: Optional[torch.device] = None,
    ) -> None:
        self.num_timesteps = num_timesteps
        betas = torch.linspace(beta_start, beta_end, num_timesteps, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat([torch.ones(1), alphas_cumprod[:-1]])
        self.betas = betas
        self.alphas = alphas
        self.alphas_cumprod = alphas_cumprod
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / alphas_cumprod - 1)
        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_variance = posterior_variance
        self.posterior_log_variance_clipped = torch.log(torch.clamp(posterior_variance, min=1e-20))
        self.posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        if device is not None:
            self.to(device)

    def to(self, device: torch.device | str) -> "GaussianDiffusion":
        for name, value in vars(self).items():
            if torch.is_tensor(value):
                setattr(self, name, value.to(device))
        return self

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = torch.randn_like(x_start) if noise is None else noise
        return _extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start + _extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        ) * noise

    def training_losses(self, model: nn.Module, x_start: torch.Tensor, y: torch.Tensor) -> Dict[str, torch.Tensor]:
        t = torch.randint(0, self.num_timesteps, (x_start.shape[0],), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        model_output = model(x_t, t, y)
        if getattr(model, "cfg", None) is not None and model.cfg.learn_sigma:
            model_output = model_output[:, : model.cfg.in_channels]
        loss = F.mse_loss(model_output, noise, reduction="none").mean(dim=(1, 2, 3))
        return {"loss": loss.mean(), "per_sample_loss": loss.detach(), "t": t.detach()}

    def predict_xstart_from_eps(self, x_t: torch.Tensor, t: torch.Tensor, eps: torch.Tensor) -> torch.Tensor:
        return _extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - _extract(
            self.sqrt_recipm1_alphas_cumprod, t, x_t.shape
        ) * eps

    def q_posterior_mean_variance(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = _extract(self.posterior_mean_coef1, t, x_t.shape) * x_start + _extract(
            self.posterior_mean_coef2, t, x_t.shape
        ) * x_t
        posterior_variance = _extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance = _extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def model_eps(self, model: nn.Module, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor, cfg_scale: float = 1.0) -> torch.Tensor:
        if cfg_scale != 1.0 and hasattr(model, "forward_with_cfg"):
            out = model.forward_with_cfg(x, t, y, cfg_scale)
        else:
            out = model(x, t, y)
        if getattr(model, "cfg", None) is not None and model.cfg.learn_sigma:
            out = out[:, : model.cfg.in_channels]
        return out

    @torch.no_grad()
    def ddpm_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        y: torch.Tensor,
        cfg_scale: float = 1.0,
        device: torch.device | str = "cuda",
        progress: bool = True,
    ) -> torch.Tensor:
        from tqdm.auto import tqdm

        model.eval()
        x = torch.randn(shape, device=device)
        iterator: Iterable[int] = reversed(range(self.num_timesteps))
        if progress:
            iterator = tqdm(list(iterator), desc="DDPM sampling")
        for i in iterator:
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            eps = self.model_eps(model, x, t, y, cfg_scale)
            pred_x0 = self.predict_xstart_from_eps(x, t, eps)
            mean, _, log_var = self.q_posterior_mean_variance(pred_x0, x, t)
            noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
            x = mean + torch.exp(0.5 * log_var) * noise
        return x

    @torch.no_grad()
    def ddim_sample_loop(
        self,
        model: nn.Module,
        shape: Tuple[int, int, int, int],
        y: torch.Tensor,
        steps: int = 100,
        cfg_scale: float = 1.0,
        eta: float = 0.0,
        device: torch.device | str = "cuda",
        progress: bool = True,
    ) -> torch.Tensor:
        from tqdm.auto import tqdm

        model.eval()
        x = torch.randn(shape, device=device)
        times = torch.linspace(self.num_timesteps - 1, 0, steps, device=device).long()
        iterator = tqdm(range(steps), desc=f"DDIM {steps} steps") if progress else range(steps)
        for idx in iterator:
            t = times[idx].repeat(shape[0])
            next_t_val = times[idx + 1].item() if idx < steps - 1 else -1
            eps = self.model_eps(model, x, t, y, cfg_scale)
            alpha_t = _extract(self.alphas_cumprod, t, x.shape)
            pred_x0 = (x - torch.sqrt(1 - alpha_t) * eps) / torch.sqrt(alpha_t)
            if next_t_val < 0:
                x = pred_x0
            else:
                next_t = torch.full_like(t, next_t_val)
                alpha_next = _extract(self.alphas_cumprod, next_t, x.shape)
                if eta == 0.0:
                    x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(1 - alpha_next) * eps
                else:
                    sigma = eta * torch.sqrt((1 - alpha_next) / (1 - alpha_t) * (1 - alpha_t / alpha_next))
                    noise = torch.randn_like(x)
                    x = torch.sqrt(alpha_next) * pred_x0 + torch.sqrt(torch.clamp(1 - alpha_next - sigma**2, min=0)) * eps + sigma * noise
        return x
