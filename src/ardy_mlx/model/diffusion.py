"""Cosine beta schedule + deterministic DDIM (eta = 0).

Reference: ``ardy/model/diffusion.py``. The schedule is **hardcoded**, not read
from the checkpoint, so it is reproduced here exactly as written -- including the
quirks:

* ``space_timesteps`` maps a requested step count onto the 10 training steps with
  ``round(arange(n) * (n_train - 1) / max(1, n - 1))``, clamped to ``n_train - 1``;
  the returned ``map_tensor`` is what indexes the diffusion buffers, while the
  loop variable indexes the *subsampled* schedule. Conflating the two silently
  picks the wrong noise level.
* ``calc_diffusion_vars`` is re-run on **every** denoising step in the reference
  (``ardy_model.py:263-264``), which matters for ``alphas_cumprod`` because it
  re-cumprods the *subsampled* betas.
* ``sqrt_recipm1_alphas_cumprod = rsqrt(alphas_cumprod / (1 - alphas_cumprod))``
  -- note it is *not* ``sqrt(1/a - 1)`` written the obvious way, though it is
  algebraically the same; the exact expression is kept so the floats match.
"""

from __future__ import annotations

import math

import mlx.core as mx

__all__ = ["get_beta_schedule", "Diffusion", "DDIMSampler"]


def get_beta_schedule(num_diffusion_timesteps: int, max_beta: float = 0.999):
    """``get_beta_schedule`` (diffusion.py:12-26), cosine schedule."""

    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    betas = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return betas


class Diffusion:
    """``Diffusion`` (diffusion.py:28-109)."""

    def __init__(self, num_base_steps: int = 10):
        self.num_base_steps = num_base_steps
        self.betas_base = mx.array(get_beta_schedule(num_base_steps), dtype=mx.float32)
        self.alphas_cumprod_base = mx.cumprod(1.0 - self.betas_base, axis=0)
        use_timesteps, self.map_tensor = self.space_timesteps(num_base_steps)
        self.calc_diffusion_vars(use_timesteps)

    def space_timesteps(self, num_denoising_steps: int):
        """``space_timesteps`` (diffusion.py:50-58) -> ``(use_timesteps, map_tensor)``."""
        n = self.num_base_steps
        frac_stride = (n - 1) / max(1, num_denoising_steps - 1)
        use = mx.round(mx.arange(n, dtype=mx.float32) * frac_stride).astype(mx.int32)
        use = mx.minimum(use, n - 1)
        return use, mx.arange(n, dtype=mx.int32)[use]

    def calc_diffusion_vars(self, use_timesteps: mx.array) -> None:
        """``calc_diffusion_vars`` (diffusion.py:60-92)."""
        alphas_cumprod = self.alphas_cumprod_base[use_timesteps]
        ones = mx.array([1.0], dtype=alphas_cumprod.dtype)
        last_alpha_cumprod = mx.concatenate([ones, alphas_cumprod[:-1]], axis=0)
        self.betas = 1.0 - alphas_cumprod / last_alpha_cumprod

        self.alphas = 1.0 - self.betas
        alphas_cumprod = mx.cumprod(self.alphas, axis=0)
        alphas_cumprod = mx.clip(alphas_cumprod, a_min=1e-9, a_max=None)
        self.alphas_cumprod = alphas_cumprod

        self.alphas_cumprod_prev = mx.concatenate([ones, alphas_cumprod[:-1]], axis=0)
        self.sqrt_recip_alphas_cumprod = mx.rsqrt(alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = mx.rsqrt(alphas_cumprod / (1.0 - alphas_cumprod))
        self.posterior_variance = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        self.sqrt_alphas_cumprod = mx.rsqrt(1.0 / alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = mx.rsqrt(1.0 / (1.0 - alphas_cumprod))


class DDIMSampler:
    """``DDIMSampler`` (diffusion.py:113-133), deterministic (eta = 0)."""

    def __init__(self, diffusion: Diffusion):
        self.diffusion = diffusion

    def __call__(self, x_t: mx.array, pred_xstart: mx.array, t: int) -> mx.array:
        d = self.diffusion
        t_scalar = int(t)
        eps = (
            d.sqrt_recip_alphas_cumprod[t_scalar] * x_t - pred_xstart
        ) / d.sqrt_recipm1_alphas_cumprod[t_scalar]
        alpha_bar_prev = d.alphas_cumprod_prev[t_scalar]
        return pred_xstart * mx.sqrt(alpha_bar_prev) + mx.sqrt(1 - alpha_bar_prev) * eps
