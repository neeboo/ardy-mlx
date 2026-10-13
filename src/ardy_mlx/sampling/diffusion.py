"""Diffusion schedule and DDIM step, copied from ``docs/motion-contract.md`` S4.

Source of truth (contract S4.1-S4.3), read off ``nv-tlabs/ardy`` @ ``693f74d``:

* ``model/diffusion.py:12-26``  -- cosine beta schedule (hardcoded, not in the checkpoint)
* ``model/diffusion.py:38``     -- ``alphas_cumprod_base = cumprod(1 - betas_base)``
* ``model/diffusion.py:50-58``  -- ``space_timesteps(n)`` sub-sampling
* ``model/diffusion.py:113-133`` -- ``DDIMSampler`` with ``eta = 0``
* ``ardy_model.py:622``         -- the loop runs ``t`` from large to small
* ``scripts/generate.py:198-202`` -- ``1 <= steps <= num_base_steps``

Two details that are easy to get wrong and that the tests lock down:

1. ``alpha_bar`` is ``cos((t + 0.008) / 1.008 * pi / 2) ** 2``. The ``0.008``
   numerator offset and the ``1.008`` denominator are *both* required; the
   common Stable-Diffusion variant ``((t + 0.008) / 1.008)`` has the same shape
   here but the literal in the reference is the one above.
2. The DDIM update indexes the schedule by a **scalar** ``t``. The reference
   writes ``alphas_cumprod_prev[t, None, None]`` on a 1-D table, which does not
   index a batch axis at all; this module keeps the table 1-D and takes a python
   ``int`` so the semantics are identical instead of accidentally broadcasting.

``float64`` is not supported on the MLX GPU backend, so the schedule is built
with Python floats (double precision) and stored ``float32`` -- which is also
the dtype of the released weights (contract S3.0: all tensors are ``F32``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence

import mlx.core as mx

from ..io import NUM_BASE_STEPS

#: ``cos((t + 0.008) / 1.008 * pi / 2) ** 2`` -- diffusion.py:14-15
COSINE_SCHEDULE_OFFSET = 0.008
COSINE_SCHEDULE_SCALE = 1.008

#: ``max_beta=0.999`` -- diffusion.py:12
MAX_BETA = 0.999

__all__ = [
    "COSINE_SCHEDULE_OFFSET",
    "COSINE_SCHEDULE_SCALE",
    "MAX_BETA",
    "CosineSchedule",
    "DiffusionSchedule",
    "DDIMSampler",
    "alpha_bar",
    "cosine_beta_schedule",
    "space_timesteps",
    "resolve_num_steps",
    "get_diffusion_schedule",
]


def alpha_bar(t: float) -> float:
    """``cos((t + 0.008) / 1.008 * pi/2) ** 2`` (diffusion.py:14-15).

    ``t`` is a normalized time in ``[0, 1]``.
    """
    return math.cos((t + COSINE_SCHEDULE_OFFSET) / COSINE_SCHEDULE_SCALE * math.pi / 2) ** 2


def cosine_beta_schedule(num_diffusion_timesteps: int, max_beta: float = MAX_BETA) -> List[float]:
    """Cosine beta schedule (diffusion.py:12-26), verbatim arithmetic.

    ``betas[i] = min(1 - alpha_bar(t2)/alpha_bar(t1), max_beta)`` with
    ``t1 = i/N`` and ``t2 = (i+1)/N``.

    Returned as python floats (double precision) on purpose: see the module
    docstring about the MLX GPU backend.
    """
    if num_diffusion_timesteps < 1:
        raise ValueError(
            f"num_diffusion_timesteps must be >= 1, got {num_diffusion_timesteps}"
        )
    betas: List[float] = []
    for i in range(num_diffusion_timesteps):
        t1 = i / num_diffusion_timesteps
        t2 = (i + 1) / num_diffusion_timesteps
        betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), max_beta))
    return betas


@dataclass(frozen=True)
class CosineSchedule:
    """The raw 1-D tables of one cosine schedule (contract S4.1).

    All six arrays have length ``num_base_steps`` and are ``float32``.
    """

    num_base_steps: int
    betas: mx.array  # (N,)
    alphas_cumprod: mx.array  # (N,)  cumprod(1 - betas)
    alphas_cumprod_prev: mx.array  # (N,)  shift right with a leading 1.0
    sqrt_alphas_cumprod: mx.array  # (N,)
    sqrt_one_minus_alphas_cumprod: mx.array  # (N,)
    sqrt_recip_alphas_cumprod: mx.array  # (N,)  1/sqrt(alpha_bar)
    sqrt_recipm1_alphas_cumprod: mx.array  # (N,)  sqrt(1/alpha_bar - 1)
    ddim_alpha_bar_prev: mx.array  # (N,)  what the DDIM loop actually multiplies by


def _build_schedule(num_base_steps: int) -> CosineSchedule:
    betas = cosine_beta_schedule(num_base_steps)

    alphas_cumprod: List[float] = []
    acc = 1.0
    for b in betas:
        acc *= 1.0 - b
        alphas_cumprod.append(acc)

    # diffusion.py:38 area -- the "previous" cumulative product, first entry 1.0.
    prev = [1.0] + alphas_cumprod[:-1]

    # DDIM (diffusion.py:128-130) needs the *previous* alpha bar one step down
    # the sub-sampled chain. Kept as its own field so the sampler never has to
    # guess which table it is indexing.
    ddim_prev = list(prev)

    f32 = lambda xs: mx.array([float(x) for x in xs], dtype=mx.float32)
    return CosineSchedule(
        num_base_steps=num_base_steps,
        betas=f32(betas),
        alphas_cumprod=f32(alphas_cumprod),
        alphas_cumprod_prev=f32(prev),
        sqrt_alphas_cumprod=f32([math.sqrt(a) for a in alphas_cumprod]),
        sqrt_one_minus_alphas_cumprod=f32([math.sqrt(1.0 - a) for a in alphas_cumprod]),
        sqrt_recip_alphas_cumprod=f32([1.0 / math.sqrt(a) for a in alphas_cumprod]),
        sqrt_recipm1_alphas_cumprod=f32(
            [math.sqrt(1.0 / a - 1.0) for a in alphas_cumprod]
        ),
        ddim_alpha_bar_prev=f32(ddim_prev),
    )


def space_timesteps(num_base_steps: int, num_steps: int) -> List[int]:
    """``space_timesteps`` (diffusion.py:50-58).

    ``use_timesteps = round(arange(num_base_steps) * (num_base_steps - 1) / max(1, n - 1))``,
    clamped to ``[0, num_base_steps - 1]``, deduplicated with order preserved.

    The reference uses it to pick which base timesteps a shorter chain visits.
    """
    if num_base_steps < 1:
        raise ValueError(f"num_base_steps must be >= 1, got {num_base_steps}")
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    denominator = max(1, num_steps - 1)
    out: List[int] = []
    for i in range(num_base_steps):
        raw = round(i * (num_base_steps - 1) / denominator)
        idx = min(max(int(raw), 0), num_base_steps - 1)
        out.append(idx)
    # Preserve first-seen order, drop duplicates (py version of torch.unique).
    seen = set()
    unique: List[int] = []
    for idx in out:
        if idx not in seen:
            seen.add(idx)
            unique.append(idx)
    return unique


def resolve_num_steps(num_steps: Optional[int], num_base_steps: int = NUM_BASE_STEPS) -> int:
    """``--diffusion_steps`` resolution (generate.py:196-202).

    ``None`` falls back to ``num_base_steps`` (10 for every released checkpoint).
    Allowed range is ``1 <= steps <= num_base_steps``; anything else raises.
    """
    if num_base_steps < 1:
        raise ValueError(f"num_base_steps must be >= 1, got {num_base_steps}")
    if num_steps is None:
        return num_base_steps
    if num_steps < 1 or num_steps > num_base_steps:
        raise ValueError(
            f"diffusion steps must satisfy 1 <= steps <= num_base_steps "
            f"({num_base_steps}), got {num_steps}"
        )
    return int(num_steps)


@dataclass(frozen=True)
class DiffusionSchedule:
    """A schedule plus the concrete denoising chain the sampler walks.

    ``timesteps`` is ordered the way the loop consumes it: **large to small**
    (contract S4.3, ``ardy_model.py:622``).
    """

    base: CosineSchedule
    num_steps: int
    used_timesteps: Sequence[int]  # ascending, from space_timesteps

    @property
    def timesteps(self) -> List[int]:
        """Denoising order: ``range(num_steps)[::-1]`` (ardy_model.py:622)."""
        return list(range(self.num_steps))[::-1]

    def alpha_bar_at(self, t: int) -> mx.array:
        """``alphas_cumprod`` at base index ``used_timesteps[t]``."""
        base_index = int(self.used_timesteps[t])
        return mx.take(self.base.alphas_cumprod, mx.array([base_index]))[0]

    def alpha_bar_prev_at(self, t: int) -> mx.array:
        """``alphas_cumprod_prev`` at base index ``used_timesteps[t]``."""
        base_index = int(self.used_timesteps[t])
        return mx.take(self.base.alphas_cumprod_prev, mx.array([base_index]))[0]

    def sqrt_recip_alphas_at(self, t: int) -> mx.array:
        base_index = int(self.used_timesteps[t])
        return mx.take(self.base.sqrt_recip_alphas_cumprod, mx.array([base_index]))[0]

    def sqrt_recipm1_alphas_at(self, t: int) -> mx.array:
        base_index = int(self.used_timesteps[t])
        return mx.take(self.base.sqrt_recipm1_alphas_cumprod, mx.array([base_index]))[0]


def get_diffusion_schedule(
    num_steps: Optional[int] = None, num_base_steps: int = NUM_BASE_STEPS
) -> DiffusionSchedule:
    """Build the schedule for ``num_steps`` denoising steps (contract S4.1/S4.3).

    The reference rebuilds the schedule **every autoregressive step**
    (``ardy_model.py:263-264`` and ``:627-628``); it depends only on
    ``num_base_steps`` and ``num_steps``, so one instance is reused here and the
    result is the same.
    """
    resolved = resolve_num_steps(num_steps, num_base_steps)
    base = _build_schedule(num_base_steps)
    used = space_timesteps(num_base_steps, resolved)
    return DiffusionSchedule(base=base, num_steps=resolved, used_timesteps=used)


class DDIMSampler:
    """Deterministic DDIM sampler, ``eta = 0`` (diffusion.py:113-133).

    One call = one denoising step::

        eps = (sqrt_recip_alphas_cumprod[t] * x_t - pred_xstart) / sqrt_recipm1_alphas_cumprod[t]
        x   = pred_xstart * sqrt(alpha_bar_prev[t]) + sqrt(1 - alpha_bar_prev[t]) * eps

    ``pred_xstart`` is the **clean-sample** prediction, not epsilon (contract
    S3.3: the denoiser predicts ``token_seq_pred_clean``).
    """

    def __init__(self, schedule: DiffusionSchedule):
        self.schedule = schedule

    def __call__(self, x_t: mx.array, pred_xstart: mx.array, t: int) -> mx.array:
        return self.step(x_t, pred_xstart, t)

    def step(self, x_t: mx.array, pred_xstart: mx.array, t: int) -> mx.array:
        """One DDIM update at chain index ``t`` (0-based over ``schedule.num_steps``)."""
        if x_t.shape != pred_xstart.shape:
            raise ValueError(
                f"x_t and pred_xstart must share a shape, got {x_t.shape} vs {pred_xstart.shape}"
            )
        sched = self.schedule
        sqrt_recip = sched.sqrt_recip_alphas_at(t)
        sqrt_recipm1 = sched.sqrt_recipm1_alphas_at(t)
        alpha_bar_prev = sched.alpha_bar_prev_at(t)
        eps = (sqrt_recip * x_t - pred_xstart) / sqrt_recipm1
        return pred_xstart * mx.sqrt(alpha_bar_prev) + mx.sqrt(1.0 - alpha_bar_prev) * eps

    @staticmethod
    def predict_eps(x_t: mx.array, pred_xstart: mx.array, schedule: DiffusionSchedule, t: int) -> mx.array:
        """The ``eps`` recovery used by ``step`` (diffusion.py:128-130).

        Exposed separately so tests can compare the intermediate directly.
        """
        sqrt_recip = schedule.sqrt_recip_alphas_at(t)
        sqrt_recipm1 = schedule.sqrt_recipm1_alphas_at(t)
        return (sqrt_recip * x_t - pred_xstart) / sqrt_recipm1
