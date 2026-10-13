"""Sampling side of ardy-mlx (B side): diffusion schedule, DDIM, CFG, autoregressive loop.

Everything here mirrors ``docs/motion-contract.md``. The contract is the only
interface; this package must not invent field names, dimensions or orders.

Modules
-------
``diffusion``
    Cosine beta schedule (contract S4.1) and the deterministic DDIM step (S4.2).
``cfg``
    ``separated`` three-pass classifier-free guidance and its composition
    formula (S4.4).
``window``
    Pure index/arithmetic helpers for the autoregressive advance: window
    planning, history cropping, recentring, offsets (S5).
``autoregressive``
    The chunk loop itself (S5.3-S5.5), model-agnostic through a step callable.
``demo``
    A shape-faithful **placeholder** backend so the loop, the CLI and the tests
    can run end to end before A's converted weights exist. It is explicitly not
    the real model; see the module docstring.
``pipeline``
    Embedding loading, output assembly and the on-disk layout of contract S7.
"""

from __future__ import annotations

from .autoregressive import (
    ArdyAutoregressiveConfig,
    ArdySampler,
    ArdySampleResult,
    DenoiseRequest,
    DenoiseResult,
)
from .cfg import CfgWeights, apply_separated_cfg, make_three_pass_batch, split_three_pass
from .diffusion import (
    COSINE_SCHEDULE_OFFSET,
    COSINE_SCHEDULE_SCALE,
    DDIMSampler,
    DiffusionSchedule,
    alpha_bar,
    cosine_beta_schedule,
    get_diffusion_schedule,
    resolve_num_steps,
    space_timesteps,
)
from .window import (
    WindowPlan,
    accumulated_translation,
    crop_history,
    cropped_history_start,
    default_history_frames,
    extract_root_pos,
    extract_root_heading,
    frame_heading_angles,
    max_window_frames,
    plan_autoregressive_steps,
    plan_window,
    recenter_frames,
    shift_root_xy,
)

__all__ = [
    # diffusion
    "COSINE_SCHEDULE_OFFSET",
    "COSINE_SCHEDULE_SCALE",
    "DDIMSampler",
    "DiffusionSchedule",
    "alpha_bar",
    "cosine_beta_schedule",
    "get_diffusion_schedule",
    "resolve_num_steps",
    "space_timesteps",
    # cfg
    "CfgWeights",
    "apply_separated_cfg",
    "make_three_pass_batch",
    "split_three_pass",
    # window
    "WindowPlan",
    "accumulated_translation",
    "crop_history",
    "cropped_history_start",
    "default_history_frames",
    "extract_root_pos",
    "extract_root_heading",
    "frame_heading_angles",
    "max_window_frames",
    "plan_autoregressive_steps",
    "plan_window",
    "recenter_frames",
    "shift_root_xy",
    # autoregressive
    "ArdyAutoregressiveConfig",
    "ArdySampler",
    "ArdySampleResult",
    "DenoiseRequest",
    "DenoiseResult",
]
