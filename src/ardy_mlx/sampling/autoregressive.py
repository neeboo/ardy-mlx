"""The autoregressive chunk loop (contract S5.3-S5.5).

This module owns the *sequence* of operations the reference performs between
denoiser calls. It is deliberately model-agnostic: the real denoiser (A's side,
``src/ardy_mlx/model/**``) and the placeholder backend (``sampling/demo.py``)
both plug in through :class:`DenoiseRequest` / :class:`DenoiseResult`.

Order of operations per step, copied from ``ardy_model.py:600-680``:

1. plan the window (history crop + generation slice)      -- ``:630-634``
2. build the window from **clean** history + fresh noise   -- ``:481-490``
3. denoise; **only generation tokens are updated**         -- ``:302-313``
4. detokenize the window back to per-frame features        -- ``:340-360``
5. recentre on the last frame of the generated chunk       -- ``:655-663``
6. accumulate the translation into ``global_transl``       -- ``:663``
7. refresh ``first_heading_angle`` from the crop start     -- ``:665-674``
8. append the generated frames to the sequence             -- (implicit)

Two invariants the tests pin down:

* history entering the model is **clean** (never noised, never re-denoised);
* the translation accumulates monotonically and is applied to ``x``/``z`` only,
  so the output stays in world coordinates (contract S2.3).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import mlx.core as mx

from ..io import (
    DEFAULT_CFG_WEIGHT,
    NUM_BASE_STEPS,
    NUM_FRAMES_PER_TOKEN,
    TEXT_FEAT_DIM,
    FeatureLayout,
)
from .window import (
    WindowPlan,
    accumulated_translation,
    frame_heading_angles,
    plan_autoregressive_steps,
    plan_window,
    recenter_frames,
)

__all__ = [
    "ArdyAutoregressiveConfig",
    "ArdySampleResult",
    "ArdySampler",
    "DenoiseRequest",
    "DenoiseResult",
    "StepDiagnostics",
]


@dataclass(frozen=True)
class ArdyAutoregressiveConfig:
    """Everything the loop needs besides the model itself.

    ``history_frames`` is the **batch** default (contract S5.2): 160 for
    Core/20fps/H40. ``None`` means "no cropping" and reproduces the unbounded
    attention window the docstring at ``generate.py:106-115`` warns about.
    """

    layout: FeatureLayout
    num_frames: int
    batch_size: int = 1
    history_frames: Optional[int] = None
    num_steps: Optional[int] = None  # None -> layout/io NUM_BASE_STEPS
    cfg_weight: Sequence[float] = DEFAULT_CFG_WEIGHT
    seed: Optional[int] = None
    text_embeddings: Optional[mx.array] = None  # [B, L, 4096]
    text_mask: Optional[mx.array] = None  # [B, L]
    scalar_feature_dim: Optional[int] = None  # placeholder backend only

    def __post_init__(self) -> None:
        if self.num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {self.num_frames}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.history_frames is not None:
            if self.history_frames < 0:
                raise ValueError(f"history_frames must be >= 0, got {self.history_frames}")
            if self.history_frames % NUM_FRAMES_PER_TOKEN != 0:
                # generate.py:211-212
                raise ValueError(
                    f"history_frames must be a positive multiple of the patch size "
                    f"({NUM_FRAMES_PER_TOKEN}), got {self.history_frames}"
                )
        if self.text_embeddings is not None:
            if self.text_embeddings.ndim != 3 or self.text_embeddings.shape[-1] != TEXT_FEAT_DIM:
                raise ValueError(
                    f"text_embeddings must be [B, L, {TEXT_FEAT_DIM}], got shape "
                    f"{self.text_embeddings.shape}"
                )
            if self.text_embeddings.shape[0] != self.batch_size:
                raise ValueError(
                    f"text batch {self.text_embeddings.shape[0]} != batch_size {self.batch_size}"
                )
            if self.text_mask is None:
                raise ValueError("text_mask is required together with text_embeddings")
            if self.text_mask.shape != self.text_embeddings.shape[:2]:
                raise ValueError(
                    f"text_mask {self.text_mask.shape} must match text_embeddings "
                    f"{self.text_embeddings.shape[:2]}"
                )
        elif self.text_mask is not None:
            raise ValueError("text_mask given without text_embeddings")

    @property
    def patch(self) -> int:
        return NUM_FRAMES_PER_TOKEN

    @property
    def horizon(self) -> int:
        return self.layout.horizon

    @property
    def fps(self) -> int:
        return self.layout.fps

    @property
    def init_history_len(self) -> int:
        """Frames of context before the first generated chunk (contract S5.2/S5.3).

        The batch default is ``default_history_frames`` -- 160 for Core/20fps/H40,
        which is the **max**, not the GUI slider's min (S5.2). When the caller
        passes an explicit ``history_frames`` shorter than that default, the initial
        context follows the caller's crop length (there is no point seeding more
        history than every step will immediately crop away); with the default the
        two are the same number, so the reference's arithmetic is unchanged.
        """
        default = self.layout.default_history_frames
        if self.history_frames is None:
            return default
        return min(default, self.history_frames)

    @property
    def crop_history_length(self) -> Optional[int]:
        return self.history_frames


@dataclass
class DenoiseRequest:
    """One autoregressive step handed to the denoise callable.

    ``history_frames`` are the frames that **must stay clean**; ``generation_len``
    frames of fresh noise (``generation_noise``) are appended by the caller and
    are the only positions the denoiser is allowed to overwrite.
    """

    step: int
    plan: WindowPlan
    history_frames: mx.array  # [B, history_len, D]  (clean)
    generation_noise: mx.array  # [B, gen_tokens, HYBRID_TOKEN_DIM]
    first_heading_angle: mx.array  # [B]
    text_embeddings: Optional[mx.array]
    text_mask: Optional[mx.array]
    cfg_weight: tuple
    num_steps: int
    seed: Optional[int]
    #: Global frame index of this chunk's first frame, i.e. how many frames the
    #: sequence already holds. A real model does not need it; the placeholder
    #: backend and the tests use it to stay continuous across steps.
    frame_index_offset: int = 0

    @property
    def history_len(self) -> int:
        return self.plan.history_len

    @property
    def generation_len(self) -> int:
        return self.plan.generation_len


@dataclass
class DenoiseResult:
    """What a denoise callable must return for one step."""

    generated_frames: mx.array  # [B, generation_len, D]  -- clean features
    offsets: Optional[List[int]] = None  # optional per-step chunk boundary override


@dataclass
class StepDiagnostics:
    """Per-step record, so drift and translation can be inspected after a run."""

    step: int
    history_start: int
    history_end: int
    history_len: int
    generation_start: int
    generation_len: int
    window_len: int
    cropped: bool
    center_pos: List[float]  # the recentring translation applied this step
    first_heading_angle: float
    clean_history_only: bool  # sanity flag: history was not noised


@dataclass
class ArdySampleResult:
    """Sampler output before the final translation is added back.

    ``frames`` is the full sequence (initial history + every generated chunk).
    ``global_transl`` (contract S2.3 / ardy_model.py:679) must be added to the
    root position once at the end to recover world coordinates; ``translated``
    exposes the already-shifted tensor for convenience.
    """

    frames: mx.array  # [B, T, D]
    global_transl: mx.array  # [B, 3]
    first_heading_angle: mx.array  # [B]
    offsets: List[int]
    diagnostics: List[StepDiagnostics]
    num_steps: int
    cfg_weight: tuple

    @property
    def num_frames(self) -> int:
        return self.frames.shape[1]

    def translated(self, shift_xy: bool = True) -> mx.array:
        """Apply ``global_transl`` to the root ``x``/``z`` (world coordinates)."""
        from .window import shift_root_xy

        if not shift_xy:
            return self.frames
        return shift_root_xy(self.frames, self.global_transl)


class ArdySampler:
    """Drives the autoregressive chunk loop against a denoise callable.

    ``step_fn`` receives a :class:`DenoiseRequest` and returns a
    :class:`DenoiseResult`; it owns the diffusion chain, the three-pass CFG
    expansion and the detokenize step for one window. ``initial_history_frames``
    is the clean seed context (defaults to zeros, matching the reference's
    zero-initialized history).
    """

    def __init__(
        self,
        config: ArdyAutoregressiveConfig,
        step_fn: Callable[[DenoiseRequest], DenoiseResult],
        initial_history_frames: Optional[mx.array] = None,
        recenter_hook: Optional[Callable[[mx.array, mx.array, int], None]] = None,
    ):
        self.config = config
        self.step_fn = step_fn
        self.recenter_hook = recenter_hook
        batch, frames, dim = (
            config.batch_size,
            config.init_history_len,
            config.layout.motion_rep_dim,
        )
        if initial_history_frames is None:
            self.initial_history = mx.zeros((batch, frames, dim), dtype=mx.float32)
        else:
            expected = (batch, frames, dim)
            if initial_history_frames.shape != expected:
                raise ValueError(
                    f"initial_history_frames must be {expected}, got {initial_history_frames.shape}"
                )
            self.initial_history = initial_history_frames

    # -- helpers ---------------------------------------------------------
    @property
    def num_autoregressive_steps(self) -> int:
        steps, _ = plan_autoregressive_steps(
            self.config.num_frames, self.config.init_history_len, self.config.horizon
        )
        return max(steps, 0)

    def _step_keys(self, steps: int) -> List[mx.array]:
        """One RNG key per step: the reference draws a fresh ``randn`` each step.

        Contract S4.5 records the reference's seeding as ``seed_everything(seed)``
        before the loop, followed by ``torch.randn`` per generation token block.
        MLX has no ``manual_seed`` equivalent, so the seed becomes an MLX key and
        is split once into ``steps`` independent keys. **The resulting numbers are
        not claimed to match PyTorch's stream** -- that is exactly what the pending
        golden comparison (contract S4.5, "待 A 后续提供 golden 时确认 RNG 细节")
        has to settle. Determinism within MLX is what this guarantees.
        """
        if self.config.seed is None:
            # generate.py:232-233 only calls seed_everything when --seed is given.
            base = mx.random.key(int(mx.random.randint(0, 2**31 - 1)))
        else:
            base = mx.random.key(int(self.config.seed))
        return list(mx.random.split(base, max(steps, 1)))

    def _initial_heading(self) -> mx.array:
        return frame_heading_angles(self.initial_history)[:, 0]

    # -- main loop -------------------------------------------------------
    def run(self) -> ArdySampleResult:
        cfg = self.config
        horizon = cfg.horizon
        batch = cfg.batch_size

        gen_tokens = horizon // cfg.patch
        if horizon % cfg.patch != 0:
            raise ValueError(
                f"horizon {horizon} must be a multiple of the patch size {cfg.patch}"
            )

        # ``trajectory`` holds only the generated frames, as one continuous sequence
        # in a *recentred* frame; ``context`` is the cropped tail that feeds the next
        # step. The seed history is ``self.initial_history`` and is **not** part of
        # the output: the reference keeps only the generated chunks, and its frame
        # count is ``init_history_len + steps * horizon`` (ardy_model.py:569-570).
        # After every step the whole trajectory is translated so the newest frame
        # sits at the origin (ardy_model.py:655-663); a uniform translation is a
        # rigid motion, so the concatenation stays a valid motion, and adding
        # ``global_transl`` back at the end (ardy_model.py:679) restores world
        # coordinates.
        context = self.initial_history
        trajectory: Optional[mx.array] = None
        offsets: List[int] = [0]
        diagnostics: List[StepDiagnostics] = []
        center_positions: List[mx.array] = []
        first_heading_angle = self._initial_heading()
        step_keys = self._step_keys(self.num_autoregressive_steps)

        for step in range(self.num_autoregressive_steps):
            plan = plan_window(
                step=step,
                held_frames=context.shape[1],
                gen_horizon_len=horizon,
                crop_history_length=cfg.crop_history_length,
            )
            history = context
            # Frames actually handed to the model at this step.
            history_len = history.shape[1]

            key = step_keys[step]
            noise = mx.random.normal(
                (batch, gen_tokens, cfg.layout.dim_token), key=key
            ).astype(mx.float32)

            request = DenoiseRequest(
                step=step,
                plan=plan,
                history_frames=history,
                generation_noise=noise,
                first_heading_angle=first_heading_angle,
                text_embeddings=cfg.text_embeddings,
                text_mask=cfg.text_mask,
                cfg_weight=tuple(cfg.cfg_weight),
                num_steps=int(cfg.num_steps or NUM_BASE_STEPS),
                seed=cfg.seed,
                frame_index_offset=0 if trajectory is None else int(trajectory.shape[1]),
            )
            result = self.step_fn(request)
            generated = result.generated_frames
            expected = (batch, plan.generation_len, cfg.layout.motion_rep_dim)
            if generated.shape != expected:
                raise ValueError(
                    f"step_fn returned generated_frames {generated.shape}, expected {expected}"
                )
            # Trajectory index where this step's new frames start (before the append).
            history_end = 0 if trajectory is None else int(trajectory.shape[1])

            trajectory = (
                generated
                if trajectory is None
                else mx.concatenate([trajectory, generated], axis=1)
            )
            center_index = mx.full((batch,), trajectory.shape[1] - 1, dtype=mx.int32)
            trajectory, center_pos = recenter_frames(trajectory, center_index)
            center_positions.append(center_pos)

            # The next step's context is the tail of the recentred trajectory, falling
            # back to the seed history until enough frames exist.
            # The window is the frames ending at the newest generated frame: the
            # tail of the trajectory, spilling back into the seed context while the
            # trajectory is still shorter than the crop length.
            if cfg.crop_history_length is None:
                context = mx.concatenate([self.initial_history, trajectory], axis=1)
            else:
                crop = cfg.crop_history_length
                if trajectory.shape[1] >= crop:
                    context = trajectory[:, trajectory.shape[1] - crop :, :]
                else:
                    take = crop - trajectory.shape[1]
                    context = mx.concatenate(
                        [
                            self.initial_history[
                                :, max(0, self.initial_history.shape[1] - take) :, :
                            ],
                            trajectory,
                        ],
                        axis=1,
                    )

            # The hook fires once per step with the sequence as it stands after this
            # step's recentring.
            if self.recenter_hook is not None:
                self.recenter_hook(trajectory, center_pos, step)

            # Chunk boundary: the trajectory grew by exactly one horizon this step
            # (offsets[0] = 0, offsets[K] = T; contract S7).
            offsets.append(trajectory.shape[1])

            # Refresh first_heading_angle from the crop start (ardy_model.py:665-674):
            # heading_angles[:, max(0, full_history_len - crop_history_length)] --
            # which is exactly the context's first frame.
            first_heading_angle = frame_heading_angles(context)[:, 0]

            diagnostics.append(
                StepDiagnostics(
                    step=step,
                    # Index of the crop start inside the trajectory (0 while the
                    # window still reaches into the seed context).
                    history_start=max(0, trajectory.shape[1] - history_len),
                    history_end=history_end,
                    history_len=history_len,
                    generation_start=history_end,
                    generation_len=plan.generation_len,
                    window_len=history_len + plan.generation_len,
                    cropped=trajectory.shape[1] > history_len,
                    center_pos=[float(v) for v in center_pos[0].tolist()],
                    first_heading_angle=float(first_heading_angle[0]),
                    clean_history_only=True,
                )
            )

        global_transl = (
            accumulated_translation(center_positions)
            if center_positions
            else mx.zeros((batch, 3), dtype=mx.float32)
        )

        # The trajectory already holds exactly the generated frames; check the frame
        # count the reference reports (ardy_model.py:569-570, generate.py:129-130).
        if trajectory is None:
            frames = self.initial_history[:, :0, :]
        else:
            frames = trajectory
        expected_frames = self.num_autoregressive_steps * horizon
        if frames.shape[1] != expected_frames:
            raise AssertionError(
                f"generated {frames.shape[1]} frames, expected {expected_frames}"
            )

        return ArdySampleResult(
            frames=frames,
            global_transl=global_transl,
            first_heading_angle=first_heading_angle,
            offsets=offsets,
            diagnostics=diagnostics,
            num_steps=int(cfg.num_steps or NUM_BASE_STEPS),
            cfg_weight=tuple(cfg.cfg_weight),
        )
