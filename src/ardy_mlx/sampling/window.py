"""Autoregressive window arithmetic (contract S5).

Pure index and root-translation helpers -- no model, no randomness, so every
rule in contract S5 can be tested directly.

Reference locations (``nv-tlabs/ardy`` @ ``693f74d``), all quoted in the contract:

* ``ardy_model.py:569-570`` -- ``num_autoregressive_steps`` / ``num_gen_frames``
* ``ardy_model.py:630-634`` -- ``history_end_frame`` / ``history_start_frame``
* ``ardy_model.py:655-663`` -- recentring around the last generated frame
* ``ardy_model.py:665-674`` -- ``first_heading_angle`` refresh after cropping
* ``scripts/generate.py:106-115`` -- default history length (batch path = max)

The two contract "potholes" this module deliberately preserves:

1. The batch default is the **max** history (160 frames for Core/20fps/H40),
   while the GUI slider default is the **min** (``patch`` = 4). They are
   different numbers and must not be mixed (contract S5.2 warning).
2. ``default_history_frames`` has **no lower clamp**: with a large enough
   horizon it can return ``0`` or a negative number. That is what the reference
   does (``generate.py:115``), so it is copied faithfully and pinned by a test
   rather than "fixed".
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Tuple

import mlx.core as mx

from ..io import NUM_FRAMES_PER_TOKEN, ROOT_BLOCK_DIM, default_history_frames

#: Root block layout inside ``frames (T, D)`` (contract S1.1): ``root_pos[3]`` then
#: ``global_root_heading[2]``.
ROOT_POS_SLICE = slice(0, 3)
ROOT_HEADING_SLICE = slice(3, 5)

__all__ = [
    "ROOT_POS_SLICE",
    "ROOT_HEADING_SLICE",
    "WindowPlan",
    "accumulated_translation",
    "crop_history",
    "default_history_frames",
    "extract_root_heading",
    "extract_root_pos",
    "frame_heading_angles",
    "max_window_frames",
    "plan_autoregressive_steps",
    "plan_window",
    "recenter_frames",
    "shift_root_xy",
]


def max_window_frames(fps: float, patch: int = NUM_FRAMES_PER_TOKEN, max_seconds: int = 10) -> int:
    """``(int(10 * fps) // patch) * patch`` (generate.py:114).

    Core/20fps -> 200 frames; G1/25fps -> 248 frames (contract S5.6).
    """
    if patch < 1:
        raise ValueError(f"patch must be >= 1, got {patch}")
    return (int(max_seconds * fps) // patch) * patch


def plan_autoregressive_steps(
    num_frames: int, init_history_len: int, gen_horizon_len: int
) -> Tuple[int, int]:
    """``num_autoregressive_steps`` / ``num_gen_frames`` (ardy_model.py:569-570).

    ``ceil((num_frames - init_history_len) / gen_horizon_len)``, then the frame
    count actually produced is ``steps * horizon + init_history_len`` (which can
    overshoot ``num_frames``).
    """
    if gen_horizon_len < 1:
        raise ValueError(f"gen_horizon_len must be >= 1, got {gen_horizon_len}")
    if init_history_len < 0:
        raise ValueError(f"init_history_len must be >= 0, got {init_history_len}")
    steps = int(math.ceil((num_frames - init_history_len) / gen_horizon_len))
    return steps, steps * gen_horizon_len + init_history_len


@dataclass(frozen=True)
class WindowPlan:
    """Frame-level slice plan for one autoregressive step (ardy_model.py:630-634).

    ``window`` indexes the *concatenated* sequence that is fed to the model:

        ``[cropped history (history_len frames)] + [generation (horizon frames)]``

    ``history_start``/``history_end`` index the full sequence held by the loop;
    ``local_history``/``local_generation`` index the concatenated window.
    """

    step: int
    history_end: int
    history_start: int
    history_len: int
    generation_start: int
    generation_len: int
    window_len: int
    cropped: bool

    @property
    def local_history(self) -> slice:
        return slice(0, self.history_len)

    @property
    def local_generation(self) -> slice:
        return slice(self.history_len, self.history_len + self.generation_len)

    @property
    def window(self) -> slice:
        return slice(0, self.window_len)


def plan_window(
    step: int,
    held_frames: int,
    gen_horizon_len: int,
    crop_history_length: Optional[int],
) -> WindowPlan:
    """Plan one step's history crop and generation slice (``ardy_model.py:630-634``).

    ``held_frames`` is the length of the **context window** the loop currently
    holds. The reference right-aligns the crop on that window:

        ``history_end   = held_frames``
        ``history_start = 0 if crop is None else max(0, history_end - crop)``
        ``generation    = horizon frames appended after history_end``

    so the window handed to the model is ``history + horizon`` frames. With the
    batch default (``crop = 160``, ``horizon = 40``, initial context 160) every
    step's window is exactly 200 frames = the trained 10 s budget (contract S5.2).
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    if gen_horizon_len < 1:
        raise ValueError(f"gen_horizon_len must be >= 1, got {gen_horizon_len}")
    held = int(held_frames)
    if held < 0:
        raise ValueError(f"held_frames must be >= 0, got {held_frames}")

    history_end = held
    if crop_history_length is None:
        history_start = 0
    else:
        if crop_history_length < 0:
            raise ValueError(f"crop_history_length must be >= 0, got {crop_history_length}")
        history_start = max(0, history_end - crop_history_length)

    history_len = history_end - history_start
    generation_start = history_end
    generation_len = gen_horizon_len
    window_len = history_len + generation_len
    return WindowPlan(
        step=step,
        history_end=history_end,
        history_start=history_start,
        history_len=history_len,
        generation_start=generation_start,
        generation_len=generation_len,
        window_len=window_len,
        cropped=history_start > 0,
    )


def extract_root_pos(frames: mx.array) -> mx.array:
    """``frames[..., 0:3]`` -- world root position (contract S1.1, metres)."""
    if frames.ndim != 3:
        raise ValueError(f"frames must be [B, T, D], got shape {frames.shape}")
    if frames.shape[-1] < ROOT_BLOCK_DIM:
        raise ValueError(
            f"frames last dim must be >= {ROOT_BLOCK_DIM} to hold the root block, got {frames.shape[-1]}"
        )
    return frames[..., ROOT_POS_SLICE]


def extract_root_heading(frames: mx.array) -> mx.array:
    """``frames[..., 3:5]`` -- ``(cos theta, sin theta)`` (contract S1.1)."""
    if frames.ndim != 3:
        raise ValueError(f"frames must be [B, T, D], got shape {frames.shape}")
    if frames.shape[-1] < ROOT_BLOCK_DIM:
        raise ValueError(
            f"frames last dim must be >= {ROOT_BLOCK_DIM} to hold the root block, got {frames.shape[-1]}"
        )
    return frames[..., ROOT_HEADING_SLICE]


def frame_heading_angles(frames: mx.array) -> mx.array:
    """Per-frame heading angle ``theta`` in radians, shape ``(B, T)``.

    ``theta`` is recovered from the stored ``(cos, sin)`` pair (contract S1.1),
    i.e. ``atan2(sin, cos)``. The reference can also derive it from hip positions
    (``motion_rep/tools.py:124-126``, ``theta = atan2(dz, -dx)`` with
    ``d = right_hip - left_hip``); this module only owns the ``(cos, sin)`` form
    so it needs no skeleton joint indices. Using the hip-based path as the
    authority is left open -- see ``docs/status-claude.md``.
    """
    heading = extract_root_heading(frames)
    return mx.arctan2(heading[..., 1], heading[..., 0])


def cropped_history_start(full_history_len: int, crop_history_length: Optional[int]) -> int:
    """``max(0, full_history_len - crop_history_length)`` (ardy_model.py:471)."""
    if crop_history_length is None:
        return 0
    if crop_history_length < 0:
        raise ValueError(f"crop_history_length must be >= 0, got {crop_history_length}")
    return max(0, full_history_len - crop_history_length)


def crop_history(frames: mx.array, crop_history_length: Optional[int]) -> mx.array:
    """Keep the last ``crop_history_length`` frames (right-aligned)."""
    if frames.ndim != 3:
        raise ValueError(f"frames must be [B, T, D], got shape {frames.shape}")
    start = cropped_history_start(frames.shape[1], crop_history_length)
    return frames[:, start:, :]


def shift_root_xy(frames: mx.array, delta: mx.array) -> mx.array:
    """Translate the root position in the ground plane only (contract S2.3/S5.4).

    Only ``x`` and ``z`` move; ``y`` (height) is untouched, and **the ``y``
    component of ``delta`` is ignored** rather than applied -- matching
    ``translate_normalized_root_motion`` (``ardy_model.py:29-38``), which asserts
    the translation has a zero y component.
    """
    if frames.ndim != 3:
        raise ValueError(f"frames must be [B, T, D], got shape {frames.shape}")
    if delta.shape != (frames.shape[0], 3):
        raise ValueError(
            f"delta must be [B, 3] with the same batch as frames {frames.shape[0]}, got {delta.shape}"
        )
    # Applied in place on the coordinate channels: MLX 0.32.3 mis-reports the
    # width of ``concatenate(computed, frames[..., 5:])``, so the row is rebuilt
    # from a 3 + 2 + tail split instead of an interior slice.
    xyz = mx.concatenate(
        [delta[..., 0:1], mx.zeros_like(delta[..., 1:2]), delta[..., 2:3]], axis=-1
    ).reshape(delta.shape[0], 1, 3)
    shifted = frames[..., 0:5].at[..., 0:3].add(xyz)
    result = mx.concatenate([shifted, frames[..., 5:]], axis=-1)
    if result.shape != frames.shape:
        raise AssertionError(
            f"shift_root_xy changed the shape: {frames.shape} -> {result.shape}"
        )
    return result


def recenter_frames(frames: mx.array, center_frame_index: mx.array) -> Tuple[mx.array, mx.array]:
    """Recentre on ``center_frame_index`` (ardy_model.py:655-663).

    Equivalent of ``recenter_root_motion`` (``ardy_motionrep.py:112-134``): take
    the root ``x``/``z`` of frame ``center_frame_index``, subtract it from every
    frame in the ground plane, and return the applied translation so it can be
    accumulated into ``global_transl`` and added back at the very end
    (``ardy_model.py:679``).

    Returns ``(recentered_frames, center_pos)`` with ``center_pos`` of shape
    ``(B, 3)`` whose ``y`` entry is **0.0**.
    """
    if frames.ndim != 3:
        raise ValueError(f"frames must be [B, T, D], got shape {frames.shape}")
    batch, num_frames = frames.shape[0], frames.shape[1]
    if center_frame_index.shape != (batch,):
        raise ValueError(
            f"center_frame_index must have shape ({batch},), got {center_frame_index.shape}"
        )

    pos = extract_root_pos(frames)
    idx = mx.maximum(center_frame_index, mx.zeros_like(center_frame_index))  # .clamp(min=0)
    idx = mx.minimum(idx, mx.full(idx.shape, num_frames - 1, dtype=idx.dtype))
    gather = mx.take(pos, idx, axis=1)  # (B, 1, 3)
    center_pos = mx.concatenate(
        [gather[:, 0, 0:1], mx.zeros((batch, 1), dtype=gather.dtype), gather[:, 0, 2:3]], axis=-1
    )
    return shift_root_xy(frames, -center_pos), center_pos


def accumulated_translation(center_positions: List[mx.array]) -> mx.array:
    """Sum of per-step recentring translations (``global_transl`` accumulation).

    ``global_transl = global_transl + center_pos`` (ardy_model.py:663) applied to
    the list of per-step translations.
    """
    if not center_positions:
        raise ValueError("accumulated_translation needs at least one center_pos")
    total = center_positions[0]
    for nxt in center_positions[1:]:
        if nxt.shape != total.shape:
            raise ValueError(
                f"all center_pos entries must share a shape, got {total.shape} and {nxt.shape}"
            )
        total = total + nxt
    return total
