"""The numbers decision 0008 is judged by -- displacement, speed, rotation amplitude.

These exist because "the picture looks different" is not evidence. Each function reads
one feature block of the contract's ``(T, D)`` layout and returns a number in physical
units, so two clips produced by two control settings can be compared as numbers and the
comparison can fail.

Why these three, and why these definitions
------------------------------------------
* :func:`displacement_m` measures the root's travel in the **x-z plane only**. ``y`` is
  vertical: a hop would otherwise be counted as forward progress, and the speed
  criterion would then be measuring the wrong motion.
* :func:`mean_speed_mps` is the mean per-frame root step times ``fps``. It is the
  "unit-time displacement" the criterion names, and it is deliberately *not* read from
  the ``velocities`` block: measuring the positions and the stored velocities separately
  is what lets a mismatch between them show up.
* :func:`joint_rotation_deg` is the mean axis-angle **angle** of each joint's global
  rotation. It is the "joint rotation amplitude" of the criterion, read from
  ``global_rot_data``, which is the same block ``amplitude`` deforms -- so criterion 4
  measures exactly what the deformation claims to change.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

__all__ = [
    "MotionStats",
    "displacement_m",
    "joint_rotation_deg",
    "mean_speed_mps",
    "motion_stats",
]


def _frames(frames: Any) -> np.ndarray:
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim == 3:
        if array.shape[0] < 1:
            raise ValueError("frames carries no batch")
        return array[0]
    if array.ndim == 2:
        return array
    raise ValueError(f"frames must be (T, D) or (B, T, D), got shape {np.shape(frames)}")


def _root_xz(frames: np.ndarray, layout) -> np.ndarray:
    """The root's ``(x, z)`` track, in metres."""
    block = layout.slices["root_pos"]
    root = frames[:, block]
    return root[:, [0, 2]]


def displacement_m(frames: Any, layout) -> float:
    """Distance the root travels across the clip, in the ground plane, in metres."""
    track = _root_xz(_frames(frames), layout)
    if track.shape[0] < 2:
        return 0.0
    return float(np.linalg.norm(track[-1] - track[0]))


def mean_speed_mps(frames: Any, layout, fps: Optional[int] = None) -> float:
    """Mean root speed over the clip, in metres/second.

    ``fps`` defaults to the layout's own rate; it is a parameter so a caller can ask the
    question for a different playout rate without rebuilding a layout.
    """
    rate = float(layout.fps if fps is None else fps)
    track = _root_xz(_frames(frames), layout)
    if track.shape[0] < 2:
        return 0.0
    steps = np.linalg.norm(np.diff(track, axis=0), axis=1)
    return float(steps.mean() * rate)


def joint_rotation_deg(frames: Any, layout, *, skip_first: bool = False) -> float:
    """Mean per-joint global rotation angle, in degrees.

    ``skip_first`` drops frame 0, which is what a continuation needs: that frame is
    carried over from the previous clip and is exempt from the amplitude deformation, so
    including it would dilute the very statistic the criterion is about.
    """
    from ..model.skeleton import cont6d_to_matrix
    import mlx.core as mx

    array = _frames(frames)
    if skip_first:
        array = array[1:]
    if array.shape[0] < 1:
        return 0.0
    block = layout.slices["global_rot_data"]
    rotation = array[:, block].reshape(array.shape[0], layout.nbjoints, 6).astype(np.float32)
    matrices = np.asarray(cont6d_to_matrix(mx.array(rotation)), dtype=np.float64)
    trace = np.trace(matrices, axis1=-2, axis2=-1)
    # arccos of the trace is the angle of the *principal* rotation, which is the range
    # the deformation itself clamps into; the two agree by construction.
    angle = np.arccos(np.clip((trace - 1.0) / 2.0, -1.0, 1.0))
    if angle.size == 0:
        return 0.0
    return float(np.degrees(angle).mean())


@dataclass(frozen=True)
class MotionStats:
    """The three numbers, together, as one comparable record."""

    frames: int
    displacement_m: float
    mean_speed_mps: float
    joint_rotation_deg: float

    def as_dict(self) -> dict:
        return {
            "frames": int(self.frames),
            "displacement_m": round(self.displacement_m, 6),
            "mean_speed_mps": round(self.mean_speed_mps, 6),
            "joint_rotation_deg": round(self.joint_rotation_deg, 4),
        }


def motion_stats(frames: Any, layout, *, skip_first: bool = False) -> MotionStats:
    """All three statistics for one clip."""
    array = _frames(frames)
    return MotionStats(
        frames=int(array.shape[0]),
        displacement_m=displacement_m(array, layout),
        mean_speed_mps=mean_speed_mps(array, layout),
        joint_rotation_deg=joint_rotation_deg(array, layout, skip_first=skip_first),
    )
