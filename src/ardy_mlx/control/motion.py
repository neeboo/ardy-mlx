"""The two deformations, and the one parameter that also changes the request.

Frozen criteria: ``docs/decisions/0008-motion-controls.md``. Read that first; this
module is the implementation of it and nothing more.

Shape of the thing
------------------
``frames`` is the contract's ``(B, T, D)`` motion feature block, in **physical units**
(metres, metres/second, 6D rotations) -- not the normalised space the autoregressive
loop carries between windows. Speed and amplitude are physical statements ("twice as
fast", "half the swing"), so they are applied where the units are physical.

Order of operations, and why it is this order
---------------------------------------------
1. **resample the time axis** (``speed``), which is what turns ``source_frames`` into
   ``output_frames``;
2. **rescale the ``velocities`` block** by ``speed``, so the stored velocity agrees
   with what differencing the resampled positions gives;
3. **renormalise the heading**, because linear interpolation of ``(cos, sin)`` leaves a
   vector that is no longer unit length;
4. **scale the rotation angles** (``amplitude``) -- last, so that step 1's exactness at
   index 0 is what decides whether frame 0 survives.

Every step is skipped outright when its parameter is 1.0, which is what makes the
identity case bit-identical rather than merely close.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "CONTROL_AMPLITUDE_RANGE",
    "CONTROL_MAX_SOURCE_FACTOR",
    "CONTROL_SPEED_RANGE",
    "ControlError",
    "ControlResult",
    "MotionControls",
    "apply_controls",
    "required_source_frames",
]

#: Accepted ``speed`` band. The router's own tempo mapping only reaches 0.65..1.35;
#: the band is wider so a caller can drive the layer directly (the tests do), and
#: bounded so that "faster" can never mean "more source frames than the cap below".
CONTROL_SPEED_RANGE: Tuple[float, float] = (0.25, 2.0)
#: Accepted ``amplitude`` band. The upper bound is 2.0 rather than the router's 1.4
#: because a scaled joint angle is still a valid rotation well past the router's range,
#: and it is 2.0 rather than more because the reference walk already reaches 89 degrees
#: per joint (measured) and a rotation's principal angle saturates at 180.
CONTROL_AMPLITUDE_RANGE: Tuple[float, float] = (0.1, 2.0)
#: Hard ceiling on the source expansion: ``source_frames <= output_frames * this``.
#: A ``speed`` above it fails by name rather than silently generating an unbounded run.
CONTROL_MAX_SOURCE_FACTOR = 2.0

#: Below this a rotation is treated as the identity: the axis of a zero rotation is
#: undefined, and dividing by ``sin(theta)`` there is how naive implementations produce
#: NaN on straight-limbed rest poses.
_ZERO_ANGLE = 1e-9


class ControlError(Exception):
    """A control parameter that cannot be honoured, named rather than clamped.

    A clamped parameter is the failure mode this class exists to prevent: the page
    would show ``speed 0.5`` and the body would walk at something else. Every rejection
    here carries a stable ``name`` the API returns verbatim.
    """

    def __init__(self, name: str, message: str):
        super().__init__(message)
        self.name = name
        self.message = message


def _finite(value: Any, *, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ControlError(f"control_{name}_not_a_number", f"{name} must be a number, got {value!r}") from exc
    if not np.isfinite(number):
        raise ControlError(f"control_{name}_not_finite", f"{name} must be finite, got {value!r}")
    return number


@dataclass(frozen=True)
class MotionControls:
    """The parameters that are allowed to change a clip, and their provenance.

    ``expression``/``gaze`` are carried because the router produces them and the page
    displays them, but ``applied`` never contains them: ``expression_placeholder`` is
    ``True`` for every value this repository can construct.
    """

    speed: float = 1.0
    amplitude: float = 1.0
    expression: Optional[str] = None
    gaze: Optional[str] = None
    expression_placeholder: bool = True
    #: Exempt frame 0 from the amplitude deformation. Set on the continuation path,
    #: where frame 0 is the previous clip's last frame and must survive untouched.
    preserve_first_frame: bool = False

    def __post_init__(self) -> None:
        speed = _finite(self.speed, name="speed")
        amplitude = _finite(self.amplitude, name="amplitude")
        low, high = CONTROL_SPEED_RANGE
        if not (low <= speed <= high):
            raise ControlError(
                "control_speed_out_of_range",
                f"speed must be within [{low}, {high}], got {speed}",
            )
        low, high = CONTROL_AMPLITUDE_RANGE
        if not (low <= amplitude <= high):
            raise ControlError(
                "control_amplitude_out_of_range",
                f"amplitude must be within [{low}, {high}], got {amplitude}",
            )
        if speed > CONTROL_MAX_SOURCE_FACTOR:  # pragma: no cover - the band above is already tighter
            raise ControlError(
                "control_speed_exceeds_source_factor",
                f"speed {speed} would need more than {CONTROL_MAX_SOURCE_FACTOR}x the source frames",
            )
        object.__setattr__(self, "speed", speed)
        object.__setattr__(self, "amplitude", amplitude)

    @classmethod
    def from_params(cls, params: Optional[Mapping[str, Any]], **overrides: Any) -> "MotionControls":
        """Build from the router's ``decision.params`` mapping.

        A missing ``speed``/``amplitude`` is ``1.0`` (neutral), which is the same
        direction every other default in this repository takes: an absent reading must
        not become a value.
        """
        source: Mapping[str, Any] = params or {}
        fields = {
            "speed": source.get("speed", 1.0),
            "amplitude": source.get("amplitude", 1.0),
            "expression": source.get("expression"),
            "gaze": source.get("gaze"),
            # Never taken from the payload: the placeholder flag is a fact about this
            # rig, not something a caller may assert away.
            "expression_placeholder": True,
        }
        fields.update(overrides)
        return cls(**fields)

    @property
    def identity(self) -> bool:
        """True when this control set cannot change any clip."""
        return self.speed == 1.0 and self.amplitude == 1.0

    def applied_names(self) -> Tuple[str, ...]:
        """The parameters that **actually did something**, in application order.

        ``expression``/``gaze`` are absent by construction: they are placeholders and
        saying otherwise is exactly the overclaim this method exists to prevent.
        """
        names = []
        if self.speed != 1.0:
            names.append("speed")
        if self.amplitude != 1.0:
            names.append("amplitude")
        return tuple(names)

    def as_dict(self) -> dict:
        return {
            "speed": self.speed,
            "amplitude": self.amplitude,
            "expression": self.expression,
            "gaze": self.gaze,
            "expression_placeholder": self.expression_placeholder,
            "preserve_first_frame": self.preserve_first_frame,
        }


def required_source_frames(output_frames: int, speed: float, *, horizon: int) -> int:
    """How many frames the engine must produce for ``output_frames`` at ``speed``.

    ``speed <= 1`` needs no extra frames: the read pointer ``i * speed`` stays inside
    ``[0, output_frames)``. ``speed > 1`` reads past the end, so the source must be
    longer, and it is rounded **up to a whole ``horizon``** because
    ``validate_duration`` has already frozen "a clip is a whole number of horizons" and
    this layer is not the place to loosen that.

    Returning a number here is the "conditional input" half of ``speed``: it changes
    what the sampler is asked for, not just what comes out of it.
    """
    if output_frames < 1:
        raise ControlError("control_output_frames_invalid", f"output_frames must be >= 1, got {output_frames}")
    if horizon < 1:
        raise ControlError("control_horizon_invalid", f"horizon must be >= 1, got {horizon}")
    speed = _finite(speed, name="speed")
    if speed <= 1.0:
        return int(output_frames)
    wanted = int(np.ceil(output_frames * speed - 1e-9))
    windows = int(np.ceil(wanted / horizon))
    source = max(windows, 1) * horizon
    ceiling = int(np.ceil(output_frames * CONTROL_MAX_SOURCE_FACTOR / horizon)) * horizon
    if source > ceiling:
        raise ControlError(
            "control_source_frames_exceed_cap",
            f"speed {speed} needs {source} source frames, above the {ceiling}-frame cap "
            f"for {output_frames} output frames",
        )
    return source


@dataclass(frozen=True)
class ControlResult:
    """A controlled clip plus the facts needed to report it honestly."""

    frames: np.ndarray
    output_frames: int
    source_frames: int
    speed: float
    amplitude: float
    applied: Tuple[str, ...]
    #: Fraction of (frame, joint) rotations whose scaled angle crossed 180 degrees and
    #: was therefore clamped. Non-zero means the amplitude statistic saturates for that
    #: clip; it is reported rather than hidden.
    saturated_fraction: float
    preserve_first_frame: bool
    identity: bool

    def as_dict(self) -> dict:
        return {
            "speed": self.speed,
            "amplitude": self.amplitude,
            "applied": list(self.applied),
            # Stated in the payload: nothing about a face or eyes is wired.
            "expression_placeholder": True,
            "output_frames": int(self.output_frames),
            "source_frames": int(self.source_frames),
            "saturated_fraction": round(float(self.saturated_fraction), 6),
            "preserve_first_frame": bool(self.preserve_first_frame),
            "identity": bool(self.identity),
        }


# --------------------------------------------------------------------------- #
# rotation helpers (matrix <-> quaternion, and angle scaling)
# --------------------------------------------------------------------------- #
def _matrices_to_quaternions(matrices: np.ndarray) -> np.ndarray:
    """``(..., 3, 3)`` rotation matrices -> ``(..., 4)`` unit quaternions ``(w, x, y, z)``.

    Shepperd's method: four candidate branches, and the branch with the largest
    denominator is the numerically stable one. Branching on the trace alone (the usual
    short implementation) loses precision exactly at 180 degrees, which is where a
    saturated amplitude lands.
    """
    eps = 1e-12
    m00, m11, m22 = matrices[..., 0, 0], matrices[..., 1, 1], matrices[..., 2, 2]
    trace = m00 + m11 + m22
    s0 = np.sqrt(np.maximum(trace + 1.0, 0.0)) * 2.0
    s1 = np.sqrt(np.maximum(1.0 + m00 - m11 - m22, 0.0)) * 2.0
    s2 = np.sqrt(np.maximum(1.0 - m00 + m11 - m22, 0.0)) * 2.0
    s3 = np.sqrt(np.maximum(1.0 - m00 - m11 + m22, 0.0)) * 2.0
    d0 = np.maximum(s0, eps)
    d1 = np.maximum(s1, eps)
    d2 = np.maximum(s2, eps)
    d3 = np.maximum(s3, eps)

    q0 = np.stack([0.25 * s0, (matrices[..., 2, 1] - matrices[..., 1, 2]) / d0,
                   (matrices[..., 0, 2] - matrices[..., 2, 0]) / d0,
                   (matrices[..., 1, 0] - matrices[..., 0, 1]) / d0], axis=-1)
    q1 = np.stack([(matrices[..., 2, 1] - matrices[..., 1, 2]) / d1, 0.25 * s1,
                   (matrices[..., 0, 1] + matrices[..., 1, 0]) / d1,
                   (matrices[..., 0, 2] + matrices[..., 2, 0]) / d1], axis=-1)
    q2 = np.stack([(matrices[..., 0, 2] - matrices[..., 2, 0]) / d2,
                   (matrices[..., 0, 1] + matrices[..., 1, 0]) / d2, 0.25 * s2,
                   (matrices[..., 1, 2] + matrices[..., 2, 1]) / d2], axis=-1)
    q3 = np.stack([(matrices[..., 1, 0] - matrices[..., 0, 1]) / d3,
                   (matrices[..., 0, 2] + matrices[..., 2, 0]) / d3,
                   (matrices[..., 1, 2] + matrices[..., 2, 1]) / d3, 0.25 * s3], axis=-1)

    which = np.argmax(np.stack([s0, s1, s2, s3], axis=-1), axis=-1)
    chosen = np.where(
        (which == 0)[..., None], q0,
        np.where((which == 1)[..., None], q1, np.where((which == 2)[..., None], q2, q3)),
    )
    norm = np.linalg.norm(chosen, axis=-1, keepdims=True)
    quaternions = chosen / np.maximum(norm, eps)
    # q and -q are the same rotation; forcing w >= 0 makes the principal angle fall in
    # [0, pi], which is the range the amplitude statistic measures.
    sign = np.where(quaternions[..., :1] < 0.0, -1.0, 1.0)
    return quaternions * sign


def _quaternions_to_matrices(quaternions: np.ndarray) -> np.ndarray:
    """``(..., 4)`` unit quaternions -> ``(..., 3, 3)`` rotation matrices."""
    w, x, y, z = (quaternions[..., i] for i in range(4))
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    one = np.ones_like(w)
    return np.stack(
        [
            np.stack([one - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)], axis=-1),
            np.stack([2 * (xy + wz), one - 2 * (xx + zz), 2 * (yz - wx)], axis=-1),
            np.stack([2 * (xz - wy), 2 * (yz + wx), one - 2 * (xx + yy)], axis=-1),
        ],
        axis=-2,
    )


def scale_rotation_angles(matrices: np.ndarray, amplitude: float) -> Tuple[np.ndarray, float]:
    """Scale every rotation's angle about the identity, returning ``(matrices, saturated)``.

    ``R'`` keeps ``R``'s axis and carries angle ``amplitude * theta``, clamped to
    ``pi``: a rotation cannot represent more than half a turn in its principal angle,
    and silently wrapping would make the amplitude statistic non-monotone. The clamped
    fraction is returned so a caller can report saturation instead of assuming none.
    """
    quaternions = _matrices_to_quaternions(matrices)
    vector = quaternions[..., 1:]
    norm = np.linalg.norm(vector, axis=-1, keepdims=True)
    # 2*atan2(|v|, w) is stable for both tiny and near-pi angles; acos(w) is not.
    theta = 2.0 * np.arctan2(norm[..., 0], quaternions[..., 0])
    scaled = amplitude * theta
    saturated = float(np.mean(scaled > np.pi)) if scaled.size else 0.0
    scaled = np.minimum(scaled, np.pi)

    safe = np.maximum(norm[..., 0], _ZERO_ANGLE)
    axis = vector / safe[..., None]
    half = scaled / 2.0
    out = np.empty_like(quaternions)
    out[..., 0] = np.cos(half)
    out[..., 1:] = np.sin(half)[..., None] * axis
    # A zero-angle rotation has no axis; its scaling is the identity by definition.
    degenerate = norm[..., 0] < _ZERO_ANGLE
    if np.any(degenerate):
        out[degenerate] = np.array([1.0, 0.0, 0.0, 0.0])
    return _quaternions_to_matrices(out), saturated


# --------------------------------------------------------------------------- #
# the transform
# --------------------------------------------------------------------------- #
def _as_batch(frames: np.ndarray) -> Tuple[np.ndarray, bool]:
    array = np.asarray(frames, dtype=np.float64)
    if array.ndim == 2:
        return array[None, ...], True
    if array.ndim == 3:
        return array, False
    raise ControlError(
        "control_frames_rank_invalid",
        f"frames must be (T, D) or (B, T, D), got shape {np.shape(frames)}",
    )


def _resample_time(source: np.ndarray, speed: float, output_frames: int, layout) -> np.ndarray:
    """Linear resample of the time axis, reading ``source`` at rate ``speed``.

    Foot contacts are picked, never interpolated: the four channels are booleans, and a
    linear blend would invent a half-contact state that the contract's consumers read as
    a real value.
    """
    total = source.shape[1]
    index = np.arange(output_frames, dtype=np.float64) * speed
    if index.size and index[-1] > total - 1 + 1e-9:
        raise ControlError(
            "control_source_too_short",
            f"reading {output_frames} frames at speed {speed} needs source frame "
            f"{index[-1]:.2f}, but the source holds {total}",
        )
    low = np.floor(index).astype(np.int64)
    high = np.minimum(low + 1, total - 1)
    weight = (index - low)[None, :, None]
    out = source[:, low, :] * (1.0 - weight) + source[:, high, :] * weight
    nearest = np.where((index - low) < 0.5, low, high)
    contacts = layout.slices["foot_contacts"]
    out[:, :, contacts] = source[:, nearest, :][:, :, contacts]
    return out


def _rescale_velocities(frames: np.ndarray, speed: float, layout) -> None:
    """``velocities`` is metres/second, so a time-axis change scales it by ``speed``.

    Leaving it alone would make the block disagree with differencing the positions it
    sits next to -- a self-inconsistent feature vector, which is worse than either
    choice on its own.
    """
    if speed == 1.0:
        return
    block = layout.slices["velocities"]
    frames[..., block] *= speed


def _renormalize_heading(frames: np.ndarray, layout) -> None:
    """Put the ``(cos, sin)`` heading back on the unit circle after interpolation."""
    block = layout.slices["global_root_heading"]
    heading = frames[..., block]
    norm = np.linalg.norm(heading, axis=-1, keepdims=True)
    degenerate = norm < 1e-9
    normalised = heading / np.where(degenerate, 1.0, norm)
    if np.any(degenerate):
        # A heading that interpolated to nothing has no direction left. ``(1, 0)`` is the
        # only substitute that is still a unit vector, and a caller measuring the clip
        # sees it in the numbers rather than being told the heading was fine.
        normalised = np.where(degenerate, np.array([1.0, 0.0]), normalised)
    frames[..., block] = normalised


def apply_controls(
    frames: Any,
    controls: MotionControls,
    layout,
    *,
    output_frames: Optional[int] = None,
) -> ControlResult:
    """Apply ``controls`` to one clip. Pure: no clock, no IO, no model.

    ``output_frames`` defaults to the source length. When ``speed > 1`` the caller is
    expected to have asked the engine for :func:`required_source_frames` frames; a
    source that is too short raises ``control_source_too_short`` rather than clamping.
    """
    if not isinstance(controls, MotionControls):
        raise ControlError("control_object_invalid", f"controls must be a MotionControls, got {type(controls).__name__}")
    source, was_2d = _as_batch(frames)
    total = source.shape[1]
    wanted = total if output_frames is None else int(output_frames)
    if wanted < 1:
        raise ControlError("control_output_frames_invalid", f"output_frames must be >= 1, got {output_frames}")

    if controls.identity and wanted == total:
        # The negative control of decision 0008 criterion 1: a neutral control set must
        # reproduce the input bit for bit, so nothing here may touch a float.
        out = source.copy()
        return ControlResult(
            frames=out[0] if was_2d else out,
            output_frames=total,
            source_frames=total,
            speed=1.0,
            amplitude=1.0,
            applied=(),
            saturated_fraction=0.0,
            preserve_first_frame=controls.preserve_first_frame,
            identity=True,
        )

    if controls.speed != 1.0:
        out = _resample_time(source, controls.speed, wanted, layout)
    else:
        if wanted > total:
            raise ControlError(
                "control_source_too_short",
                f"{wanted} output frames requested but the source holds {total} and speed is 1.0",
            )
        out = source[:, :wanted, :].copy()

    _rescale_velocities(out, controls.speed, layout)
    if controls.speed != 1.0:
        # Only the interpolation can push the heading off the unit circle. Renormalising a
        # heading that was never touched would move bits for nothing, and on a continuation
        # every unnecessary mutation of the boundary frame is a defect.
        _renormalize_heading(out, layout)

    saturated = 0.0
    if controls.amplitude != 1.0:
        saturated = _apply_amplitude(out, controls, layout)

    out = out.astype(np.float32)
    return ControlResult(
        frames=out[0] if was_2d else out,
        output_frames=wanted,
        source_frames=total,
        speed=controls.speed,
        amplitude=controls.amplitude,
        applied=controls.applied_names(),
        saturated_fraction=saturated,
        preserve_first_frame=controls.preserve_first_frame,
        identity=False,
    )


def _apply_amplitude(frames: np.ndarray, controls: MotionControls, layout) -> float:
    """Scale the ``global_rot_data`` block in place; returns the saturated fraction.

    The 6D/window conversions go through ``model/skeleton.py``'s own helpers rather than
    a private reimplementation, so this deformation cannot drift from the convention the
    checkpoint was trained with.
    """
    from ..model.skeleton import cont6d_to_matrix, matrix_to_cont6d  # local: keeps this module numpy-only
    import mlx.core as mx

    joints = layout.nbjoints
    block = layout.slices["global_rot_data"]
    batch, total, _ = frames.shape

    # The boundary frame is copied out *before* anything is converted: the 6D -> matrix ->
    # 6D round trip re-orthogonalises, so a frame that merely passes through the conversion
    # comes back a few 1e-7 different. On a continuation that is the difference between
    # "0.000 mm" and "0.00024 mm", and the requirement is the former.
    boundary = frames[:, :1, block].copy() if controls.preserve_first_frame else None

    rotation = frames[:, :, block].reshape(batch, total, joints, 6).astype(np.float32)
    matrices = np.asarray(cont6d_to_matrix(mx.array(rotation)), dtype=np.float64)

    if controls.preserve_first_frame and total > 1:
        scaled_tail, saturated = scale_rotation_angles(matrices[:, 1:, :, :, :], controls.amplitude)
        matrices = np.concatenate([matrices[:, :1], scaled_tail], axis=1)
    else:
        matrices, saturated = scale_rotation_angles(matrices, controls.amplitude)

    back = np.asarray(matrix_to_cont6d(mx.array(matrices.astype(np.float32))), dtype=np.float64)
    frames[:, :, block] = back.reshape(batch, total, joints * 6)
    if boundary is not None:
        frames[:, :1, block] = boundary
    return saturated
