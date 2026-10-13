"""The control layer, against the criteria frozen in ``docs/decisions/0008``.

These are the tests that decide whether ``speed`` and ``amplitude`` **do** anything.
Each one is written so it can go red in the way that matters: dropping a parameter, or
letting it leak into the other statistic, has to fail here rather than produce a clip
that merely looks different.

Two sources are used, and the difference between them is the point:

* a **deterministic constant-velocity** block, where "displacement ratio = speed ratio"
  is an identity and can therefore be asserted tightly (decision 0008 criterion 2a);
* the **real golden walk**, which has a startup ramp, so the same ratio is not 3 and
  the honest criterion there is monotonicity and clear separation (2b). Those tests skip
  when ``artifacts/golden-run*/out`` is absent -- it is deliberately not tracked.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import mlx.core as mx  # noqa: E402

from ardy_mlx.control import (  # noqa: E402
    CONTROL_AMPLITUDE_RANGE,
    CONTROL_SPEED_RANGE,
    ControlError,
    MotionControls,
    apply_controls,
    displacement_m,
    joint_rotation_deg,
    mean_speed_mps,
    motion_stats,
    required_source_frames,
)
from ardy_mlx.control.motion import scale_rotation_angles  # noqa: E402
from ardy_mlx.io import PRESETS  # noqa: E402
from ardy_mlx.model.skeleton import cont6d_to_matrix, matrix_to_cont6d  # noqa: E402

CORE = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
GOLDEN_2S = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"
GOLDEN_4S = ROOT / "artifacts" / "golden-run-4s" / "out" / "golden_core_h40_4s_seed0.npz"


def _rodrigues(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    cross = np.array(
        [[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]]
    )
    return np.eye(3) + math.sin(angle) * cross + (1.0 - math.cos(angle)) * (cross @ cross)


def build_frames(
    layout,
    num_frames: int,
    *,
    speed_mps: float = 1.0,
    joint_deg: float = 30.0,
) -> np.ndarray:
    """A deterministic ``(1, T, D)`` block: constant root velocity, constant joint angle.

    Constant on purpose. Every claim of the form "the ratio of these two statistics is
    the ratio of the two parameters" is only an identity when the source does not change
    speed underneath the measurement, so the source is built to have no such change and
    the real clip is tested separately for what it actually does.
    """
    slices = layout.slices
    frames = np.zeros((1, num_frames, layout.motion_rep_dim), dtype=np.float32)
    steps = np.arange(num_frames, dtype=np.float64)
    frames[0, :, slices["root_pos"].start + 0] = steps * speed_mps / layout.fps
    frames[0, :, slices["global_root_heading"].start + 0] = 1.0  # (cos, sin) = (1, 0)

    rotation = _rodrigues(np.array([0.0, 0.0, 1.0]), math.radians(joint_deg))
    six = np.asarray(matrix_to_cont6d(mx.array(rotation[None].astype(np.float32))), dtype=np.float32)
    frames[0, :, slices["global_rot_data"]] = np.tile(
        np.tile(six, layout.nbjoints), (num_frames, 1)
    )
    return frames


def golden_frames(path: Path) -> np.ndarray:
    """The real golden clip as a contract ``(1, T, 330)`` block.

    ``velocities`` is left at zero: the reference npz does not carry it, and none of the
    statistics under test reads it (that is deliberate -- :func:`mean_speed_mps` measures
    positions, so it cannot be fooled by a stale velocity block).
    """
    data = np.load(path)
    global_rot = np.asarray(data["global_rot_mats"], dtype=np.float32)
    six = np.asarray(matrix_to_cont6d(mx.array(global_rot)), dtype=np.float32)
    root = np.asarray(data["root_positions"], dtype=np.float32)
    heading = np.asarray(data["global_root_heading"], dtype=np.float32)
    contacts = np.asarray(data["foot_contacts"]).astype(np.float32)
    total = root.shape[0]
    frames = np.zeros((1, total, CORE.motion_rep_dim), dtype=np.float32)
    slices = CORE.slices
    frames[0, :, slices["root_pos"]] = root
    frames[0, :, slices["global_root_heading"]] = heading
    frames[0, :, slices["global_rot_data"]] = six.reshape(total, CORE.nbjoints * 6)
    frames[0, :, slices["foot_contacts"]] = contacts
    return frames


# --------------------------------------------------------------------------- #
# criterion 1 -- the neutral control set is the identity, bit for bit
# --------------------------------------------------------------------------- #
def test_neutral_controls_reproduce_the_input_bit_for_bit():
    source = build_frames(CORE, 40)
    result = apply_controls(source, MotionControls(1.0, 1.0), CORE)
    assert np.array_equal(result.frames, source)
    assert result.applied == ()
    assert result.identity is True


def test_neutral_controls_over_the_real_clip_are_also_bit_identical():
    source = build_frames(CORE, 40) * np.float32(1.0)
    source[:, :, CORE.slices["root_pos"].start] += np.float32(0.0137) * np.arange(40)
    result = apply_controls(source, MotionControls.from_params({"speed": 1.0, "amplitude": 1.0}), CORE)
    assert np.array_equal(result.frames, source)


def test_negative_control_the_neutral_path_is_not_taken_by_accident():
    """The criterion above can go red: a non-neutral request must not come back exact.

    Two calls on the same source differing in one field, plus the awkward-value case
    (float32 numbers a float round trip would move) -- so the identity assertion is a
    property of the neutral path and not of the data happening to be easy to preserve.
    """
    source = build_frames(CORE, 80, speed_mps=1.0)

    neutral = apply_controls(source[:, :40], MotionControls(1.0, 1.0), CORE, output_frames=40)
    assert np.array_equal(neutral.frames, source[:, :40])
    assert neutral.identity is True

    bumped = apply_controls(source[:, :80], MotionControls(1.5, 1.0), CORE, output_frames=40)
    assert bumped.identity is False
    assert not np.array_equal(bumped.frames, source[:, :40])

    awkward = source[:, :40].copy()
    awkward[0, :, CORE.slices["root_pos"].start] = np.float32(1.0000001) * np.arange(
        40, dtype=np.float32
    )
    exact = apply_controls(awkward, MotionControls(1.0, 1.0), CORE, output_frames=40)
    assert np.array_equal(exact.frames, awkward)


# --------------------------------------------------------------------------- #
# criterion 2a -- speed on a constant-velocity source is an exact time scale
# --------------------------------------------------------------------------- #
def test_speed_scales_displacement_and_speed_exactly_on_a_constant_velocity_source():
    source = build_frames(CORE, 80, speed_mps=1.0)
    slow = apply_controls(source[:, :40], MotionControls(speed=0.5), CORE, output_frames=40)
    fast = apply_controls(source[:, :80], MotionControls(speed=1.5), CORE, output_frames=40)

    slow_disp = displacement_m(slow.frames, CORE)
    fast_disp = displacement_m(fast.frames, CORE)
    slow_speed = mean_speed_mps(slow.frames, CORE)
    fast_speed = mean_speed_mps(fast.frames, CORE)

    # 1.0 m/s at 20 fps over 40 frames is 1.95 m of source time; half of it is 0.975.
    assert slow_disp == pytest.approx(0.975, abs=1e-6)
    assert fast_disp == pytest.approx(2.925, abs=1e-6)
    assert fast_disp / slow_disp == pytest.approx(3.0, rel=0.01)
    assert fast_speed / slow_speed == pytest.approx(3.0, rel=0.01)


def test_negative_control_dropping_speed_leaves_the_ratio_at_one():
    source = build_frames(CORE, 80, speed_mps=1.0)
    dropped = apply_controls(source[:, :40], MotionControls(speed=1.0), CORE, output_frames=40)
    assert mean_speed_mps(dropped.frames, CORE) == pytest.approx(1.0, abs=1e-6)
    assert displacement_m(dropped.frames, CORE) == pytest.approx(1.95, abs=1e-6)


def test_required_source_frames_only_expands_above_one():
    assert required_source_frames(40, 1.0, horizon=40) == 40
    assert required_source_frames(40, 0.5, horizon=40) == 40
    # 40 * 1.35 = 54 -> the next whole horizon.
    assert required_source_frames(40, 1.35, horizon=40) == 80


def test_a_source_shorter_than_the_requested_read_fails_by_name():
    source = build_frames(CORE, 20)
    with pytest.raises(ControlError) as excinfo:
        apply_controls(source, MotionControls(speed=1.5), CORE, output_frames=40)
    assert excinfo.value.name == "control_source_too_short"


# --------------------------------------------------------------------------- #
# criterion 2b -- the real clip: monotone and clearly separable
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not GOLDEN_4S.exists(), reason="the 4 s golden run is not on disk")
def test_the_real_walk_separates_by_speed_monotonically():
    source = golden_frames(GOLDEN_4S)
    measured = {}
    for speed in (0.5, 1.0, 1.5):
        needed = required_source_frames(40, speed, horizon=CORE.horizon)
        result = apply_controls(source[:, :needed], MotionControls(speed=speed), CORE, output_frames=40)
        measured[speed] = motion_stats(result.frames, CORE)

    displacements = [measured[s].displacement_m for s in (0.5, 1.0, 1.5)]
    speeds = [measured[s].mean_speed_mps for s in (0.5, 1.0, 1.5)]
    assert displacements == sorted(displacements), displacements
    assert speeds == sorted(speeds), speeds
    # "Clearly separable" rather than "exactly 3": this clip starts near standstill, so
    # half speed keeps the whole clip inside the ramp and the ratio runs high. That is a
    # property of the source, not of the transform, and it is reported rather than assumed.
    assert measured[1.5].mean_speed_mps / measured[0.5].mean_speed_mps > 3.0


@pytest.mark.skipif(not GOLDEN_4S.exists(), reason="the 4 s golden run is not on disk")
def test_the_real_walk_rotation_amplitude_is_unaffected_by_speed():
    """Speed is a time-axis change; it must not move the rotation statistic."""
    source = golden_frames(GOLDEN_4S)
    slow = apply_controls(source[:, :40], MotionControls(speed=0.5), CORE, output_frames=40)
    fast = apply_controls(source[:, :80], MotionControls(speed=1.5), CORE, output_frames=40)
    # Not equal frame for frame -- a resample reads different source frames -- but the
    # *distribution* of joint angles is the same motion, so the means stay close.
    assert joint_rotation_deg(fast.frames, CORE) == pytest.approx(
        joint_rotation_deg(slow.frames, CORE), rel=0.05
    )


# --------------------------------------------------------------------------- #
# criterion 4 and 5 -- amplitude moves rotations, and only rotations
# --------------------------------------------------------------------------- #
def test_amplitude_scales_the_joint_rotation_angle_by_its_own_factor():
    source = build_frames(CORE, 40, joint_deg=30.0)
    half = apply_controls(source, MotionControls(amplitude=0.5), CORE)
    one_and_a_half = apply_controls(source, MotionControls(amplitude=1.5), CORE)

    assert half.saturated_fraction == 0.0
    assert one_and_a_half.saturated_fraction == 0.0
    assert joint_rotation_deg(half.frames, CORE) == pytest.approx(15.0, abs=0.05)
    assert joint_rotation_deg(one_and_a_half.frames, CORE) == pytest.approx(45.0, abs=0.05)
    assert joint_rotation_deg(one_and_a_half.frames, CORE) / joint_rotation_deg(
        half.frames, CORE
    ) == pytest.approx(3.0, rel=0.02)


def test_negative_control_dropping_amplitude_leaves_the_rotation_statistic_alone():
    source = build_frames(CORE, 40, joint_deg=30.0)
    dropped = apply_controls(source, MotionControls(amplitude=1.0), CORE)
    assert joint_rotation_deg(dropped.frames, CORE) == pytest.approx(30.0, abs=1e-4)


def test_amplitude_does_not_touch_translation():
    """Criterion 5: no crosstalk between the two parameters."""
    source = build_frames(CORE, 40, speed_mps=1.0, joint_deg=30.0)
    stats = [
        motion_stats(apply_controls(source, MotionControls(amplitude=a), CORE).frames, CORE)
        for a in (0.5, 1.5)
    ]
    assert stats[0].displacement_m == stats[1].displacement_m
    assert stats[0].mean_speed_mps == stats[1].mean_speed_mps


@pytest.mark.skipif(not GOLDEN_4S.exists(), reason="the 4 s golden run is not on disk")
def test_amplitude_on_the_real_clip_is_exact_and_free_of_saturation():
    source = golden_frames(GOLDEN_4S)[:, :40]
    baseline = joint_rotation_deg(source, CORE)
    results = {
        a: apply_controls(source, MotionControls(amplitude=a), CORE) for a in (0.5, 1.5)
    }
    assert all(result.saturated_fraction == 0.0 for result in results.values())
    low = joint_rotation_deg(results[0.5].frames, CORE)
    high = joint_rotation_deg(results[1.5].frames, CORE)
    # The reference walk already reaches 89 degrees per joint, so 1.5x stays under a half
    # turn per joint and the scaling is exactly linear in the angle.
    assert high / low == pytest.approx(3.0, rel=0.01)
    assert low == pytest.approx(baseline * 0.5, rel=0.01)


def test_saturation_is_reported_rather_than_silently_wrapped():
    source = build_frames(CORE, 8, joint_deg=150.0)
    result = apply_controls(source, MotionControls(amplitude=2.0), CORE)
    assert result.saturated_fraction > 0.0
    # Clamped at a half turn, so the statistic sits exactly at the clamp, not past it.
    assert joint_rotation_deg(result.frames, CORE) <= 180.0 + 1e-6


def test_rotation_conversion_round_trips_at_amplitude_one():
    """The quaternion path must reproduce a rotation it was not asked to change."""
    matrices = np.asarray(
        cont6d_to_matrix(
            mx.array(build_frames(CORE, 4, joint_deg=37.0)[0, :, CORE.slices["global_rot_data"]]
                     .reshape(4, CORE.nbjoints, 6).astype(np.float32))
        ),
        dtype=np.float64,
    )
    scaled, saturated = scale_rotation_angles(matrices, 1.0)
    assert saturated == 0.0
    assert np.allclose(scaled, matrices, atol=1e-9)


# --------------------------------------------------------------------------- #
# criterion 6 -- the continuation boundary frame
# --------------------------------------------------------------------------- #
def test_preserve_first_frame_leaves_frame_zero_untouched():
    source = build_frames(CORE, 24, joint_deg=40.0)
    block = CORE.slices["global_rot_data"]

    preserved = apply_controls(
        source, MotionControls(amplitude=1.5, preserve_first_frame=True), CORE
    )
    assert np.array_equal(preserved.frames[0, 0, block], source[0, 0, block])
    # ... and the rest of the clip really did change, or the exemption would be a no-op.
    assert not np.array_equal(preserved.frames[0, 1:, block], source[0, 1:, block])

    unguarded = apply_controls(source, MotionControls(amplitude=1.5), CORE)
    assert not np.array_equal(unguarded.frames[0, 0, block], source[0, 0, block])


def test_speed_already_preserves_frame_zero_without_an_exemption():
    """Index ``0 * speed`` is index 0, so a time rescale cannot move the boundary frame."""
    source = build_frames(CORE, 80, speed_mps=1.0, joint_deg=30.0)
    result = apply_controls(source, MotionControls(speed=1.5), CORE, output_frames=40)
    assert np.array_equal(result.frames[0, 0], source[0, 0])


# --------------------------------------------------------------------------- #
# criterion 7 -- out of range is named, never clamped
# --------------------------------------------------------------------------- #
def test_out_of_range_parameters_fail_by_name_instead_of_clamping():
    low, high = CONTROL_SPEED_RANGE
    with pytest.raises(ControlError) as too_fast:
        MotionControls(speed=high + 0.5)
    assert too_fast.value.name == "control_speed_out_of_range"

    with pytest.raises(ControlError) as too_slow:
        MotionControls(speed=low - 0.1)
    assert too_slow.value.name == "control_speed_out_of_range"

    low, high = CONTROL_AMPLITUDE_RANGE
    with pytest.raises(ControlError) as too_large:
        MotionControls(amplitude=high + 0.5)
    assert too_large.value.name == "control_amplitude_out_of_range"

    with pytest.raises(ControlError) as zero:
        MotionControls(amplitude=0.0)
    assert zero.value.name == "control_amplitude_out_of_range"


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_non_finite_parameters_fail_by_name(bad):
    with pytest.raises(ControlError) as excinfo:
        MotionControls(speed=bad)
    assert excinfo.value.name == "control_speed_not_finite"


def test_a_non_numeric_parameter_fails_by_name():
    with pytest.raises(ControlError) as excinfo:
        MotionControls(speed="fast")
    assert excinfo.value.name == "control_speed_not_a_number"


def test_an_out_of_bounds_frame_count_fails_by_name():
    with pytest.raises(ControlError) as excinfo:
        apply_controls(build_frames(CORE, 4), MotionControls(1.0, 1.0), CORE, output_frames=0)
    assert excinfo.value.name == "control_output_frames_invalid"


def test_from_params_defaults_missing_values_to_neutral():
    controls = MotionControls.from_params({})
    assert controls.speed == 1.0 and controls.amplitude == 1.0
    assert controls.identity is True


# --------------------------------------------------------------------------- #
# the other blocks, and the placeholder boundary
# --------------------------------------------------------------------------- #
def test_velocities_scale_with_speed_so_the_block_agrees_with_the_positions():
    source = build_frames(CORE, 80)
    source[0, :, CORE.slices["velocities"]] = 0.25
    result = apply_controls(source, MotionControls(speed=1.5), CORE, output_frames=40)
    assert np.allclose(result.frames[0, :, CORE.slices["velocities"]], 0.25 * 1.5, atol=1e-6)


def test_foot_contacts_are_picked_never_interpolated():
    source = build_frames(CORE, 40)
    block = CORE.slices["foot_contacts"]
    source[0, ::2, block] = 1.0
    result = apply_controls(source, MotionControls(speed=0.5), CORE, output_frames=20)
    values = set(np.unique(result.frames[0, :, block]).tolist())
    assert values <= {0.0, 1.0}, values


def test_the_heading_stays_a_unit_vector_after_interpolation():
    source = build_frames(CORE, 40)
    block = CORE.slices["global_root_heading"]
    angle = np.linspace(0.0, math.pi / 2, 40)
    source[0, :, block.start + 0] = np.cos(angle)
    source[0, :, block.start + 1] = np.sin(angle)
    result = apply_controls(source, MotionControls(speed=0.7), CORE, output_frames=40)
    lengths = np.linalg.norm(result.frames[0, :, block], axis=-1)
    assert np.allclose(lengths, 1.0, atol=1e-6)


def test_expression_and_gaze_are_carried_but_never_reported_as_applied():
    controls = MotionControls.from_params(
        {"speed": 1.5, "amplitude": 0.8, "expression": "smile", "gaze": "averted"}
    )
    payload = controls.as_dict()
    assert payload["expression"] == "smile"
    assert payload["gaze"] == "averted"
    assert payload["expression_placeholder"] is True
    result = apply_controls(build_frames(CORE, 80), controls, CORE, output_frames=40)
    assert result.applied == ("speed", "amplitude")
    assert "expression" not in result.applied
    assert "gaze" not in result.applied
    # The rig has no face; a payload that claimed otherwise would be the overclaim.
    assert result.as_dict()["expression_placeholder"] is True
