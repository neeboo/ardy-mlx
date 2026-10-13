"""Window-arithmetic tests for the autoregressive advance (contract S5).

These are pure index tests: no model, no diffusion. They pin the rules that are
easy to get subtly wrong, including the two "potholes" the contract calls out:

* S5.2 — the batch default history is the **max** (160 for Core/20fps/H40), the
  GUI slider default is the **min** (one patch = 4 frames); do not mix them;
* S5.1 — ``_default_history_frames`` has **no lower clamp**, so a large enough
  horizon makes it return 0 or a negative number. The reference does that, so
  the copy here must too, and the test locks the behaviour instead of "fixing" it.

Reference: ``scripts/generate.py:106-115``, ``ardy_model.py:569-570``, ``:630-634``,
``:655-674``.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import NUM_FRAMES_PER_TOKEN, PRESETS  # noqa: E402
from ardy_mlx.sampling.window import (  # noqa: E402
    accumulated_translation,
    crop_history,
    cropped_history_start,
    default_history_frames,
    extract_root_heading,
    extract_root_pos,
    frame_heading_angles,
    max_window_frames,
    plan_autoregressive_steps,
    plan_window,
    recenter_frames,
    shift_root_xy,
)

CORE = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
G1 = PRESETS["ARDY-G1-RP-25FPS-Horizon52"]


def assert_close(value, expected, tol: float = 1e-6) -> None:
    """Compare an MLX/numpy array or nested list to expected floats.

    ``pytest.approx`` refuses nested sequences, so this goes through numpy.
    """
    converted = np.asarray(value.tolist() if hasattr(value, "tolist") else value, dtype=np.float64)
    np.testing.assert_allclose(converted, np.asarray(expected, dtype=np.float64), rtol=0, atol=tol)


def scalar(value) -> float:
    """``float(mx_array)`` returns a 0-d array; this gives a real python float."""
    return float(np.asarray(value.tolist() if hasattr(value, "tolist") else value))


# --------------------------------------------------------------------------
# contract S5.1/S5.2/S5.6 -- the trained window budget
# --------------------------------------------------------------------------
def test_max_window_frames_matches_the_contract():
    assert max_window_frames(20) == 200  # S5.2
    assert max_window_frames(25) == 248  # S5.6
    assert CORE.max_window_frames == 200
    assert G1.max_window_frames == 248


def test_default_history_frames_is_the_batch_maximum():
    # generate.py:114-115 -- ((max_window_len - horizon) // patch) * patch
    assert default_history_frames(20, 40, 4) == 160  # S5.2 table
    assert default_history_frames(25, 52, 4) == 196  # 248 - 52 = 196
    assert CORE.default_history_frames == 160
    assert G1.default_history_frames == 196
    assert CORE.default_history_frames + CORE.horizon == 200


def test_default_history_frames_has_no_lower_clamp():
    """The reference floors but does not clamp at 0; a faithful copy does the same."""
    # 200 - 500 = -300 -> (-75) * 4 = -300. The reference returns this value.
    assert default_history_frames(20, 500, 4) == -300
    assert default_history_frames(20, 400, 4) == -200
    assert default_history_frames(20, 240, 4) == -40
    assert default_history_frames(20, 200, 4) == 0
    # Not raised, not clamped: this is the pinned (and slightly alarming) behaviour.
    assert default_history_frames(20, 200, 4) >= 0


def test_batch_default_is_not_the_gui_default():
    """S5.2 pothole: max (batch) vs min (GUI slider, one patch) are different numbers."""
    gui_default = NUM_FRAMES_PER_TOKEN  # README.md:194 -- min = patch size
    batch_default = default_history_frames(20, 40, 4)
    assert gui_default == 4
    assert batch_default == 160
    assert gui_default != batch_default


# --------------------------------------------------------------------------
# contract S5.3 -- step planning and cropping
# --------------------------------------------------------------------------
def test_plan_autoregressive_steps_matches_the_ceiling():
    # ardy_model.py:569-570
    assert plan_autoregressive_steps(100, 160, 40) == (-1, 120)  # degenerate but literal
    assert plan_autoregressive_steps(200, 160, 40) == (1, 200)
    assert plan_autoregressive_steps(240, 160, 40) == (2, 240)
    assert plan_autoregressive_steps(241, 160, 40) == (3, 280)
    assert plan_autoregressive_steps(161, 160, 40) == (1, 200)
    for num_frames, hist, horizon in ((240, 160, 40), (1000, 40, 8), (33, 8, 8)):
        steps, produced = plan_autoregressive_steps(num_frames, hist, horizon)
        assert steps == math.ceil((num_frames - hist) / horizon)
        assert produced == steps * horizon + hist


def test_plan_window_matches_history_end_and_start_arithmetic():
    """ardy_model.py:630-634 with crop_history_length = 160.

    ``held_frames`` is the length of the context window the loop currently holds:
    it starts at the 160-frame seed context and grows by one horizon per step, and
    the crop right-aligns on it.
    """
    first = plan_window(0, 160, 40, 160)
    assert (first.history_end, first.history_start, first.history_len) == (160, 0, 160)
    assert (first.generation_start, first.generation_len, first.window_len) == (160, 40, 200)
    assert first.cropped is False

    second = plan_window(1, 200, 40, 160)
    assert (second.history_end, second.history_start, second.history_len) == (200, 40, 160)
    assert (second.generation_start, second.generation_len) == (200, 40)
    assert second.cropped is True

    third = plan_window(2, 240, 40, 160)
    assert (third.history_end, third.history_start) == (240, 80)

    # The window is always exactly history + horizon frames (S5.2 last row).
    for step in range(6):
        plan = plan_window(step, 160 + step * 40, 40, 160)
        assert plan.window_len == 200
        assert plan.history_len == 160
        assert plan.generation_start == plan.history_end


def test_plan_window_without_cropping_keeps_everything():
    plan = plan_window(3, 160 + 3 * 40, 40, None)
    assert (plan.history_start, plan.history_end) == (0, 280)
    assert plan.history_len == 160 + 3 * 40
    assert plan.window_len == plan.history_len + 40
    assert plan.cropped is False


def test_plan_window_clips_at_zero_for_short_sequences():
    # With a 40-frame context and crop 160, max(0, 40 - 160) = 0: nothing cropped.
    plan = plan_window(0, 40, 40, 160)
    assert (plan.history_end, plan.history_start) == (40, 0)
    assert plan.history_len == 40
    assert plan.window_len == 80
    assert plan.cropped is False

    plan = plan_window(1, 80, 40, 160)
    assert (plan.history_end, plan.history_start) == (80, 0)
    assert plan.window_len == 120
    assert plan.cropped is False

    # Once the context passes the crop length the window stops growing.
    plan = plan_window(2, 200, 40, 160)
    assert (plan.history_end, plan.history_start) == (200, 40)
    assert plan.window_len == 200
    assert plan.cropped is True


def test_plan_window_reads_the_held_context_length():
    # A 120-frame context with a 80-frame crop keeps the last 80 frames.
    plan = plan_window(2, 120, 40, 80)
    assert plan.history_end == 120
    assert plan.history_start == 40
    assert plan.history_len == 80

    # A zero-length context is legal (the first step may start from nothing).
    empty = plan_window(0, 0, 40, 160)
    assert (empty.history_end, empty.history_start, empty.history_len) == (0, 0, 0)
    assert empty.window_len == 40

    with pytest.raises(ValueError):
        plan_window(1, -4, 40, 80)


def test_plan_window_rejects_bad_arguments():
    with pytest.raises(ValueError):
        plan_window(-1, 160, 40, 160)
    with pytest.raises(ValueError):
        plan_window(0, 160, 0, 160)
    with pytest.raises(ValueError):
        plan_window(0, 160, 40, -4)


def test_consecutive_windows_slide_by_exactly_one_horizon():
    """Successive plans must move the crop boundary by one horizon, not more."""
    horizon, init_history = 8, 16
    steps, total = plan_autoregressive_steps(init_history + 3 * horizon, init_history, horizon)
    assert steps == 3
    covered: list = []
    for step in range(steps):
        # The context grows by one horizon per step (seed + generated so far).
        plan = plan_window(step, init_history + step * horizon, horizon, init_history)
        covered.append((plan.history_start, plan.history_end))
        assert plan.generation_start == plan.history_end
    assert covered == [(0, 16), (8, 24), (16, 32)]
    # With a longer context the crop boundary itself advances by one horizon.
    later = [
        plan_window(step, 200 + step * horizon, horizon, 32) for step in range(3)
    ]
    assert [p.history_start for p in later] == [168, 176, 184]
    assert [p.history_end for p in later] == [200, 208, 216]


def test_crop_history_keeps_the_tail():
    frames = mx.arange(1 * 10 * 2, dtype=mx.float32).reshape(1, 10, 2)
    cropped = crop_history(frames, 4)
    assert cropped.shape == (1, 4, 2)
    assert cropped.tolist() == frames[:, 6:, :].tolist()
    assert crop_history(frames, None).shape == frames.shape
    assert cropped_history_start(10, 4) == 6
    assert cropped_history_start(10, 40) == 0
    assert cropped_history_start(10, None) == 0


# --------------------------------------------------------------------------
# contract S5.4 -- recentring and translation accumulation
# --------------------------------------------------------------------------
def _frames_from_root_xz(batch: int = 1, length: int = 4, height: float = 1.5) -> mx.array:
    """Frames whose root block is ``[t, height, -t]`` and whose body is a marker."""
    rows = []
    for t in range(length):
        body = [float(t) + 0.25 * d for d in range(CORE.body_dim)]
        rows.append([float(t), height, float(-t)] + [1.0, 0.0] + body)
    value = [rows for _ in range(batch)]
    array = mx.array(value, dtype=mx.float32)
    assert array.shape == (batch, length, CORE.motion_rep_dim)
    return array


def test_root_block_slices_match_the_contract():
    frames = _frames_from_root_xz(1, 3)
    pos = extract_root_pos(frames)
    heading = extract_root_heading(frames)
    assert pos.shape == (1, 3, 3)
    assert heading.shape == (1, 3, 2)
    assert pos[0, 2].tolist() == [2.0, 1.5, -2.0]
    assert heading[0, 0].tolist() == [1.0, 0.0]


def test_frame_heading_angles_recovers_theta_from_cos_sin():
    frames = mx.zeros((1, 2, CORE.motion_rep_dim), dtype=mx.float32)
    frames = frames.at[:, 0, 3:5].add(mx.array([[1.0, 0.0]], dtype=mx.float32))
    frames = frames.at[:, 1, 3:5].add(
        mx.array([[math.cos(0.75), math.sin(0.75)]], dtype=mx.float32)
    )
    angles = frame_heading_angles(frames)
    assert float(angles[0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert float(angles[0, 1]) == pytest.approx(0.75, abs=1e-6)


def test_recenter_frames_puts_the_center_frame_at_the_origin_in_xz_only():
    frames = _frames_from_root_xz(1, 5, height=1.5)
    center = mx.array([2], dtype=mx.int32)
    recentered, center_pos = recenter_frames(frames, center)

    assert center_pos[0].tolist() == [2.0, 0.0, -2.0], "y must be 0 (S2.3)"
    pos = extract_root_pos(recentered)[0].tolist()
    assert_close(pos[2], [0.0, 1.5, 0.0])  # centre frame lands on the origin
    # x/z shift by -center, y is untouched.
    assert_close(pos[0], [-2.0, 1.5, 2.0])
    assert_close(pos[4], [2.0, 1.5, -2.0])
    
    # Only the root block moves: the body is byte-identical.
    assert recentered[..., 5:].tolist() == frames[..., 5:].tolist()


def test_recenter_frames_clamps_a_negative_index():
    frames = _frames_from_root_xz(1, 3)
    recentered, center_pos = recenter_frames(frames, mx.array([-5], dtype=mx.int32))
    assert center_pos[0].tolist() == [0.0, 0.0, 0.0]
    assert recentered.tolist() == frames.tolist()


def test_translation_accumulates_and_reproduces_world_coordinates():
    """Sum of per-step recentring == the single shift that brings the last frame back."""
    frames = _frames_from_root_xz(1, 6)
    total = mx.zeros((1, 3), dtype=mx.float32)
    running = frames
    centers = []
    for index in (1, 3, 5):
        running, center = recenter_frames(running, mx.array([index], dtype=mx.int32))
        centers.append(center)
    total = accumulated_translation(centers)
    restored = shift_root_xy(running, total)
    # 1 + 3 + 5 = 9 accumulated on x, and the root returns to the original frame 5.
    # Each recentring is relative to the *already recentred* sequence, so the
    # accumulated shift is 1 + 2 + 2 = 5 (not 1 + 3 + 5): the later centre frames
    # sit 2 frames after the previous anchor.
    assert_close(total[0], [5.0, 0.0, -5.0])
    assert_close(extract_root_pos(running)[0, 5], [0.0, 1.5, 0.0])

    # Adding the accumulated translation back restores the original world frame
    # exactly (x, y, z), which is what ardy_model.py:679 does at the very end.
    restored_center = extract_root_pos(restored)[0, 5]
    original_center = extract_root_pos(frames)[0, 5]
    assert_close(restored_center, [5.0, 1.5, -5.0])
    assert_close(restored_center, original_center)
    # The whole ground-plane path matches the original too (no y drift).
    assert_close(extract_root_pos(restored)[0], extract_root_pos(frames)[0])


def test_accumulated_translation_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        accumulated_translation([])
    with pytest.raises(ValueError):
        accumulated_translation(
            [mx.zeros((1, 3), dtype=mx.float32), mx.zeros((2, 3), dtype=mx.float32)]
        )


def test_shift_root_xy_rejects_bad_shapes():
    frames = _frames_from_root_xz(1, 2)
    with pytest.raises(ValueError):
        shift_root_xy(frames, mx.zeros((1, 2), dtype=mx.float32))
    with pytest.raises(ValueError):
        shift_root_xy(frames, mx.zeros((2, 3), dtype=mx.float32))


# --------------------------------------------------------------------------
# negative controls
# --------------------------------------------------------------------------
def test_negative_control_using_the_gui_minimum_would_change_the_window():
    """Using the GUI default (patch) instead of the batch max changes the crop."""
    batch_max = default_history_frames(20, 40, 4)
    gui_min = NUM_FRAMES_PER_TOKEN
    step = 3
    from_max = plan_window(step, 160, 40, batch_max)
    from_min = plan_window(step, 160, 40, gui_min)
    assert from_max.history_len == 160
    assert from_min.history_len == gui_min
    assert from_max.window_len == 200
    assert from_min.window_len == 44
    assert from_max.history_start != from_min.history_start


def test_shifting_the_ground_plane_ignores_the_y_component():
    """A non-zero y in the delta must be dropped, not applied (S2.3).

    ``translate_normalized_root_motion`` asserts the translation has no y
    component (``ardy_model.py:29-38``), so a caller that passes one is a bug;
    the helper enforces the contract instead of silently lifting the skeleton.
    """
    frames = _frames_from_root_xz(1, 3, height=1.5)
    baseline = shift_root_xy(frames, mx.array([[1.0, 0.0, 2.0]], dtype=mx.float32))
    with_y = shift_root_xy(frames, mx.array([[1.0, 0.5, 2.0]], dtype=mx.float32))
    assert_close(
        extract_root_pos(baseline)[0],
        [[1.0, 1.5, 2.0], [2.0, 1.5, 1.0], [3.0, 1.5, 0.0]],
    )
    assert extract_root_pos(with_y)[0].tolist() == extract_root_pos(baseline)[0].tolist()
    # A negative control for the control: if the helper *did* apply y, frame 0
    # would be 2.0 tall instead of 1.5.
    assert scalar(extract_root_pos(baseline)[0, 0, 1]) == pytest.approx(1.5, abs=1e-6)
    assert scalar(extract_root_pos(baseline)[0, 0, 1]) != pytest.approx(2.0, abs=1e-6)


def test_negative_control_forgetting_to_accumulate_loses_the_offset():
    """If the per-step translation is dropped, the final frame is not back at 0."""
    frames = _frames_from_root_xz(1, 4)
    running, center = recenter_frames(frames, mx.array([3], dtype=mx.int32))
    with_offset = shift_root_xy(running, accumulated_translation([center]))
    without_offset = running
    assert scalar(extract_root_pos(running)[0, 3, 0]) == pytest.approx(0.0, abs=1e-6)
    assert scalar(extract_root_pos(without_offset)[0, 3, 0]) == pytest.approx(0.0, abs=1e-6)
    # The last frame is 0 in both cases by construction; the *history* is what differs.
    assert scalar(extract_root_pos(with_offset)[0, 0, 0]) == pytest.approx(0.0, abs=1e-6)
    assert scalar(extract_root_pos(without_offset)[0, 0, 0]) == pytest.approx(-3.0, abs=1e-6)


def test_negative_control_heading_not_refreshed_after_crop():
    """S5.5: after cropping, first_heading_angle is the crop start's heading, not [:, 0]."""
    frames = mx.zeros((1, 8, CORE.motion_rep_dim), dtype=mx.float32)
    for t in range(8):
        frames = frames.at[:, t, 3:5].add(
            mx.array([[math.cos(0.1 * t), math.sin(0.1 * t)]], dtype=mx.float32)
        )
    headings = frame_heading_angles(frames)
    crop_history_length = 4
    full_len = frames.shape[1]
    correct = float(headings[0, max(0, full_len - crop_history_length)])
    naively_kept = float(headings[0, 0])
    assert correct == pytest.approx(0.4, abs=1e-6)
    assert naively_kept == pytest.approx(0.0, abs=1e-6)
    assert correct != naively_kept


def test_negative_control_default_history_would_be_wrong_if_clamped():
    """A clamp at 0 would be a deviation from ``generate.py:115``."""
    clamped = max(0, default_history_frames(20, 400, 4))
    faithful = default_history_frames(20, 400, 4)
    assert faithful == -200
    assert clamped == 0
    assert clamped != faithful
