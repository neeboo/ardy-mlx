"""Tests for the autoregressive chunk loop (contract S5.3-S5.5).

The loop is driven by a *fake* step function so these tests can check the parts
that do not need the diffusion model:

* history entering the model is always **clean** (S5.3: only generation tokens
  are denoised);
* the history crop right-aligns on the current generation start;
* every step recentres on the last generated frame and **accumulates** the
  translation (S5.4), so applying ``global_transl`` at the end brings the last
  frame back to the world frame;
* ``first_heading_angle`` is refreshed from the crop start (S5.5);
* ``offsets`` are the cumulative chunk boundaries (S7);
* the same seed gives the same noise and therefore the same output (S4.5).

The "Core" numbers here are the real ones (20 fps, horizon 40, D=330, history
160), because the arithmetic only means anything at those sizes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import NUM_FRAMES_PER_TOKEN, PRESETS  # noqa: E402
from ardy_mlx.sampling.autoregressive import (  # noqa: E402
    ArdyAutoregressiveConfig,
    ArdySampler,
    DenoiseRequest,
    DenoiseResult,
)
from ardy_mlx.sampling.demo import make_demo_denoise_fn, sanitize_placeholder_output  # noqa: E402
from ardy_mlx.sampling.window import (  # noqa: E402
    extract_root_pos,
    frame_heading_angles,
    plan_window,
    shift_root_xy,
)

CORE = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
HORIZON = CORE.horizon  # 40
INIT_HISTORY = CORE.default_history_frames  # 160


def scalar(value) -> float:
    """``float(mx_array)`` returns a 0-d array; this gives a real python float."""
    return float(np.asarray(value.tolist() if hasattr(value, "tolist") else value))


def assert_close(value, expected, tol: float = 1e-5) -> None:
    converted = np.asarray(value.tolist() if hasattr(value, "tolist") else value, dtype=np.float64)
    np.testing.assert_allclose(converted, np.asarray(expected, dtype=np.float64), rtol=0, atol=tol)


class RecordingStep:
    """Deterministic stand-in for the denoise+detokenize step.

    The generated root position is ``(index, 1.25, -index)`` with ``index`` the
    global frame number and heading ``(1, 0)``, so the recentring arithmetic has
    exact values to check; the body is a constant. Every history window and every
    noise tensor handed to it is recorded, which is how the "clean history" and
    "noise is generation-only" invariants are checked.
    """

    def __init__(self, layout=CORE, base_index: int = 0):
        self.layout = layout
        # Absolute index of the first generated frame; the seed context occupies
        # 0 .. base_index - 1, so generated indices stay continuous with it.
        self.base_index = base_index
        self.seen_history: list = []
        self.seen_noise: list = []
        self.seen_requests: list = []

    def __call__(self, request: DenoiseRequest) -> DenoiseResult:
        self.seen_history.append(request.history_frames)
        self.seen_noise.append(request.generation_noise)
        self.seen_requests.append(request)
        # Generate in the *working frame*: continue one frame at a time from the last
        # root x the history ended on (the seed context contributes 0). This is what
        # makes the chunk continuous with its context, which is what the real
        # denoiser has to do.
        last_history_x = float(request.history_frames[0, -1, 0]) if request.history_len else 0.0
        rows = []
        for local in range(request.generation_len):
            index = last_history_x + 1.0 + local
            body = [0.5] * self.layout.body_dim
            rows.append([index, 1.25, -index, 1.0, 0.0] + body)
        generated = mx.array([rows] * request.history_frames.shape[0], dtype=mx.float32)
        expected = (
            request.history_frames.shape[0],
            request.generation_len,
            self.layout.motion_rep_dim,
        )
        assert generated.shape == expected, f"{generated.shape} != {expected}"
        return DenoiseResult(generated_frames=generated)


def config(**overrides) -> ArdyAutoregressiveConfig:
    base = dict(
        layout=CORE,
        num_frames=INIT_HISTORY + 2 * HORIZON,  # 240 -> exactly two chunks
        batch_size=1,
        history_frames=INIT_HISTORY,
        num_steps=10,
        seed=0,
    )
    base.update(overrides)
    return ArdyAutoregressiveConfig(**base)


# --------------------------------------------------------------------------
# planning
# --------------------------------------------------------------------------
def test_step_count_and_offsets_follow_the_contract():
    sampler = ArdySampler(config(), RecordingStep(base_index=INIT_HISTORY))
    assert sampler.num_autoregressive_steps == 2
    result = sampler.run()
    # Two chunks of 40 frames; the seed context is not part of the clip.
    assert result.num_frames == 80
    assert result.offsets == [0, 40, 80]
    assert result.num_steps == 10
    assert result.cfg_weight == (2.0, 2.0)


def test_offsets_are_cumulative_and_end_at_the_total():
    result = ArdySampler(
        config(num_frames=INIT_HISTORY + 3 * HORIZON), RecordingStep(base_index=INIT_HISTORY)
    ).run()
    assert result.offsets == [0, 40, 80, 120]
    assert result.offsets[0] == 0
    assert result.offsets[-1] == result.num_frames
    assert result.num_frames == 3 * HORIZON


def test_window_lengths_are_history_plus_horizon_at_every_step():
    result = ArdySampler(
        config(num_frames=INIT_HISTORY + 3 * HORIZON), RecordingStep(base_index=INIT_HISTORY)
    ).run()
    assert [(d.history_len, d.generation_len, d.window_len) for d in result.diagnostics] == [
        (160, 40, 200),
        (160, 40, 200),
        (160, 40, 200),
    ]
    # history_start is the crop boundary inside the trajectory once it is long
    # enough to crop (step 2: 80 frames generated, keep the last 160 of the
    # context -> the window spills back into the seed history).
    assert [d.history_start for d in result.diagnostics] == [0, 0, 0]
    assert [d.history_end for d in result.diagnostics] == [0, 40, 80]
    assert [d.generation_start for d in result.diagnostics] == [0, 40, 80]
    # Cropping only starts once the trajectory alone exceeds the crop length.
    assert [d.cropped for d in result.diagnostics] == [False, False, False]


# --------------------------------------------------------------------------
# contract S5.3 -- history stays clean and equals the recentred preceding frames
# --------------------------------------------------------------------------
def test_history_fed_to_the_model_is_the_clean_preceding_sequence():
    """The history handed to the model is the clean tail, not a noised or new tensor.

    Checks per step:

    * the history matches the corresponding suffix of the sequence the loop built,
      with only the root x/z allowed to differ by a single constant translation
      (the recentring); the body must match byte for byte. This is what makes
      "history stays clean" falsifiable -- noised history or a regenerated chunk
      would not match any suffix;
    * whatever part of the window is still seed context is exactly zero.
    """
    step = RecordingStep(base_index=INIT_HISTORY)
    sampler = ArdySampler(config(num_frames=INIT_HISTORY + 3 * HORIZON), step)
    result = sampler.run()
    assert len(step.seen_history) == 3

    # Step 0: the window is nothing but the seed context, all zeros.
    assert step.seen_history[0].shape == (1, INIT_HISTORY, CORE.motion_rep_dim)
    assert float(mx.max(mx.abs(step.seen_history[0]))) == 0.0

    for index, seen in enumerate(step.seen_history):
        newest = result.diagnostics[index].generation_start
        overlap = min(newest, seen.shape[1])
        if overlap == 0:
            continue
        reference = result.frames[:, newest - overlap : newest, :]
        actual = seen[:, -overlap:, :]

        # Body must be identical: only root x/z may be translated.
        assert_close(actual[..., 5:], reference[..., 5:], tol=0.0)

        # Root x/z must differ by one constant translation per axis.
        for axis in (0, 2):
            delta = np.asarray((actual[..., axis] - reference[..., axis]).tolist(), dtype=np.float64)
            assert np.abs(delta - delta[0, 0]).max() < 1e-5, (
                f"step {index}: root axis {axis} differs by more than a rigid translation"
            )
        # Height is never touched by recentring.
        assert_close(actual[..., 1], reference[..., 1], tol=0.0)

        # The frames before the overlap are the seed context and must still be zero.
        if seen.shape[1] > overlap:
            assert float(mx.max(mx.abs(seen[:, : seen.shape[1] - overlap, :]))) == 0.0

    # What the loop guarantees about the seam is *interface*, not smoothness: each
    # step receives the clean previous tail plus fresh noise, and the diffusion
    # model is what makes the new chunk continue it. A synthetic generator that
    # ignores its history cannot demonstrate continuity, so that property is left
    # to the golden comparison (docs/status-claude.md, "open items").
    #
    # What can be checked here is the loose end of each step: after recentring, the
    # last generated frame sits at the origin of the working frame.
    assert_close(result.frames[0, -1, 0:3], [0.0, 1.25, 0.0], tol=1e-3)

    # The accumulated translation is the sum of the three recentrings (3 chunks).
    assert_close(result.global_transl[0], [120.0, 0.0, -120.0], tol=1e-3)


def test_noise_is_generation_only_and_never_touches_history():
    step = RecordingStep(base_index=INIT_HISTORY)
    ArdySampler(config(), step).run()
    for noise in step.seen_noise:
        # Horizon 40 / patch 4 = 10 tokens, hybrid width 148.
        assert noise.shape == (1, HORIZON // NUM_FRAMES_PER_TOKEN, CORE.dim_token)


def test_generation_noise_differs_between_steps():
    step = RecordingStep(base_index=INIT_HISTORY)
    ArdySampler(config(), step).run()
    first, second = step.seen_noise
    assert float(mx.max(mx.abs(first - second))) > 1e-6


# --------------------------------------------------------------------------
# contract S5.4/S5.5 -- recentring, accumulation, heading refresh
# --------------------------------------------------------------------------
def test_each_step_recentres_on_its_own_last_generated_frame():
    """Frame ``history_end + horizon - 1`` is at x=0, z=0 after that step.

    Step 0 centres frame 199, step 1 centres frame 239; only the very last
    recentring survives into the final array, so both claims are tested through
    the recorded translations plus the final frame.
    """
    step = RecordingStep(base_index=INIT_HISTORY)
    result = ArdySampler(config(num_frames=INIT_HISTORY + 2 * HORIZON), step).run()
    assert len(result.diagnostics) == 2

    # Step 0's chunk is absolute frames 160..199 (the seed context occupies 0..159),
    # so it centres on frame 199.
    # The synthetic generator continues one frame past the zero seed context and
    # produces x = 1..40 in every chunk, so each recentring translation is 40.
    assert result.diagnostics[0].center_pos[0] == pytest.approx(40.0, abs=1e-5)
    assert result.diagnostics[0].center_pos[2] == pytest.approx(-40.0, abs=1e-5)
    assert result.diagnostics[1].center_pos[0] == pytest.approx(40.0, abs=1e-5)
    assert result.diagnostics[1].center_pos[2] == pytest.approx(-40.0, abs=1e-5)

    # The last frame of the clip is at the origin of the working frame, and the
    # chunk seam is continuous (frame 39 -> 40 differs by exactly one frame).
    final = extract_root_pos(result.frames)[0]
    assert scalar(final[-1, 0]) == pytest.approx(0.0, abs=1e-5)
    assert scalar(final[HORIZON, 0]) - scalar(final[HORIZON - 1, 0]) == pytest.approx(1.0, abs=1e-4)

    # Adding the accumulated translation back recovers absolute world coordinates.
    # In world coordinates the generator emits x = 1..40 per chunk and the clip has
    # two chunks, so the first frame is 1 and the last is 80 (the seed context is not
    # part of the output; see ArdySampleResult).
    translated = extract_root_pos(result.translated())[0]
    assert translated[0].tolist() == pytest.approx([1.0, 1.25, -1.0], abs=1e-4)
    assert translated[-1].tolist() == pytest.approx([80.0, 1.25, -80.0], abs=1e-4)
    # 199 + 79 (see test_translation_accumulates_across_steps for the general rule).
    assert result.global_transl[0].tolist() == pytest.approx([80.0, 0.0, -80.0], abs=1e-4)


def test_translation_accumulates_across_steps():
    result = ArdySampler(config(num_frames=INIT_HISTORY + 3 * HORIZON), RecordingStep(base_index=INIT_HISTORY)).run()
    total = np.asarray(result.global_transl.tolist(), dtype=np.float64)
    per_step = np.asarray([d.center_pos for d in result.diagnostics], dtype=np.float64)
    assert total.shape == (1, 3)
    np.testing.assert_allclose(total[0], per_step.sum(axis=0), rtol=0, atol=1e-5)
    # The y channel is never accumulated (S2.3).
    assert total[0, 1] == 0.0
    assert per_step[:, 1].tolist() == [0.0, 0.0, 0.0]


def test_world_coordinates_are_recovered_by_adding_the_translation():
    result = ArdySampler(config(), RecordingStep(base_index=INIT_HISTORY)).run()
    translated = result.translated()
    delta = np.asarray(
        mx.stack(
            [
                translated[..., 0] - result.frames[..., 0],
                translated[..., 1] - result.frames[..., 1],
                translated[..., 2] - result.frames[..., 2],
            ],
            axis=-1,
        ).tolist(),
        dtype=np.float64,
    )
    expected = np.broadcast_to(
        np.asarray(result.global_transl.tolist(), dtype=np.float64)[:, None, :],
        delta.shape,
    )
    # x/z move by the translation, y does not move at all.
    np.testing.assert_allclose(delta[..., 0], expected[..., 0], atol=1e-4)
    np.testing.assert_allclose(delta[..., 2], expected[..., 2], atol=1e-4)
    np.testing.assert_allclose(delta[..., 1], 0.0, atol=1e-9)


def test_first_heading_angle_is_read_from_the_crop_start():
    """S5.5: ``first_heading_angle = heading_angles[:, max(0, len - crop)]``."""
    result = ArdySampler(config(num_frames=INIT_HISTORY + 3 * HORIZON), RecordingStep(base_index=INIT_HISTORY)).run()
    headings = frame_heading_angles(result.frames)
    assert result.first_heading_angle.shape == (1,)
    for diagnostic in result.diagnostics:
        assert float(result.first_heading_angle[0]) == pytest.approx(
            float(headings[0, diagnostic.history_start]), abs=1e-6
        )


def test_heading_refresh_uses_the_crop_start_not_frame_zero():
    """With a slanted heading path the crop-start rule is observable."""
    layout = CORE

    class SlantedStep:
        def __call__(self, request: DenoiseRequest) -> DenoiseResult:
            rows = []
            for local in range(request.generation_len):
                index = request.history_len + local
                angle = 0.25 * float(index)
                body = [0.5] * layout.body_dim
                rows.append(
                    [float(index), 1.25, -float(index), float(np.cos(angle)), float(np.sin(angle))]
                    + body
                )
            return DenoiseResult(generated_frames=mx.array([rows], dtype=mx.float32))

    result = ArdySampler(
        config(num_frames=INIT_HISTORY + 2 * HORIZON), SlantedStep()
    ).run()
    headings = frame_heading_angles(result.frames)
    # ``first_heading_angle`` is the first frame of the context the last step saw,
    # which for this short run is the seed context (heading 0).
    assert result.diagnostics[-1].history_start == 0
    assert float(result.first_heading_angle[0]) == pytest.approx(0.0, abs=1e-6)
    assert float(headings[0, 0]) != pytest.approx(0.0, abs=1e-3)


# --------------------------------------------------------------------------
# determinism and configuration
# --------------------------------------------------------------------------
def test_same_seed_gives_an_identical_run():
    first = ArdySampler(config(seed=1234), RecordingStep(base_index=INIT_HISTORY)).run()
    second = ArdySampler(config(seed=1234), RecordingStep(base_index=INIT_HISTORY)).run()
    assert first.frames.tolist() == second.frames.tolist()
    assert first.global_transl.tolist() == second.global_transl.tolist()


def test_different_seed_changes_the_noise():
    step_a = RecordingStep(base_index=INIT_HISTORY)
    step_b = RecordingStep(base_index=INIT_HISTORY)
    ArdySampler(config(seed=1), step_a).run()
    ArdySampler(config(seed=2), step_b).run()
    assert step_a.seen_noise[0].tolist() != step_b.seen_noise[0].tolist()


def test_config_rejects_a_history_length_that_is_not_a_patch_multiple():
    with pytest.raises(ValueError):
        config(history_frames=10)
    with pytest.raises(ValueError):
        config(history_frames=-4)


def test_uncropped_history_keeps_growing():
    step = RecordingStep(base_index=INIT_HISTORY)
    result = ArdySampler(
        config(num_frames=INIT_HISTORY + 3 * HORIZON, history_frames=None), step
    ).run()
    # Without cropping the context is the seed history plus everything generated:
    # step 0 waits on the 160-frame seed, then 200, then 240.
    assert [d.history_len for d in result.diagnostics] == [160, 200, 240]
    assert [d.window_len for d in result.diagnostics] == [200, 240, 280]
    assert all(not d.cropped for d in result.diagnostics)
    assert step.seen_history[0].shape == (1, 160, CORE.motion_rep_dim)
    assert step.seen_history[1].shape == (1, 200, CORE.motion_rep_dim)
    assert step.seen_history[2].shape == (1, 240, CORE.motion_rep_dim)


def test_recenter_hook_is_called_once_per_step():
    seen = []
    sampler = ArdySampler(
        config(num_frames=INIT_HISTORY + 2 * HORIZON),
        RecordingStep(base_index=INIT_HISTORY),
        recenter_hook=lambda frames, center, step: seen.append(
            (step, frames.shape, center.tolist())
        ),
    )
    sampler.run()
    assert [entry[0] for entry in seen] == [0, 1]
    assert [entry[1] for entry in seen] == [
        (1, HORIZON, CORE.motion_rep_dim),
        (1, 2 * HORIZON, CORE.motion_rep_dim),
    ]


def test_step_fn_result_shape_is_validated():
    class BadStep:
        def __call__(self, request: DenoiseRequest) -> DenoiseResult:
            return DenoiseResult(generated_frames=mx.zeros((1, 4, CORE.motion_rep_dim)))

    with pytest.raises(ValueError):
        ArdySampler(config(), BadStep()).run()


# --------------------------------------------------------------------------
# placeholder backend end to end (fast mode)
# --------------------------------------------------------------------------
def test_placeholder_backend_runs_end_to_end_and_keeps_shapes():
    sampler = ArdySampler(
        config(num_frames=8 + 3 * 8, history_frames=8, layout=CORE),
        make_demo_denoise_fn(CORE, mode="fast", seed=0),
    )
    result = sampler.run()
    # num_frames 32 with an 8-frame seed context -> ceil(24 / 40) = 1 chunk of 40.
    assert result.frames.shape == (1, HORIZON, CORE.motion_rep_dim)
    assert result.offsets == [0, HORIZON]

    sanitized = sanitize_placeholder_output(result.frames, CORE)
    assert sanitized.shape == result.frames.shape
    contacts = set(
        np.unique(np.asarray(sanitized[..., CORE.slices["foot_contacts"]].tolist())).tolist()
    )
    assert contacts <= {0.0, 1.0}
    heading = np.asarray(sanitized[..., CORE.slices["global_root_heading"]].tolist())
    np.testing.assert_allclose(np.linalg.norm(heading, axis=-1), 1.0, atol=1e-5)


def test_placeholder_backend_is_deterministic_for_a_seed():
    def run():
        return ArdySampler(
            config(num_frames=8 + 2 * 8, history_frames=8, seed=5),
            make_demo_denoise_fn(CORE, mode="fast", seed=5),
        ).run()

    assert run().frames.tolist() == run().frames.tolist()


# --------------------------------------------------------------------------
# negative controls: these tests must be able to fail
# --------------------------------------------------------------------------
def test_negative_control_noised_history_would_break_the_clean_chain():
    """Corrupt one history window and the suffix match fails."""

    class NoisyHistoryStep(RecordingStep):
        def __call__(self, request: DenoiseRequest) -> DenoiseResult:
            result = super().__call__(request)
            if request.step == 1:
                self.seen_history[-1] = self.seen_history[-1] + 0.5
            return result

    step = NoisyHistoryStep(base_index=INIT_HISTORY)
    result = ArdySampler(config(num_frames=INIT_HISTORY + 2 * HORIZON), step).run()

    # The clean windows match their suffix byte for byte in the body; the corrupted
    # one does not.
    for index, seen in enumerate(step.seen_history):
        newest = result.diagnostics[index].generation_start
        overlap = min(newest, seen.shape[1])
        if overlap == 0:
            continue
        body_delta = np.asarray(
            (
                seen[:, -overlap:, 5:]
                - result.frames[:, newest - overlap : newest, 5:]
            ).tolist()
        )
        worst = float(np.abs(body_delta).max())
        if index == 1:
            assert worst >= 0.5, "the corrupted history should be detected"
        else:
            assert worst == 0.0


def test_negative_control_missing_accumulation_would_break_world_recovery():
    result = ArdySampler(config(), RecordingStep(base_index=INIT_HISTORY)).run()
    correct = extract_root_pos(shift_root_xy(result.frames, result.global_transl))
    without = extract_root_pos(result.frames)
    assert float(mx.max(mx.abs(correct - without))) > 1.0


def test_negative_control_left_aligned_crop_is_detectable():
    # A window holding 200 frames with a 160-frame crop keeps the last 160.
    right = plan_window(step=0, held_frames=200, gen_horizon_len=HORIZON, crop_history_length=160)
    assert (right.history_start, right.history_end, right.history_len) == (40, 200, 160)
    # A left-aligned implementation would keep the first 160 instead.
    assert (right.history_start, right.history_end) != (0, 160)


def test_negative_control_gui_minimum_history_is_distinguishable():
    with_min = ArdySampler(
        config(history_frames=NUM_FRAMES_PER_TOKEN), RecordingStep(base_index=INIT_HISTORY)
    ).run()
    with_max = ArdySampler(config(history_frames=INIT_HISTORY), RecordingStep(base_index=INIT_HISTORY)).run()
    assert with_min.diagnostics[0].history_len == NUM_FRAMES_PER_TOKEN
    assert with_max.diagnostics[0].history_len == INIT_HISTORY
    assert with_min.diagnostics[0].window_len != with_max.diagnostics[0].window_len
