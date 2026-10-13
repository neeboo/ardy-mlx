"""Multi-window alignment: the 4.0 s (two-window) golden clip.

The single-window clip in ``tests/test_model_alignment.py`` cannot see any of the
multi-window code paths: with 40 frames there is one window, ``future_len == 0``,
``history_token_end == 0``, one recentring step, and a decode whose single chunk is
indistinguishable from a whole-sequence decode. This module covers the second
authorised reference run (``--duration 4.0`` = 80 frames = two windows) and pins
the four paths that only it reaches:

1. **history cropping** -- ``history_start_frame`` / window token count, plus the
   elementwise token masks (``latent_utils.py:192-220``);
2. **per-step recentring with an accumulated translation** -- two ``center_pos``
   values and ``global_transl`` growing across them;
3. **``first_heading_angle`` refresh after cropping** (``ardy_model.py:665-674``) --
   the value fed into window 1 comes from the recentred window-0 root motion;
4. **chunked autoregressive decode** (``latent_utils.py:103-140``).

Every assertion is accompanied by a falsifiable negative control in
``test_L4_negative_controls``. Weight-derived golden data is deliberately not
committed (NOTICE), so without ``artifacts/golden-run-4s/raw`` these tests **skip
with an explicit reason** rather than passing quietly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mlx.core as mx  # noqa: E402

from ardy_mlx.model.ardy import (  # noqa: E402
    Ablation,
    assemble_window_latent,
    autoregressive_chunk_slices,
    convert_frame_mask_to_token_mask,
    get_three_mask_from_len,
    timestep_indices,
)

CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"
RAW4 = ROOT / "artifacts" / "golden-run-4s" / "raw"
NPZ4 = ROOT / "artifacts" / "golden-run-4s" / "out" / "golden_core_h40_4s_seed0.npz"

TOL = 1e-3


def _load_align():
    path = ROOT / "scripts" / "ardy_align.py"
    spec = importlib.util.spec_from_file_location("ardy_align_multiwin", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


align = _load_align()


def _require_golden4():
    missing = []
    if not (CKPT / "mlx" / "denoiser.mlx.safetensors").is_file():
        missing.append(f"{CKPT}/mlx/denoiser.mlx.safetensors (run scripts/ardy_convert.py)")
    if not (RAW4 / "randn_0.npy").is_file():
        missing.append(f"{RAW4}/ (the 4.0 s golden run dumps)")
    if not (RAW4 / "window1_first_heading_in.npy").is_file():
        missing.append(f"{RAW4}/window1_first_heading_in.npy (v2 hook, per-window records)")
    if missing:
        pytest.skip("4.0 s golden data not present: " + "; ".join(missing))


@pytest.fixture(scope="module")
def harness4():
    _require_golden4()
    h = align.Harness(CKPT, RAW4, NPZ4, device="gpu", label="run2")
    h.load()
    return h


@pytest.fixture(scope="module")
def report4(harness4):
    """The clean two-window alignment, computed once for the whole module."""
    return harness4.level4()


# --------------------------------------------------------------------- pure
# These need no checkpoint and no golden data: they pin the index arithmetic
# that the multi-window run exercises.
def test_chunk_slices_cover_every_frame_exactly_once():
    chunks = autoregressive_chunk_slices(token_len=20, gen_tokens=10, crop_tokens=40)
    assert chunks == [(0, 0, 10), (0, 10, 20)]
    covered = [(g_start, g_end) for _, g_start, g_end in chunks]
    assert covered == [(0, 10), (10, 20)]


def test_chunk_slices_truncate_history_when_the_crop_bites():
    # crop_history_length = 4 frames -> 1 token of history for every later chunk
    assert autoregressive_chunk_slices(20, 10, 1) == [(0, 0, 10), (9, 10, 20)]
    # no cropping at all: every chunk still sees the whole sequence from token 0
    assert autoregressive_chunk_slices(20, 10, None) == [(0, 0, 10), (0, 10, 20)]


def test_future_token_mask_is_zero_when_there_are_no_constraints():
    """latent_utils.py:207 -- the trap that only a multi-window clip can expose.

    ``future_len == 40`` frames are masked out on window 0 of the 80-frame clip, but
    ``future_token_mask`` must still be *all zeros* because ``motion_mask is None``
    (the whole no-constraint batch path). Deriving it from ``future_mask`` instead
    changes the denoiser's attention key padding and is caught by
    ``test_L4_negative_controls``.
    """
    history_len = mx.zeros((1,))
    generation_len = mx.full((1,), 40.0)
    future_len = mx.full((1,), 40.0)
    hm, gm, fm = get_three_mask_from_len(history_len, generation_len, future_len, 80)
    assert float(np.asarray(fm).sum()) == 40.0  # 40 future *frames*
    h_tok, g_tok, f_tok = convert_frame_mask_to_token_mask(hm, gm, fm, motion_mask=None)
    assert int(np.asarray(h_tok).sum()) == 0
    assert int(np.asarray(g_tok).sum()) == 10
    assert int(np.asarray(f_tok).sum()) == 0, "future_token_mask must be zero without constraints"
    # and the mask that *would* have been derived from future_mask, to show the gap
    assert int(np.asarray(fm.reshape(1, -1, 4).all(axis=-1)).sum()) == 10


def test_assemble_window_latent_puts_history_in_front_and_noise_in_the_generation_span():
    history = mx.full((1, 2, 3), 1.0)
    x_t = mx.full((1, 2, 3), 9.0)
    x = np.asarray(assemble_window_latent(history, x_t, 2, 4, 5)).reshape(5, 3)
    assert x[0].tolist() == [1.0, 1.0, 1.0]
    assert x[1].tolist() == [1.0, 1.0, 1.0]
    assert x[2].tolist() == [9.0, 9.0, 9.0]
    assert x[3].tolist() == [9.0, 9.0, 9.0]
    assert x[4].tolist() == [0.0, 0.0, 0.0], "the trailing future region stays zero"


def test_timestep_indices_are_descending():
    assert timestep_indices(10) == [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
    with pytest.raises(ValueError):
        timestep_indices(0)


# ------------------------------------------------------- multi-window golden
def test_L4_per_window_geometry_and_masks_match_the_reference(report4):
    assert report4["num_windows"] == 2
    assert len(report4["per_window"]) == 2
    for geom in report4["per_window"]:
        w = geom["window"]
        assert geom["geometry_matches_reference"], geom
        assert geom["history_start_frame"] == geom["history_start_frame_ref"], w
        assert geom["history_end_frame"] == geom["history_end_frame_ref"], w
        assert geom["history_token_end"] == geom["history_token_end_ref"], w
        assert geom["total_tokens"] == geom["total_tokens_ref"], w
        assert geom["future_len"] == geom["future_len_ref"], w
        for key in ("token_history_mask_equal", "token_generation_mask_equal", "token_future_mask_equal"):
            assert geom[key] is True, (w, key, geom)
    # the two windows are genuinely different shapes of problem
    w0, w1 = report4["per_window"]
    assert (w0["history_token_end"], w0["future_len"]) == (0, 40)
    assert (w1["history_token_end"], w1["future_len"]) == (10, 0)
    assert w0["total_tokens"] == w1["total_tokens"] == 20


def test_L4_history_cropping_code_path_is_taken(report4):
    """``crop_history_length = 160`` is passed, so the cropping branch runs.

    Honest limitation, asserted rather than glossed over: with only 80 generated
    frames the 160-frame budget is never reached, so ``history_start_frame`` is 0 in
    both windows and **no frame is actually truncated**. The truncation arithmetic
    itself is covered by ``test_chunk_slices_truncate_history_when_the_crop_bites``
    and by the ``crop=4`` negative control.
    """
    for geom in report4["per_window"]:
        assert geom["history_start_frame"] == 0
        ref = align.read_jsonl(RAW4 / "windows.jsonl")
        assert any(r.get("future_len") is not None for r in ref)
    assert report4["summary"]["crop_history_length"] == 160


def test_L4_window_inputs_and_recentring_match_the_reference(report4):
    """The strongest per-window check: the exact tensor each window starts from."""
    for geom in report4["per_window"]:
        w = geom["window"]
        assert geom["x0"]["max_abs"] <= TOL, (w, geom["x0"])
        for key in ("first_heading_angle_in", "global_transl_in", "center_pos", "recentered_hybrid"):
            st = geom[key]
            assert st["max_abs"] <= TOL, (w, key, st)
        # window 0 has no history, so its starting latent is exactly the reference's
        if w == 0:
            assert geom["x0"]["max_abs"] == 0.0
            assert geom["center_frame_index"] == 39
        else:
            assert geom["center_frame_index"] == 79


def test_L4_first_heading_angle_refresh_is_observable(report4):
    """``ardy_model.py:665-674`` -- multi-window only.

    Window 0 is fed the initial angle (0.0). Window 1 is fed the heading at the
    start of the recentred accumulated root motion, which is **not** zero on this
    clip, so the refresh is doing real work rather than being a no-op.
    """
    w0, w1 = report4["per_window"]
    assert w0["first_heading_angle_in"]["max_abs"] == 0.0  # window 0 gets the initial 0.0
    # my refreshed value for window 0 must equal the reference's window-1 input,
    # which is exactly what the report compares:
    assert w0["first_heading_angle_after"]["max_abs"] <= TOL
    assert w1["first_heading_angle_in"]["max_abs"] <= TOL
    refreshed = float(np.load(RAW4 / "window1_first_heading_in.npy").reshape(-1)[0])
    assert abs(refreshed) > 1e-4, "the refreshed heading must be non-zero for this test to mean anything"


def test_L4_accumulated_translation_matches_at_every_step(report4):
    w0, w1 = report4["per_window"]
    # window 0 starts with no translation at all; the reference dump is zeros
    assert w0["global_transl_in"]["reference_abs_max"] == 0.0
    assert w0["global_transl_in"]["max_abs"] == 0.0
    assert w0["global_transl_after"]["max_abs"] <= TOL
    assert w1["global_transl_in"]["max_abs"] <= TOL
    assert w1["center_pos"]["max_abs"] <= TOL
    # the reference's own invariant: window 1's incoming translation is window 0's
    # accumulated centre position, and it is not zero (so accumulation is observable)
    ref_gt1 = np.load(RAW4 / "window1_global_transl_in.npy").reshape(-1)
    ref_cp0 = np.load(RAW4 / "recenter0_center_pos.npy").reshape(-1)
    np.testing.assert_allclose(ref_gt1, ref_cp0, atol=TOL)
    assert float(np.abs(ref_gt1).max()) > 1e-2


def test_L4_all_twenty_denoiser_steps_align(report4):
    per_step = report4["per_step"]
    assert len(per_step) == 20
    for row in per_step:
        assert row["raw_max_abs"] <= TOL, row
        assert row["cfg_max_abs"] <= TOL, row
        assert row["ddim_max_abs"] <= TOL, row
        assert row["history_tokens_untouched_in_reference"], row
    # a single bad window cannot hide in the maximum
    for w in (0, 1):
        worst = max(r["cfg_max_abs"] for r in per_step if r["window"] == w)
        assert worst <= TOL, (w, worst)


def test_L4_final_clip_and_per_window_chunks_align(report4):
    for name, st in report4["final"].items():
        if not isinstance(st, dict) or "max_abs" not in st:
            continue
        if name in align.DIAGNOSTIC_KEYS:
            # pre-FK placeholder / broken-hierarchy FK control: must be large
            assert st["max_abs"] > TOL, (name, st)
            continue
        assert st["max_abs"] <= TOL, (name, st)
    assert report4["final"]["frames"]["shape"] == [1, 80, 330]
    assert len(report4["chunks"]) == 2
    for chunk in report4["chunks"]:
        for key in ("frames", "root_positions_vs_npz", "heading_vs_npz", "posed_joints_vs_npz"):
            assert chunk[key]["max_abs"] <= TOL, (chunk["window"], key, chunk[key])
    # 0004 ruling 4, multi-window: the FK chain reproduces the golden joints
    assert report4["final"]["posed_joints_vs_npz"]["max_abs"] <= 1e-5
    assert report4["final"]["posed_joints_placeholder_vs_npz"]["max_abs"] > 0.2


def test_L4_chunked_decode_is_required_not_optional(report4):
    """The chunked (autoregressive) decode is not equivalent to a whole-sequence one.

    The Stage-A decoder is causal, so the *body* would agree; what differs is the
    local-root **condition**: ``lengths`` selects which frame receives the repeated
    velocity (``tools.py:66-79``), and ``get_explicit_motion_from_hybrid_autoregressive``
    hands each chunk only ``crop_history_length`` frames of context. On this clip the
    gap is ~4.4e-02, i.e. 44x the tolerance, and the golden agrees with the chunked
    decode (``frames`` above is <= 3.6e-05). The negative control
    ``whole-sequence decode`` confirms the harness sees it.
    """
    cmp_ = report4["decode_chunked_vs_whole"]["comparison"]
    assert cmp_["max_abs"] > TOL, cmp_
    assert report4["final"]["frames"]["max_abs"] <= TOL


def test_L4_negative_controls(harness4):
    """Every multi-window path must have a control that can make the harness fail."""
    rows = {r["control"]: r for r in harness4.negative_controls_multi()}
    must_detect = [
        "x_T += 0.25 (wrong initial noise)",
        "recentre centre one token early (-4 frames)",
        "recentre centre two tokens early (-8 frames)",
        "stale first_heading_angle (refresh disabled)",
        "future_token_mask derived from future_mask",
        "crop_history_length = 4 (truncates the history to one token)",
        "whole-sequence decode instead of the chunked decode",
    ]
    for name in must_detect:
        row = rows[name]
        assert row["detected"], row
        assert row["worst_max_abs"] > TOL, row
    # documented no-ops: the tolerance must NOT be "anything moves" either
    clean = rows["clean baseline (identical to the port's default behaviour)"]
    assert clean["worst_max_abs"] <= TOL
    assert not rows["crop_history_length = 164 (one patch more; no truncation at 80 frames)"]["detected"], (
        "an 80-frame clip never reaches the 160-frame crop budget; a one-patch crop "
        "change is vacuous here and the test must say so rather than pretend otherwise"
    )
    assert not rows["recentre centre one token late (+4 frames; clamps to the reference centre)"]["detected"]
    for row in rows.values():
        assert row["matches_expectation"], row


def test_L4_ablation_defaults_reproduce_the_reference():
    """The negative-control knobs must be inert by default."""
    a = Ablation()
    assert a == Ablation(center_index_offset=0, refresh_first_heading=True,
                         future_token_mask_from_frames=False, autoregressive_decode=True)
