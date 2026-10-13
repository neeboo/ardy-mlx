"""Contract and arithmetic tests for the MLX model port (no checkpoint needed).

Everything here runs against synthetic statistics, so the tests are meaningful in
a fresh clone where ``checkpoints/`` is empty. The checkpoint-dependent alignment
lives in ``tests/test_model_alignment.py``.

These tests pin the *traps* -- the places where a plausible-looking
implementation silently disagrees with the reference. Each one exists because it
actually bit us.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mlx.core as mx  # noqa: E402

from ardy_mlx.io import PRESETS  # noqa: E402
from ardy_mlx.model import fsq as fsq_mod  # noqa: E402
from ardy_mlx.model.diffusion import Diffusion, get_beta_schedule  # noqa: E402
from ardy_mlx.model.motion_rep import MotionRep  # noqa: E402
from ardy_mlx.model.nn import build_additive_mask, sinusoidal_pe_table  # noqa: E402


@pytest.fixture
def stats_dir(tmp_path):
    """A 334-long stats pair, as shipped next to the checkpoint."""
    folder = tmp_path / "motion"
    folder.mkdir()
    rng = np.random.default_rng(0)
    mean = rng.normal(size=334).astype(np.float32)
    std = (0.5 + rng.random(334)).astype(np.float32)
    np.save(folder / "mean.npy", mean)
    np.save(folder / "std.npy", std)
    return folder, mean, std


@pytest.fixture
def motion_rep(stats_dir):
    folder, mean, std = stats_dir
    return MotionRep(folder), mean, std


# --------------------------------------------------------------------- layout
def test_motion_rep_dimensions_match_the_contract(motion_rep):
    mr, _, _ = motion_rep
    layout = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
    assert mr.motion_rep_dim == layout.motion_rep_dim == 330
    assert mr.body_dim == layout.body_dim == 325
    assert mr.global_root_dim == 5
    assert mr.local_root_dim == 4
    assert mr.local_motion_rep_dim == 329
    assert layout.dim_token == 148


def test_full_stats_skip_the_four_local_root_slots(motion_rep):
    """stats are 334 long; the model's features are 330. Slicing is 0..4 + 9..333."""
    mr, mean, std = motion_rep
    assert mr.full_stats.dim == 330
    expected = np.concatenate([mean[:5], mean[9:]])
    np.testing.assert_allclose(mr.full_stats.mean, expected)
    expected_std = np.concatenate([std[:5], std[9:]])
    np.testing.assert_allclose(mr.full_stats.std, expected_std)
    # the 4 skipped slots are exactly the local-root stats
    np.testing.assert_allclose(mr.local_root_stats.mean, mean[5:9])


def test_normalize_uses_eps_adjusted_denominator(motion_rep):
    mr, mean, std = motion_rep
    x = mx.array(mean[:330] if mean.shape[0] == 330 else mr.full_stats.mean)
    z = mr.normalize(x)
    np.testing.assert_allclose(np.asarray(z), np.zeros(330), atol=1e-6)
    # normalise then unnormalise is the identity
    back = np.asarray(mr.unnormalize(z))
    np.testing.assert_allclose(back, np.asarray(mx.array(mr.full_stats.mean)), atol=1e-4)


# ------------------------------------------------------------------- local root
def test_local_root_is_normalised_on_output(motion_rep):
    """global_root_to_local_root(normalized=True) re-normalises its 4-vector.

    This is the single most expensive thing we got wrong: the returned
    condition is in *normalised* space, not metres and m/s.
    """
    mr, _, _ = motion_rep
    root = mx.array(mr.global_root_stats.mean)[None, None, :]  # a constant normalized root
    root = mx.broadcast_to(root, (1, 8, 5))
    got = np.asarray(mr.global_root_to_local_root(root, normalized=True))
    # the same call with normalized=False on the *un-normalised* root gives the
    # physical 4-vector; the normalized path must be exactly that, re-normalised.
    unnormalised = mr.global_root_stats.unnormalize(root)
    physical = np.asarray(mr.global_root_to_local_root(unnormalised, normalized=False))
    expected = np.asarray(mr.local_root_stats.normalize(mx.array(physical)))
    np.testing.assert_allclose(got, expected, atol=1e-6)
    # guard against the old bug, where the physical vector was returned as-is
    assert not np.allclose(got, physical, atol=1e-3)


def test_velocity_last_frame_repeats_the_previous(motion_rep):
    """v[T-1] == v[T-2], not zero -- tools.py pads with zero then copies."""
    mr, _, _ = motion_rep
    from ardy_mlx.model.motion_rep import compute_vel_xyz

    pos = mx.array(np.arange(4 * 3, dtype=np.float32).reshape(1, 4, 1, 3))
    vel = np.asarray(compute_vel_xyz(pos, fps=20.0))[0, :, 0]
    assert vel.shape[0] == 4
    assert not np.allclose(vel[3], 0.0), "last velocity must not be left at the zero pad"
    np.testing.assert_allclose(vel[3], vel[2])


def test_velocity_repeats_at_lengths_minus_one_not_at_the_sequence_end():
    """``lengths`` selects the repeated frame (tools.py:66-79).

    The denoiser passes ``lengths = history_len + generation_len`` while the tensor
    spans ``history_len + generation_len + future_len`` frames. On a single-window
    clip those are equal, so the distinction is invisible; on window 0 of a 4.0 s
    clip ``lengths == 40`` against an 80-frame tensor, and ignoring ``lengths``
    corrupts the local-root condition for **every** generation token (measured:
    ``frames`` error 6.75, i.e. 6750x the tolerance).
    """
    from ardy_mlx.model.motion_rep import compute_vel_xyz

    p = np.arange(8 * 3, dtype=np.float32).reshape(1, 8, 1, 3)
    raw = 20.0 * (p[:, 1:] - p[:, :-1])
    raw = np.concatenate([raw, np.zeros((1, 1, 1, 3), dtype=np.float32)], axis=1)

    default = np.asarray(compute_vel_xyz(mx.array(p), fps=20.0))
    expected_default = raw.copy()
    expected_default[0, 7] = raw[0, 6]
    np.testing.assert_allclose(default, expected_default, atol=1e-5)

    cut = np.asarray(compute_vel_xyz(mx.array(p), fps=20.0, lengths=mx.array([4.0])))
    expected_cut = raw.copy()
    expected_cut[0, 3] = raw[0, 2]
    np.testing.assert_allclose(cut, expected_cut, atol=1e-5)
    # the tail past `lengths` is *not* overwritten, and the two differ
    assert not np.allclose(cut, default)


def test_local_root_only_changes_at_lengths_minus_one(motion_rep):
    """The same rule, one level up: ``global_root_to_local_root(..., lengths=L)``."""
    mr, _, _ = motion_rep
    rng = np.random.default_rng(0)
    root = mx.array(rng.normal(size=(1, 8, 5)).astype(np.float32))
    full = np.asarray(mr.global_root_to_local_root(root, normalized=False))
    cut = np.asarray(mr.global_root_to_local_root(root, normalized=False, lengths=mx.array([4.0])))
    np.testing.assert_allclose(full[0, :3], cut[0, :3], atol=1e-5)
    # indices 4..6 are raw finite differences in both, so they agree
    np.testing.assert_allclose(full[0, 4:7], cut[0, 4:7], atol=1e-5)
    # exactly two frames move: the one that receives the repeated velocity in each
    # variant -- index 3 for lengths=4, index 7 for lengths=None
    assert not np.allclose(full[0, 3], cut[0, 3])
    assert not np.allclose(full[0, 7], cut[0, 7])
    # the y component is a copy of the input and never depends on the velocities
    np.testing.assert_allclose(full[0, :, 3], cut[0, :, 3], atol=0.0)


# ------------------------------------------------------------------------ FSQ
def test_fsq_bound_formula_and_bounds():
    z = mx.array([0.0, 0.05, -0.05, 3.0, -3.0])
    out = np.asarray(fsq_mod.fsq_bound(z))
    # multiples of 1/32 (levels=64 -> half_width 32)
    np.testing.assert_allclose(out * 32, np.round(out * 32), atol=1e-5)
    # the offset makes the range asymmetric: [-1.0, 31/32], not [-1, 1]
    assert out.max() <= 31 / 32 + 1e-6
    assert out.min() >= -1.0 - 1e-6
    # tanh shift is built so that z == 0 maps to exactly 0
    assert out[0] == 0.0


def test_fsq_bound_shift_matches_the_reference_constants():
    shift, half_l = fsq_mod._fsq_shift()
    assert half_l == pytest.approx((64 - 1) * (1 + 1e-3) / 2)
    assert shift == pytest.approx(math.atanh(0.5 / half_l))


def test_re_round_is_not_requantize():
    """re_round is the bare step; requantize wraps it in the post-quant stats."""
    z = mx.array([0.51, -0.51, 0.02])
    out = np.asarray(fsq_mod.re_round(z))
    np.testing.assert_allclose(out, np.array([0.5, -0.5, 0.03125]), atol=1e-6)
    # clamping is real: values outside [-1, 1] saturate at +-1
    sat = np.asarray(fsq_mod.re_round(mx.array([5.0, -5.0])))
    np.testing.assert_allclose(sat, np.array([1.0, -1.0]), atol=1e-6)


# ------------------------------------------------------------------ diffusion
def test_cosine_schedule_matches_the_reference_formula():
    betas = get_beta_schedule(10)

    def alpha_bar(t):
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    expected = [
        min(1 - alpha_bar((i + 1) / 10) / alpha_bar(i / 10), 0.999) for i in range(10)
    ]
    np.testing.assert_allclose(betas, expected, rtol=1e-7)


def test_full_step_schedule_is_the_identity_map():
    d = Diffusion(10)
    use, mapped = d.space_timesteps(10)
    np.testing.assert_array_equal(np.asarray(use), np.arange(10))
    np.testing.assert_array_equal(np.asarray(mapped), np.arange(10))


def test_subsampled_schedule_uses_the_map_tensor():
    """The loop index is not the timestep; map_tensor[] is."""
    d = Diffusion(10)
    use, mapped = d.space_timesteps(5)
    # loop indices are [4,3,2,1,0]; the timesteps actually used are these:
    timesteps = [int(mapped[i]) for i in range(4, -1, -1)]
    assert timesteps == [9, 7, 4, 2, 0], timesteps
    assert len(set(timesteps)) == 5


def test_diffusion_buffers_are_consistent():
    d = Diffusion(10)
    ac = np.asarray(d.alphas_cumprod)
    assert np.all(np.diff(ac) < 0), "alphas_cumprod must decrease"
    np.testing.assert_allclose(np.asarray(d.alphas_cumprod_prev), np.concatenate([[1.0], ac[:-1]]))
    np.testing.assert_allclose(
        np.asarray(d.sqrt_recipm1_alphas_cumprod),
        1.0 / np.sqrt(ac / (1.0 - ac)),
        rtol=1e-6,
    )


# ------------------------------------------------------------------ primitives
def test_sinusoidal_pe_first_row_is_the_textbook_row():
    pe = sinusoidal_pe_table(8, max_len=4)
    np.testing.assert_allclose(pe[0, 0::2], np.zeros(4), atol=1e-7)
    np.testing.assert_allclose(pe[0, 1::2], np.ones(4), atol=1e-7)
    assert pe.dtype == np.float32
    np.testing.assert_allclose(pe[1, 0], np.sin(1.0), rtol=1e-6)


def test_additive_mask_uses_true_equals_ignore():
    kpm = mx.array([[False, True, False]])
    mask = build_additive_mask(3, kpm, None)
    got = np.asarray(mask)
    assert got.shape == (1, 1, 1, 3)
    assert got[0, 0, 0, 1] == float("-inf")
    assert got[0, 0, 0, 0] == 0.0

    causal = build_additive_mask(3, None, mx.array(np.triu(np.ones((3, 3)), 1).astype(bool)))
    got = np.asarray(causal)
    assert got.shape == (3, 3)
    assert got[0, 1] == float("-inf")       # above the diagonal: masked
    assert got[1, 0] == 0.0                 # on/below: allowed
    assert got[0, 0] == 0.0                 # the diagonal attends to itself


def test_additive_mask_returns_none_when_nothing_is_masked():
    assert build_additive_mask(4, None, None) is None
