"""Contract tests for the ARDY feature layout (see docs/motion-contract.md S1).

These pin the numbers A read out of the reference implementation. If someone
changes a block size or the window arithmetic without changing the contract,
these fail on purpose.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# The package is not installed into the venv (decision 0001 installs only
# pip/mlx/numpy/pytest), so make src/ importable without a conftest at the root.
SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import (  # noqa: E402
    BLOCK_ORDER,
    FOOT_CONTACT_ORDER,
    HYBRID_TOKEN_DIM,
    LATENT_BODY_DIM,
    PRESETS,
    FeatureLayout,
    default_history_frames,
)

CORE = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
G1 = PRESETS["ARDY-G1-RP-25FPS-Horizon52"]


def test_core_dims_match_contract():
    # docs/motion-contract.md S1.3
    assert CORE.motion_rep_dim == 330
    assert CORE.body_dim == 325
    assert CORE.root_block_dim == 5
    assert CORE.local_motion_rep_dim == 329


def test_g1_dims_match_contract():
    assert G1.motion_rep_dim == 414
    assert G1.body_dim == 409
    assert G1.local_motion_rep_dim == 413


def test_block_sum_equals_motion_rep_dim():
    for layout in PRESETS.values():
        assert sum(layout.block_dims.values()) == layout.motion_rep_dim
        # ardy/motion_rep/reps/base.py:46
        assert layout.motion_rep_dim == layout.root_block_dim + layout.body_dim


def test_slices_are_contiguous_and_cover_the_row():
    for layout in PRESETS.values():
        slices = layout.slices
        assert tuple(slices) == BLOCK_ORDER, "block order must match the concatenation order"
        assert slices[BLOCK_ORDER[0]].start == 0
        assert slices[BLOCK_ORDER[-1]].stop == layout.motion_rep_dim
        for earlier, later in zip(BLOCK_ORDER, BLOCK_ORDER[1:]):
            assert slices[earlier].stop == slices[later].start


def test_root_block_is_the_first_five_dims():
    # ardy_motionrep.py:65 => last_root_feature == "global_root_heading"
    for layout in PRESETS.values():
        assert layout.root_slice == slice(0, 5)
        assert layout.body_slice == slice(5, layout.motion_rep_dim)
        slices = layout.slices
        assert slices["root_pos"] == slice(0, 3)
        assert slices["global_root_heading"] == slice(3, 5)


def test_hybrid_token_width():
    # ardy_model.py:479 — nframe_root_dim + latent_embedding_dim == 20 + 128
    for layout in PRESETS.values():
        assert layout.nframe_root_dim == 20
        assert layout.dim_token == 148 == HYBRID_TOKEN_DIM
        assert layout.dim_token == layout.nframe_root_dim + LATENT_BODY_DIM


def test_foot_contacts_are_left_first():
    # feet.py:52 — torch.cat((feet_l, feet_r), axis=-1)
    assert FOOT_CONTACT_ORDER == ("left_heel", "left_toe", "right_heel", "right_toe")
    for layout in PRESETS.values():
        assert layout.slices["foot_contacts"].stop == layout.motion_rep_dim
        assert layout.slices["foot_contacts"].stop - layout.slices["foot_contacts"].start == 4


def test_default_history_frames_batch_path():
    # scripts/generate.py:106-115
    assert CORE.max_window_frames == 200          # (10*20 // 4) * 4
    assert CORE.default_history_frames == 160     # ((200 - 40) // 4) * 4
    assert G1.max_window_frames == 248            # (10*25 // 4) * 4
    assert G1.default_history_frames == 196       # ((248 - 52) // 4) * 4
    assert default_history_frames(20, 40, 4) == 160
    assert default_history_frames(25, 52, 4) == 196


def test_history_plus_horizon_fits_the_trained_window():
    for layout in PRESETS.values():
        assert layout.default_history_frames + layout.horizon <= layout.max_window_frames
        assert layout.default_history_frames % 4 == 0, "must be a whole number of tokens"


def test_history_mirrors_the_reference_formula_without_clamping():
    """This helper is a *faithful* mirror of generate.py:115, which does not clamp.

    A horizon larger than the window budget therefore yields a negative number.
    That cannot happen for any released checkpoint (40 < 200, 52 < 248), so the
    contract requires callers to validate their own (fps, horizon) pair instead
    of relying on this helper to floor at zero.
    """
    assert default_history_frames(20, 400, 4) == -200
    for layout in PRESETS.values():
        assert layout.default_history_frames > 0


@pytest.mark.parametrize("nbjoints,expected", [(27, 330), (34, 414)])
def test_dim_scales_with_joint_count(nbjoints, expected):
    assert FeatureLayout(nbjoints, 20, 40, "synthetic").motion_rep_dim == expected


def test_negative_control_wrong_joint_count_is_not_330():
    """改坏就红: a 30-joint skeleton must not silently look like the Core layout."""
    wrong = FeatureLayout(30, 20, 40, "SOMASkeleton30")
    assert wrong.motion_rep_dim == 5 + (29 * 3) + (30 * 6) + (30 * 3) + 4
    assert wrong.motion_rep_dim != 330
