"""Skeleton forward kinematics: unit tests + alignment with the golden run.

``0004`` ruling 4 recorded ``posed_joints`` as a **placeholder** (root-local
joints with the root pasted in) that was **0.132** away from the reference on the
2.0 s clip. This file is the replacement's contract:

* the hierarchy / rest pose come from the reference's own source
  (``CoreSkeleton27`` + ``cskel27/joints.p``) and are transcribed in
  :mod:`ardy_mlx.model.skeleton`;
* :meth:`ardy_mlx.model.skeleton.Skeleton.fk` is a line-by-line port of
  ``skeleton/kinematics.py:batch_rigid_transform``;
* the golden comparison below is the "re-aligned with the golden standard"
  requirement of that ruling: FK driven from the golden clip's own
  ``local_rot_mats`` / ``root_positions`` reproduces the golden ``posed_joints``,
  and FK driven from the *feature vector* reproduces them too.

The golden ``.npz`` and the reference weight dumps are **not committed**
(``NOTICE``: no redistribution of NVIDIA weights or outputs derived from them),
so the golden tests skip with an explicit reason when they are absent. The unit
tests above them need nothing but the source tree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import mlx.core as mx  # noqa: E402

from ardy_mlx.model.skeleton import (  # noqa: E402
    CORE_FOOT_JOINT_NAMES,
    CORE_HIP_JOINT_NAMES,
    CORE_JOINT_NAMES,
    CORE_JOINT_PARENTS,
    cont6d_to_matrix,
    core_skeleton,
    global_rots_to_local_rots,
    matrix_to_cont6d,
    placeholder_posed_joints,
    posed_joints_from_features,
)

CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"
NPZ2S = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"
RAW2S = ROOT / "artifacts" / "golden-run" / "raw" / "motion_normalized.npy"
NPZ4S = ROOT / "artifacts" / "golden-run-4s" / "out" / "golden_core_h40_4s_seed0.npz"
RAW4S = ROOT / "artifacts" / "golden-run-4s" / "raw" / "motion_normalized.npy"

#: FK is float32 matrix algebra on both sides; measured residues are 1e-7..5e-7.
#: Anything structural (a wrong parent, a missing offset) is O(1e-1), so this is
#: still a hundred thousand times tighter than the failure mode it guards.
FK_TOL = 1e-5


# --------------------------------------------------------------------- helpers
def axis_angle_matrix(axis, angle):
    """Rodrigues, so the tests can build rotations without scipy."""
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    k = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ]
    )
    return (np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)).astype(np.float32)


def identity_batch(frames, joints=27):
    return np.broadcast_to(np.eye(3, dtype=np.float32), (frames, joints, 3, 3)).copy()


def rotate_joint(local_rots, joint, matrix):
    out = local_rots.copy()
    out[:, joint] = matrix
    return out


def _descendants(parents, joint):
    kids = [joint]
    changed = True
    while changed:
        changed = False
        for j, p in enumerate(parents):
            if p in kids and j not in kids:
                kids.append(j)
                changed = True
    return set(kids)


# ------------------------------------------------------------------- structure
def test_core_skeleton_matches_the_reference_definition():
    sk = core_skeleton()
    assert sk.nbjoints == 27
    assert sk.joint_names == CORE_JOINT_NAMES
    assert sk.joint_parents == CORE_JOINT_PARENTS
    # exactly one root, and it is Hips (the reference asserts both)
    assert [p for p in sk.joint_parents if p < 0] == [-1]
    assert sk.joint_parents[sk.root_index] == -1
    assert sk.joint_names[sk.root_index] == "Hips"
    assert sk.neutral_joints.shape == (27, 3)
    assert np.array_equal(sk.neutral_joints[sk.root_index], np.zeros(3, dtype=np.float32))
    # foot channel order: left heel, left toe, right heel, right toe (contract S1.4)
    assert sk.foot_joint_names == CORE_FOOT_JOINT_NAMES
    assert [sk.joint_names[i] for i in sk.foot_joint_idx] == [
        "LeftFoot",
        "LeftToeBase",
        "RightFoot",
        "RightToeBase",
    ]
    # hip order is [right, left] -- get_root_heading_angle depends on it
    assert sk.hip_joint_names == CORE_HIP_JOINT_NAMES
    assert [sk.joint_names[i] for i in sk.hip_joint_idx] == ["RightUpLeg", "LeftUpLeg"]


def test_skeleton_rejects_a_non_topological_hierarchy():
    from ardy_mlx.model.skeleton import Skeleton

    with pytest.raises(ValueError):
        Skeleton(
            "bad",
            ["a", "b"],
            [-1, 5],
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        )


# -------------------------------------------------------------------------- FK
def test_fk_with_identity_rotations_reproduces_the_rest_pose():
    """No rotation anywhere: the offsets chain back to the neutral joints."""
    sk = core_skeleton()
    frames = 3
    local = identity_batch(frames)
    root = np.zeros((frames, 3), dtype=np.float32)
    global_rots, posed, posed_noroot = sk.fk(local, root)
    assert np.asarray(posed_noroot).shape == (frames, 27, 3)
    assert np.allclose(np.asarray(posed_noroot), sk.neutral_joints[None], atol=1e-6)
    # the root joint's world position *is* root_positions (column 0 of the clip)
    assert np.array_equal(np.asarray(posed)[:, 0], root)
    assert np.allclose(np.asarray(global_rots), np.eye(3), atol=1e-6)


def test_fk_root_translation_moves_the_whole_skeleton_rigidly():
    sk = core_skeleton()
    local = identity_batch(1)
    a = np.asarray(sk.fk(local, np.zeros((1, 3), dtype=np.float32))[1])
    b = np.asarray(sk.fk(local, np.array([[1.5, -0.25, 2.0]], dtype=np.float32))[1])
    assert np.allclose(b - a[None][0], np.array([1.5, -0.25, 2.0]), atol=1e-6)


def test_fk_root_rotation_rotates_every_joint_about_the_root():
    sk = core_skeleton()
    rot = axis_angle_matrix([0.0, 0.0, 1.0], 0.7)  # 40 degrees about +Y
    local = rotate_joint(identity_batch(1), 0, rot)
    _, posed, _ = sk.fk(local, np.zeros((1, 3), dtype=np.float32))
    expected = sk.neutral_joints @ (rot.T.astype(np.float32))
    assert np.allclose(np.asarray(posed)[0], expected, atol=1e-6)


def test_fk_rotating_a_parent_moves_exactly_its_subtree():
    """The hierarchy is the whole point: rotating a joint must not move uncles."""
    sk = core_skeleton()
    assert sk.joint_parents[19] == 0  # RightUpLeg under Hips
    rot = axis_angle_matrix([1.0, 0.0, 0.0], 0.5)
    base = np.asarray(sk.fk(identity_batch(1), np.zeros((1, 3), dtype=np.float32))[1])[0]
    moved = np.asarray(sk.fk(rotate_joint(identity_batch(1), 19, rot), np.zeros((1, 3), dtype=np.float32))[1])[0]

    subtree = _descendants(sk.joint_parents, 19)
    # every descendant of the rotated joint moves...
    for j in subtree - {19}:
        assert np.abs(moved[j] - base[j]).max() > 1e-3, sk.joint_names[j]
    # ...the joint itself does not (it is the pivot), and nothing else does
    assert np.allclose(moved[19], base[19], atol=1e-6)
    for j in set(range(27)) - subtree:
        assert np.allclose(moved[j], base[j], atol=1e-6), sk.joint_names[j]


def test_fk_respects_leading_batch_dimensions_and_dtype():
    sk = core_skeleton()
    local = identity_batch(2 * 4, joints=27).reshape(2, 4, 27, 3, 3)
    root = np.zeros((2, 4, 3), dtype=np.float32)
    global_rots, posed, posed_noroot = sk.fk(local, root)
    assert global_rots.shape == (2, 4, 27, 3, 3)
    assert posed.shape == (2, 4, 27, 3)
    assert posed_noroot.shape == (2, 4, 27, 3)
    # flat leading dims give the same numbers as the reshaped ones
    flat = np.asarray(sk.fk(local.reshape(8, 27, 3, 3), root.reshape(8, 3))[1]).reshape(2, 4, 27, 3)
    assert np.array_equal(np.asarray(posed), flat)


def test_fk_rejects_the_wrong_joint_count():
    sk = core_skeleton()
    with pytest.raises(ValueError):
        sk.fk(identity_batch(1, joints=26), np.zeros((1, 3), dtype=np.float32))


# ------------------------------------------------------- rotation conversions
def test_cont6d_round_trip_is_orthonormal_and_right_handed():
    sk = core_skeleton()
    assert sk.nbjoints == 27
    rot = axis_angle_matrix([0.3, -0.8, 0.5], 1.1)
    matrices = np.stack([rot, np.eye(3, dtype=np.float32)])
    six = np.asarray(matrix_to_cont6d(mx.array(matrices)))
    # ``matrix_to_cont6d`` takes the first two **columns** (``M[..., 0]`` indexes
    # the last axis in the reference's torch code), not the first two rows.
    assert np.allclose(
        six, np.concatenate([matrices[:, :, 0], matrices[:, :, 1]], axis=-1), atol=1e-7
    )
    back = np.asarray(cont6d_to_matrix(mx.array(six)))
    assert np.allclose(back, matrices, atol=1e-6)
    for m in back:
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-6)
        assert np.linalg.det(m) > 0.99  # right-handed (contract S2.1)


def test_global_rots_to_local_rots_inverts_the_fk_chain():
    sk = core_skeleton()
    rng = np.random.default_rng(3)
    local = np.stack(
        [axis_angle_matrix(rng.normal(size=3), 0.3 * rng.normal()) for _ in range(27)]
    )[None]
    root = rng.normal(size=(1, 3)).astype(np.float32)
    global_rots, _, _ = sk.fk(local, root)
    back = np.asarray(global_rots_to_local_rots(global_rots, sk))
    assert np.abs(back - local).max() < 1e-5
    # the root's local rotation is its global rotation (parent == identity)
    assert np.allclose(back[0, 0], local[0, 0], atol=1e-6)


# --------------------------------------------------- feature -> posed_joints
def test_posed_joints_from_features_uses_root_pos_and_global_rot_blocks():
    sk = core_skeleton()
    rng = np.random.default_rng(11)
    local = np.stack(
        [axis_angle_matrix(rng.normal(size=3), 0.2 * rng.normal()) for _ in range(27)]
    )[None]
    root = rng.normal(size=(1, 3)).astype(np.float32)
    global_rots, posed, _ = sk.fk(local, root)

    frames = np.zeros((1, 330), dtype=np.float32)
    frames[0, 0:3] = root
    frames[0, 83 : 83 + 27 * 6] = np.asarray(matrix_to_cont6d(global_rots)).reshape(-1)
    got = np.asarray(posed_joints_from_features(mx.array(frames)))
    assert np.abs(got - np.asarray(posed)).max() < 1e-5
    assert np.array_equal(got[:, 0], root)


def test_placeholder_is_not_the_fk_result():
    """Guard for ``0004`` ruling 4: the old placeholder must not pass as FK.

    The point of the test is that the two are *structurally* different, not that
    the difference is large on any particular input -- but on this input it is
    already O(1e-1), which is what the ruling measured against the golden clip.
    """
    sk = core_skeleton()
    rng = np.random.default_rng(5)
    local = np.stack(
        [axis_angle_matrix(rng.normal(size=3), 0.4 * rng.normal()) for _ in range(27)]
    )[None]
    root = rng.normal(size=(1, 3)).astype(np.float32)
    global_rots, posed, _ = sk.fk(local, root)
    frames = np.zeros((1, 330), dtype=np.float32)
    frames[0, 0:3] = root
    frames[0, 83 : 83 + 27 * 6] = np.asarray(matrix_to_cont6d(global_rots)).reshape(-1)
    # the placeholder's own two inputs, taken from the same clip
    posed_noroot = np.asarray(sk.fk(local, root)[2])
    frames[0, 5 : 5 + 26 * 3] = posed_noroot[0, 1:].reshape(-1)
    placeholder = np.asarray(placeholder_posed_joints(mx.array(frames)))
    assert np.abs(placeholder - np.asarray(posed)).max() > 1e-2


# ------------------------------------------------------------ golden alignment
def _require_golden_npz():
    missing = [str(p) for p in (NPZ2S, NPZ4S) if not p.is_file()]
    if missing:
        pytest.skip(
            "golden npz not present (not committed: NOTICE forbids redistributing "
            "outputs derived from NVIDIA weights): " + "; ".join(missing)
        )


@pytest.mark.parametrize(
    "npz_path,raw_path,label",
    [
        (NPZ2S, RAW2S, "2.0s/40f/1window"),
        (NPZ4S, RAW4S, "4.0s/80f/2windows"),
    ],
)
def test_fk_reproduces_the_golden_posed_joints(npz_path, raw_path, label):
    """``0004`` ruling 4's re-alignment: FK vs the reference's own ``posed_joints``."""
    _require_golden_npz()
    sk = core_skeleton()
    z = np.load(npz_path)
    golden = z["posed_joints"]
    assert golden.shape[1:] == (27, 3), z["posed_joints"].shape

    # (a) FK straight from the reference's own local rotations + root positions
    global_rots, posed, posed_noroot = sk.fk(
        z["local_rot_mats"], z["root_positions"]
    )
    assert np.abs(np.asarray(posed) - golden).max() < FK_TOL, label
    assert np.abs(np.asarray(global_rots) - z["global_rot_mats"]).max() < FK_TOL, label
    # posed_joints_norootpos is the FK result before the root translation is added
    assert np.abs(np.asarray(posed_noroot) + z["root_positions"][:, None] - golden).max() < FK_TOL
    # the root joint column is exactly root_positions (bit-for-bit, by construction)
    assert np.array_equal(np.asarray(posed)[:, 0], z["root_positions"])

    # (b) the chain the port actually ships: contract feature vector -> FK
    if raw_path.is_file() and (CKPT / "stats" / "motion" / "mean.npy").is_file():
        mean = np.load(CKPT / "stats" / "motion" / "mean.npy")
        std = np.load(CKPT / "stats" / "motion" / "std.npy")
        sel = list(range(5)) + list(range(9, mean.shape[0]))
        feats = np.load(raw_path)[0] * np.sqrt(std[sel].astype(np.float32) ** 2 + 1e-5) + mean[sel]
        from_features = np.asarray(posed_joints_from_features(mx.array(feats)))
        assert np.abs(from_features - golden).max() < FK_TOL, label


def test_placeholder_error_against_the_golden_clip_is_the_recorded_one():
    """Keep the "before" number reproducible: 0.132 (2 s) / 0.236 (4 s)."""
    _require_golden_npz()
    for npz_path, raw_path, expected_floor in ((NPZ2S, RAW2S, 0.13), (NPZ4S, RAW4S, 0.23)):
        if not raw_path.is_file() or not (CKPT / "stats" / "motion" / "mean.npy").is_file():
            pytest.skip("golden raw features / stats not present")
        mean = np.load(CKPT / "stats" / "motion" / "mean.npy")
        std = np.load(CKPT / "stats" / "motion" / "std.npy")
        sel = list(range(5)) + list(range(9, mean.shape[0]))
        feats = np.load(raw_path)[0] * np.sqrt(std[sel].astype(np.float32) ** 2 + 1e-5) + mean[sel]
        golden = np.load(npz_path)["posed_joints"]
        err = np.abs(np.asarray(placeholder_posed_joints(mx.array(feats))) - golden).max()
        assert err > expected_floor, (npz_path.name, err)
