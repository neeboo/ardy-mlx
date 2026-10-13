"""Core skeleton definition (``CoreSkeleton27``) and forward kinematics.

Why this file exists
--------------------
The contract's ``posed_joints`` member (``docs/motion-contract.md`` S1.5 / S7) is
"world joint positions, straight out of the reference's ``inverse()``". Producing
it needs the skeleton: the joint hierarchy and the neutral (rest) joint offsets.
Both live in the reference's *source* (Apache-2.0), not in the checkpoint:

* hierarchy: ``ardy/skeleton/definitions.py`` ``CoreSkeleton27.bone_order_names_with_parents``
* rest pose: ``ardy/assets/skeletons/cskel27/joints.p`` (``neutral_joints``)
* the kinematics themselves: ``ardy/skeleton/kinematics.py`` ``fk`` /
  ``batch_rigid_transform`` (which is the canonical SMPL-style FK)

So ``CORE_JOINT_NAMES`` / ``CORE_JOINT_PARENTS`` / ``CORE_NEUTRAL_JOINTS`` below
are transcribed from those two files and the rest of the module is a line-by-line
port of ``batch_rigid_transform`` + ``geometry.cont6d_to_matrix`` +
``transforms.global_rots_to_local_rots``. Nothing here is fitted or guessed: with
the golden run's own ``local_rot_mats`` and ``root_positions`` this reproduces the
golden ``posed_joints`` to ``1.7e-07`` (float32 storage precision), and driven from
the feature vector's ``global_rot_data`` it reproduces them to ``4.3e-07``.

Units / frame
-------------
World coordinates, right-handed, **Y up**, metres, ground = XZ plane -- identical
to :mod:`ardy_mlx.model.motion_rep` (contract S2). The root joint is ``Hips`` and
``root_positions`` is its world position, so ``posed_joints[..., 0, :]`` equals
``root_positions`` exactly.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

__all__ = [
    "CORE_JOINT_NAMES",
    "CORE_JOINT_PARENTS",
    "CORE_NEUTRAL_JOINTS",
    "CORE_ROOT_INDEX",
    "CORE_FOOT_JOINT_NAMES",
    "CORE_HIP_JOINT_NAMES",
    "Skeleton",
    "core_skeleton",
    "cont6d_to_matrix",
    "matrix_to_cont6d",
    "global_rots_to_local_rots",
    "posed_joints_from_features",
    "placeholder_posed_joints",
]

#: ``CoreSkeleton27.bone_order_names_with_parents`` (``skeleton/definitions.py:348-381``),
#: in order. Index 0 is the root.
CORE_JOINT_NAMES: Tuple[str, ...] = (
    "Hips",
    "Spine",
    "Spine1",
    "Spine2",
    "Spine3",
    "Neck",
    "Head",
    "RightShoulder",
    "RightArm",
    "RightForeArm",
    "RightHand",
    "RightHandEnd",
    "RightHandThumb1",
    "LeftShoulder",
    "LeftArm",
    "LeftForeArm",
    "LeftHand",
    "LeftHandEnd",
    "LeftHandThumb1",
    "RightUpLeg",
    "RightLeg",
    "RightFoot",
    "RightToeBase",
    "LeftUpLeg",
    "LeftLeg",
    "LeftFoot",
    "LeftToeBase",
)

#: Parent index per joint (``-1`` = root). Same source as ``CORE_JOINT_NAMES``.
CORE_JOINT_PARENTS: Tuple[int, ...] = (
    -1,   # Hips
    0,    # Spine            <- Hips
    1,    # Spine1           <- Spine
    2,    # Spine2           <- Spine1
    3,    # Spine3           <- Spine2
    4,    # Neck             <- Spine3
    5,    # Head             <- Neck
    4,    # RightShoulder    <- Spine3
    7,    # RightArm         <- RightShoulder
    8,    # RightForeArm     <- RightArm
    9,    # RightHand        <- RightForeArm
    10,   # RightHandEnd     <- RightHand
    10,   # RightHandThumb1  <- RightHand
    4,    # LeftShoulder     <- Spine3
    13,   # LeftArm          <- LeftShoulder
    14,   # LeftForeArm      <- LeftArm
    15,   # LeftHand         <- LeftForeArm
    16,   # LeftHandEnd      <- LeftHand
    16,   # LeftHandThumb1   <- LeftHand
    0,    # RightUpLeg       <- Hips
    19,   # RightLeg         <- RightUpLeg
    20,   # RightFoot        <- RightLeg
    21,   # RightToeBase     <- RightFoot
    0,    # LeftUpLeg        <- Hips
    23,   # LeftLeg          <- LeftUpLeg
    24,   # LeftFoot         <- LeftLeg
    25,   # LeftToeBase      <- LeftFoot
)

#: ``neutral_joints`` from ``ardy/assets/skeletons/cskel27/joints.p`` (27 x 3,
#: float64 in the asset, cast to float32 for compute like the reference does).
#: Full precision is kept so the transcription is exact, not rounded.
CORE_NEUTRAL_JOINTS: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),  # Hips
    (0.0, 0.070989099999999999, -0.047326099999999927),  # Spine
    (0.0, 0.16420331819510459, -0.063762289180484336),  # Spine1
    (0.0, 0.25849533092620636, -0.072011822543843848),  # Spine2
    (0.0, 0.35314752658315185, -0.072011917118271193),  # Spine3
    (0.0, 0.60160955064807009, -0.036517567004764682),  # Neck
    (0.0, 0.72977933894832137, -0.013917896912177378),  # Head
    (-0.0319949, 0.52591957193660599, -0.018687292195083791),  # RightShoulder
    (-0.19090289999999999, 0.52591951688925365, -0.018687292195028789),  # RightArm
    (-0.48633890000000002, 0.52591941454720925, -0.018687292194824189),  # RightForeArm
    (-0.7189909000000001, 0.525919333954183, -0.018687292194582476),  # RightHand
    (-0.78860240000000015, 0.5259193098400492, -0.01868729219448604),  # RightHandEnd
    (-0.74683549356950441, 0.50735627152985363, 0.027720424223916379),  # RightHandThumb1
    (0.031994900000000007, 0.52591957193660599, -0.018687292195083791),  # LeftShoulder
    (0.19090289999999999, 0.52591957193660599, -0.018687292195083791),  # LeftArm
    (0.48633890000000002, 0.52591957193660599, -0.018687292195083791),  # LeftForeArm
    (0.7189909000000001, 0.52591957193660599, -0.018687292195083791),  # LeftHand
    (0.78860240000000004, 0.52591957193660599, -0.018687292195083791),  # LeftHandEnd
    (0.74683549999999999, 0.50735651915793001, 0.027720424223376491),  # LeftHandThumb1
    (-0.094918199999999994, -0.027728900000000001, 0.0),  # RightUpLeg
    (-0.094918199999999994, -0.43984689999999999, 0.0),  # RightLeg
    (-0.094918200000000091, -0.89593789999999995, 1.0127254590486245e-16),  # RightFoot
    (-0.094918200000000105, -0.95441282523348325, 0.16065826618182677),  # RightToeBase
    (0.094918200000000022, -0.027728900000000001, 0.0),  # LeftUpLeg
    (0.094918200000000022, -0.43984690000000004, 0.0),  # LeftLeg
    (0.094918199999999925, -0.89593790000000006, 1.0127254590486246e-16),  # LeftFoot
    (0.094918199999999911, -0.95441282523348325, 0.16065826618182655),  # LeftToeBase
)

#: Root joint (``SkeletonBase.root_idx``; the only joint with parent ``-1``).
CORE_ROOT_INDEX = 0

#: ``left_foot_joint_names + right_foot_joint_names`` (``skeleton/base.py:115``), the
#: order the 4 ``foot_contacts`` channels refer to (contract S1.4).
CORE_FOOT_JOINT_NAMES: Tuple[str, ...] = ("LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase")

#: ``hip_joint_names`` -- ``[right, left]`` (``definitions.py:346``); the ordering
#: ``get_root_heading_angle`` relies on.
CORE_HIP_JOINT_NAMES: Tuple[str, ...] = ("RightUpLeg", "LeftUpLeg")


def _cross(a: mx.array, b: mx.array) -> mx.array:
    """Batched 3-vector cross product (MLX has no ``cross``)."""
    return mx.stack(
        [
            a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
            a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
            a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0],
        ],
        axis=-1,
    )


class Skeleton:
    """Hierarchy + neutral joints + FK (port of ``SkeletonBase``'s kinematics).

    ``joint_parents`` is ordered so that every parent index is smaller than its
    children's, which is what lets :meth:`fk` walk joints once, in order, instead
    of the reference's per-level ``transforms[:, indices]`` scatter. The
    multiplication chain per joint is identical either way (``G_j =
    G_parent @ T_j``), so the numbers agree; the level grouping in the reference
    is only an implementation detail of doing it with a mutable tensor.
    """

    def __init__(
        self,
        name: str,
        joint_names: Sequence[str],
        joint_parents: Sequence[int],
        neutral_joints: Sequence[Sequence[float]],
        root_index: int = 0,
        foot_joint_names: Sequence[str] = (),
        hip_joint_names: Sequence[str] = (),
    ):
        names = tuple(joint_names)
        parents = tuple(int(p) for p in joint_parents)
        neutral = np.asarray(neutral_joints, dtype=np.float64)
        if len(names) != len(parents) or neutral.shape != (len(names), 3):
            raise ValueError(
                f"skeleton {name!r}: {len(names)} names, {len(parents)} parents, "
                f"neutral {neutral.shape}"
            )
        roots = [i for i, p in enumerate(parents) if p < 0]
        if roots != [root_index]:
            raise ValueError(f"skeleton {name!r}: expected root {root_index}, got {roots}")
        for j, p in enumerate(parents):
            if j != root_index and p >= j:
                raise ValueError(
                    f"skeleton {name!r}: joint {j} ({names[j]}) has parent {p} >= {j}; "
                    "FK relies on parents coming first"
                )

        self.name = name
        self.joint_names = names
        self.joint_parents = parents
        self.joint_parents_np = np.asarray(parents, dtype=np.int64)
        self.neutral_joints = neutral.astype(np.float32)
        self.root_index = int(root_index)
        self.nbjoints = len(names)
        self.foot_joint_names = tuple(foot_joint_names)
        self.hip_joint_names = tuple(hip_joint_names)
        self.foot_joint_idx = tuple(names.index(n) for n in self.foot_joint_names)
        self.hip_joint_idx = tuple(names.index(n) for n in self.hip_joint_names)

        #: Relative joint offsets in the parent frame: ``neutral[j] - neutral[parent[j]]``
        #: (``kinematics.py:144-148``). The root keeps its own offset. The reference
        #: casts the rest pose to the compute dtype *before* subtracting, so the
        #: float32 rest pose is the one that matters.
        neutral32 = neutral.astype(np.float32) - neutral.astype(np.float32)[root_index]
        rel = neutral32.copy()
        for j in range(self.nbjoints):
            if j != root_index:
                rel[j] = neutral32[j] - neutral32[parents[j]]
        self.rel_joints_np = rel.astype(np.float32)

    # ---------------------------------------------------------------- FK
    def fk(
        self,
        local_rot_mats: mx.array,
        root_positions: mx.array,
    ) -> Tuple[mx.array, mx.array, mx.array]:
        """``batch_rigid_transform`` (``skeleton/kinematics.py:87-171``).

        ``local_rot_mats`` is ``(..., J, 3, 3)``, ``root_positions`` ``(..., 3)``.
        Returns ``(global_rot_mats, posed_joints, posed_joints_norootpos)`` with
        shapes ``(..., J, 3, 3)``, ``(..., J, 3)``, ``(..., J, 3)``.

        ``root_positions_is_global=True`` in the reference subtracts
        ``neutral_joints[root_idx]`` from the rest pose so that ``root_positions``
        is the root's *world* position; on this skeleton that offset is exactly
        zero, but the subtraction is kept for faithfulness.
        """
        rots = (
            local_rot_mats
            if isinstance(local_rot_mats, mx.array)
            else mx.array(local_rot_mats, dtype=mx.float32)
        )
        if rots.ndim < 3 or rots.shape[-1] != 3 or rots.shape[-2] != 3:
            raise ValueError(f"local_rot_mats must be (..., J, 3, 3), got {rots.shape}")
        if rots.shape[-3] != self.nbjoints:
            raise ValueError(f"expected {self.nbjoints} joints, got {rots.shape[-3]}")

        lead = rots.shape[:-3]
        t = int(np.prod(lead)) if lead else 1
        rots = rots.reshape((t, self.nbjoints, 3, 3))
        root_pos = (
            root_positions
            if isinstance(root_positions, mx.array)
            else mx.array(root_positions, dtype=mx.float32)
        )
        root_pos = root_pos.reshape((t, 3))

        rel = mx.array(self.rel_joints_np)  # neutral[j] - neutral[parent[j]], root keeps its own

        # T_j = [[R_j, rel_j], [0, 0, 0, 1]]  (kinematics.py:106-117)
        rel_col = mx.broadcast_to(rel[None, :, :, None], (t, self.nbjoints, 3, 1))
        top = mx.concatenate([rots, rel_col], axis=-1)  # (t, J, 3, 4)
        bottom = _last_row(t, self.nbjoints)  # (t, J, 1, 4)
        tm = mx.concatenate([top, bottom], axis=-2)  # (t, J, 4, 4)

        # G_root = T_root; G_j = G_parent @ T_j  (parents always come first)
        transforms: List[mx.array] = [tm[:, 0]]
        for j in range(1, self.nbjoints):
            transforms.append(mx.matmul(transforms[self.joint_parents[j]], tm[:, j]))
        glob = mx.stack(transforms, axis=1)

        posed_norootpos = glob[..., :3, 3]
        posed_joints = posed_norootpos + root_pos[:, None, :]
        global_rot_mats = glob[..., :3, :3]

        out_shape = tuple(lead) + (self.nbjoints,)
        return (
            global_rot_mats.reshape(out_shape + (3, 3)),
            posed_joints.reshape(out_shape + (3,)),
            posed_norootpos.reshape(out_shape + (3,)),
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"Skeleton({self.name!r}, nbjoints={self.nbjoints})"


def _last_row(t: int, j: int) -> mx.array:
    """The homogeneous last row ``[0, 0, 0, 1]`` of a ``(t, j, 1, 4)`` block."""
    zeros = mx.zeros((t, j, 1, 3))
    ones = mx.ones((t, j, 1, 1))
    return mx.concatenate([zeros, ones], axis=-1)


_CORE: Optional[Skeleton] = None


def core_skeleton() -> Skeleton:
    """The Core (27-joint) skeleton used by every released Core checkpoint."""
    global _CORE
    if _CORE is None:
        _CORE = Skeleton(
            "cskel27",
            CORE_JOINT_NAMES,
            CORE_JOINT_PARENTS,
            CORE_NEUTRAL_JOINTS,
            root_index=CORE_ROOT_INDEX,
            foot_joint_names=CORE_FOOT_JOINT_NAMES,
            hip_joint_names=CORE_HIP_JOINT_NAMES,
        )
    return _CORE


# --------------------------------------------------------------------------- #
# rotation-space helpers (geometry.py / skeleton/transforms.py)
# --------------------------------------------------------------------------- #
def cont6d_to_matrix(cont6d: mx.array) -> mx.array:
    """``cont6d_to_matrix`` (``geometry.py:21-37``), Gram-Schmidt on two vectors.

    ``cont6d`` is ``(..., 6)`` = two 3-vectors that are *approximately* the first
    two basis vectors of the rotation; the Gram-Schmidt pass is what makes them
    orthonormal, and it is not cosmetic: on the golden clip the raw 6D block is
    ``0.026`` away from the matrix it reconstructs (``|r0|`` off by ``0.017``,
    ``r0 . r1`` up to ``0.027``), while re-running the conversion on its own
    output is stable to ``2.4e-07``.

    The result is ``(..., 3, 3)`` with ``x``/``y``/``z`` as **columns**, which is
    the convention the FK chain and ``global_rots_to_local_rots`` expect.
    """
    if cont6d.shape[-1] != 6:
        raise ValueError(f"the last dimension must be 6, got {cont6d.shape[-1]}")
    x_raw = cont6d[..., 0:3]
    y_raw = cont6d[..., 3:6]

    x = x_raw / mx.sqrt(mx.sum(x_raw * x_raw, axis=-1, keepdims=True))
    z = _cross(x, y_raw)
    z = z / mx.sqrt(mx.sum(z * z, axis=-1, keepdims=True))
    y = _cross(z, x)

    return mx.stack([x, y, z], axis=-1)


def matrix_to_cont6d(matrix: mx.array) -> mx.array:
    """``matrix_to_cont6d`` (``geometry.py:16-18``): ``concat(M[..., 0], M[..., 1])``.

    ``M[..., 0]`` indexes the **last** axis, so this takes the first two
    *columns* of the trailing 3x3 -- contract S1.4's "first two rows" is the
    label the port inherited, the code line is what the reference does. Verified
    by use: ``cont6d_to_matrix(matrix_to_cont6d(R)) == R`` to float32 for any
    rotation ``R``, whereas reading the same 6 numbers as rows is off by ``2.0``
    on the golden clip.
    """
    return mx.concatenate([matrix[..., :, 0], matrix[..., :, 1]], axis=-1)


def global_rots_to_local_rots(global_rot_mats: mx.array, skeleton: Skeleton) -> mx.array:
    """``global_rots_to_local_rots`` (``skeleton/transforms.py:12-43``).

    ``local_j = R_parent_global^T @ R_j_global``, with the root's parent treated
    as identity. This is what the reference's ``inverse()`` uses to turn the
    stored ``global_rot_data`` back into the local rotations FK consumes
    (``ardy_motionrep.py:261``).
    """
    rots = mx.array(global_rot_mats, dtype=mx.float32) if not isinstance(
        global_rot_mats, mx.array
    ) else global_rot_mats
    if rots.shape[-3] != skeleton.nbjoints:
        raise ValueError(f"expected {skeleton.nbjoints} joints, got {rots.shape[-3]}")
    idx = mx.array(skeleton.joint_parents_np)
    parent = rots[..., idx, :, :]
    eye = mx.broadcast_to(mx.eye(3), parent.shape[:-3] + (1, 3, 3))
    parent = mx.concatenate([eye, parent[..., 1:, :, :]], axis=-3)
    return mx.matmul(mx.swapaxes(parent, -1, -2), rots)


def posed_joints_from_features(
    frames: mx.array,
    skeleton: Optional[Skeleton] = None,
) -> mx.array:
    """World joint positions from the contract feature vector, via FK.

    ``frames`` is ``(..., T, 330)`` in **physical** units (unnormalised, contract
    S1.6). The chain is exactly the reference's ``inverse()``
    (``ardy_motionrep.py:259-264``)::

        global_rot_data (6D) -> global_rot_mats -> local_rot_mats -> fk -> posed_joints

    The two blocks read out of ``frames`` are ``root_pos`` (dims ``0:3``) and
    ``global_rot_data`` (dims ``83:245``) of the Core layout (contract S1.1).
    """
    skel = skeleton or core_skeleton()
    feats = mx.array(frames, dtype=mx.float32) if not isinstance(frames, mx.array) else frames
    if feats.shape[-1] != 330:
        raise ValueError(f"expected the 330-wide Core feature vector, got {feats.shape[-1]}")
    root_positions = feats[..., 0:3]
    global_rot_data = feats[..., 83 : 83 + 27 * 6].reshape(feats.shape[:-1] + (27, 6))
    global_rot_mats = cont6d_to_matrix(global_rot_data)
    local_rot_mats = global_rots_to_local_rots(global_rot_mats, skel)
    _, posed_joints, _ = skel.fk(local_rot_mats, root_positions)
    return posed_joints


def placeholder_posed_joints(frames: mx.array) -> mx.array:
    """The **pre-FK placeholder** that ``0004`` ruling 4 recorded, for contrast.

    It puts the root at ``root_pos`` and copies the root-local
    ``local_joints_positions`` block in unchanged. Against the golden 2.0 s clip
    this is off by **0.132** (4.0 s: **0.236**) -- which is why the ruling forbade
    shipping it. Kept only so the alignment report can keep printing the "before"
    number next to the real one.
    """
    feats = mx.array(frames, dtype=mx.float32) if not isinstance(frames, mx.array) else frames
    if feats.shape[-1] != 330:
        raise ValueError(f"expected the 330-wide Core feature vector, got {feats.shape[-1]}")
    root = feats[..., 0:3][..., None, :]
    local = feats[..., 5 : 5 + 26 * 3].reshape(feats.shape[:-1] + (26, 3))
    return mx.concatenate([root, local], axis=-2)
