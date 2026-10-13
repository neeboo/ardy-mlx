"""Motion representation: feature slicing, normalisation stats, root/local-root.

Ports the parts of ``ardy/motion_rep/reps/base.py`` and
``ardy/motion_rep/reps/ardy_motionrep.py`` that the batch generation path
actually touches, with the reference line numbers in the docstrings.

The one genuinely surprising thing here is the **statistics slicing**. The
shipped ``stats/motion/{mean,std}.npy`` vectors are 334 long, but the feature
vector that flows through the model is 330 long:

    stats layout : [ global_root 5 ][ local_root 4 ][ body 325 ]   = 334
    feature vec  : [ global_root 5 ][          body 325 ]          = 330

so the 330-wide normalisation uses indices ``0..4`` plus ``9..333``
(``base.py:66-72``), skipping the four local-root slots.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional

import mlx.core as mx
import numpy as np

__all__ = [
    "STATS_EPS",
    "Stats",
    "MotionRep",
    "diff_angles",
    "compute_vel_xyz",
    "core_preset",
]

#: ``Stats.__init__`` default (ardy/motion_rep/stats.py:27).
STATS_EPS = 1e-5


class Stats:
    """``(data - mean) / sqrt(std**2 + eps)`` (ardy/motion_rep/stats.py:15-82).

    The eps-adjusted denominator is precomputed once, because the reference
    does the same (``stats.py:70-75``) and the two must agree bit for bit.
    """

    def __init__(self, mean: np.ndarray, std: np.ndarray, eps: float = STATS_EPS):
        mean = np.asarray(mean, dtype=np.float32)
        std = np.asarray(std, dtype=np.float32)
        self.mean = mean
        self.std = std
        self.std_eps = np.sqrt(std.astype(np.float32) ** 2 + np.float32(eps)).astype(np.float32)
        self._mean_mx = mx.array(self.mean)
        self._std_eps_mx = mx.array(self.std_eps)

    @classmethod
    def from_folder(cls, folder: str | Path, eps: float = STATS_EPS) -> "Stats":
        folder = Path(folder)
        return cls(np.load(folder / "mean.npy"), np.load(folder / "std.npy"), eps=eps)

    @classmethod
    def sliced(cls, stats: "Stats", indices) -> "Stats":
        """``Stats.sliced`` (stats.py:36-43) -- select feature indices."""
        idx = np.asarray(indices, dtype=np.int64)
        return cls(stats.mean[idx], stats.std[idx])

    @property
    def dim(self) -> int:
        return int(self.mean.shape[0])

    def normalize(self, data: mx.array) -> mx.array:
        return (data - self._mean_mx) / self._std_eps_mx

    def unnormalize(self, data: mx.array) -> mx.array:
        return data * self._std_eps_mx + self._mean_mx


def diff_angles(angles: mx.array, fps: float) -> mx.array:
    """``diff_angles`` (ardy/motion_rep/tools.py:13-31): fps-scaled wrapped difference."""
    cos = mx.cos(angles)
    sin = mx.sin(angles)
    cos_diff = cos[..., 1:] * cos[..., :-1] + sin[..., 1:] * sin[..., :-1]
    sin_diff = sin[..., 1:] * cos[..., :-1] - cos[..., 1:] * sin[..., :-1]
    return fps * mx.arctan2(sin_diff, cos_diff)


def _repeat_diff_at_last(velocity: mx.array, lengths: Optional[mx.array]) -> mx.array:
    """``compute_vel_xyz`` / ``compute_vel_angle``'s scatter (tools.py:66-79, 100-110).

    The reference pads the finite-difference sequence with one zero **and then**
    writes ``v[lengths-1] = v[clamp(lengths-2, min=0)]`` with a gather/scatter.
    ``lengths`` is *not* the tensor's time extent in general: the denoiser passes
    ``lengths = history_len + generation_len`` while the tensor spans
    ``history_len + generation_len + future_len`` frames. On a single-window clip
    the two coincide, which is exactly why the difference is invisible until a
    second window exists (window 0 of the 4.0 s run: ``lengths = 40`` over an
    80-frame tensor).

    ``lengths is None`` means "the whole sequence is valid", i.e. the last entry
    is replaced -- which is what every call site that omits ``lengths`` gets.
    """
    b, t = velocity.shape[0], velocity.shape[1]
    if t < 2:
        return velocity
    if lengths is None:
        idx = mx.full((b,), t - 1, dtype=mx.int32)
    else:
        idx = mx.maximum(lengths.astype(mx.int32).reshape(b) - 1, 0)
    src = mx.maximum(idx - 1, 0)
    rows = mx.arange(b)
    src_vals = velocity[rows, src]  # [B, ...]
    ar = mx.arange(t)[None, :]
    hit = (ar == idx[:, None]).reshape((b, t) + (1,) * (velocity.ndim - 2))
    return mx.where(hit, src_vals[:, None], velocity)


def compute_vel_xyz(positions: mx.array, fps: float, lengths: Optional[mx.array] = None) -> mx.array:
    """``compute_vel_xyz`` (ardy/motion_rep/tools.py:34-80).

    ``v[t] = fps * (p[t+1] - p[t])``, zero-padded at the end, then
    ``v[lengths-1]`` is overwritten with ``v[lengths-2]`` (so the final valid
    frame repeats the previous velocity rather than staying zero). See
    :func:`_repeat_diff_at_last` for why ``lengths`` matters.
    """
    velocity = fps * (positions[:, 1:] - positions[:, :-1])
    b = velocity.shape[0]
    pad = mx.zeros((b, 1) + tuple(velocity.shape[2:]), dtype=velocity.dtype)
    velocity = mx.concatenate([velocity, pad], axis=1)
    return _repeat_diff_at_last(velocity, lengths)


class MotionRep:
    """Minimal port of ``ArdyMotionRep`` for the Core skeleton (J=27, 20 fps).

    Only the block layout, the statistics, and the root/local-root conversions
    used by the batch path are implemented. Block order is
    ``docs/motion-contract.md`` S1.1.
    """

    #: (name, per-frame element count) in concatenation order.
    BLOCKS = (
        ("root_pos", 3),
        ("global_root_heading", 2),
        ("local_joints_positions", (27 - 1) * 3),
        ("global_rot_data", 27 * 6),
        ("velocities", 27 * 3),
        ("foot_contacts", 4),
    )

    def __init__(self, stats_dir: str | Path, nbjoints: int = 27, fps: int = 20):
        self.nbjoints = nbjoints
        self.fps = fps
        self._from_blocks()
        all_stats = Stats.from_folder(stats_dir)
        if all_stats.dim != self.global_root_dim + self.local_root_dim + self.body_dim:
            raise ValueError(
                f"stats dim {all_stats.dim} != global_root {self.global_root_dim} + "
                f"local_root {self.local_root_dim} + body {self.body_dim}"
            )
        sel = list(range(self.global_root_dim)) + list(
            range(self.global_root_dim + self.local_root_dim, all_stats.dim)
        )
        self.full_stats = Stats.sliced(all_stats, sel)           # 330-wide
        self.global_root_stats = Stats.sliced(all_stats, list(range(self.global_root_dim)))
        self.local_root_stats = Stats.sliced(
            all_stats, list(range(self.global_root_dim, self.global_root_dim + self.local_root_dim))
        )
        self.body_stats = Stats.sliced(
            all_stats,
            list(range(self.global_root_dim + self.local_root_dim, all_stats.dim)),
        )

    def _from_blocks(self) -> None:
        names, sizes = zip(*self.BLOCKS)
        ends = np.cumsum([0] + list(sizes)).tolist()
        self.slice_dict = {n: slice(ends[i], ends[i + 1]) for i, n in enumerate(names)}
        self.motion_rep_dim = int(ends[-1])
        self.root_slice = slice(0, self.slice_dict["global_root_heading"].stop)
        self.body_slice = slice(self.root_slice.stop, self.motion_rep_dim)
        self.global_root_dim = self.root_slice.stop
        self.body_dim = self.body_slice.stop - self.body_slice.start
        self.local_root_dim = 4  # rot_vel(1) + planar vel(2) + root y(1)
        self.local_motion_rep_dim = self.local_root_dim + self.body_dim

    # ---- block accessors (base.py:243-255) ---------------------------------
    @property
    def root_block_dim(self) -> int:
        """Alias for ``global_root_dim`` -- the contract calls this the root block."""
        return self.global_root_dim

    def extract_root(self, features: mx.array) -> mx.array:
        return features[..., self.root_slice]

    def extract_body(self, features: mx.array) -> mx.array:
        return features[..., self.body_slice]

    def concat_root_body(self, root: mx.array, body: mx.array) -> mx.array:
        return mx.concatenate([root, body], axis=-1)

    def normalize(self, features: mx.array) -> mx.array:
        return self.full_stats.normalize(features)

    def unnormalize(self, features: mx.array) -> mx.array:
        return self.full_stats.unnormalize(features)

    # ---- recentring / translation (ardy_motionrep.py:112-134, ardy_model.py:29-38)
    def recenter_root_motion(
        self,
        root_motion: mx.array,
        center_frame_index: int,
        is_normalized: bool = True,
        to_normalize: bool = True,
        return_center_pos: bool = False,
    ):
        """Translate root x/z so ``center_frame_index`` becomes the local origin.

        The y component of the centre is forced to 0, so recentring never moves
        the character vertically.
        """
        if is_normalized:
            root_motion = self.global_root_stats.unnormalize(root_motion)
        center_pos = root_motion[:, center_frame_index, :3]
        center_pos = mx.concatenate(
            [center_pos[:, 0:1], mx.zeros_like(center_pos[:, 1:2]), center_pos[:, 2:3]], axis=-1
        )
        root_pos = root_motion[:, :, self.slice_dict["root_pos"]]
        shifted = mx.stack(
            [root_pos[..., 0] - center_pos[:, None, 0],
             root_pos[..., 1],
             root_pos[..., 2] - center_pos[:, None, 2]],
            axis=-1,
        )
        root_motion = mx.concatenate([shifted, root_motion[:, :, 3:]], axis=-1)
        if to_normalize:
            root_motion = self.global_root_stats.normalize(root_motion)
        if return_center_pos:
            return root_motion, center_pos
        return root_motion

    def translate_normalized_root_motion(self, root_motion: mx.array, translation: mx.array) -> mx.array:
        """``translate_normalized_root_motion`` (ardy_model.py:29-38).

        The translation's y component must be zero (asserted in the reference).
        """
        if translation.shape[-1] != 3:
            raise ValueError("translation must be a 3-vector")
        root_motion = self.global_root_stats.unnormalize(root_motion)
        root_pos = root_motion[:, :, self.slice_dict["root_pos"]] + translation[:, None, :]
        root_motion = mx.concatenate([root_pos, root_motion[:, :, 3:]], axis=-1)
        return self.global_root_stats.normalize(root_motion)

    def get_root_heading_angle(self, features: mx.array) -> mx.array:
        """``get_root_heading_angle`` (base.py:150-154)."""
        heading = features[:, :, self.slice_dict["global_root_heading"]]
        return mx.arctan2(heading[..., 1], heading[..., 0])

    # ---- root representations (base.py:115-160) ---------------------------
    def global_root_to_local_root(
        self,
        root_features: mx.array,
        normalized: bool,
        lengths: Optional[mx.array] = None,
    ) -> mx.array:
        """``global_root_to_local_root`` (base.py:115-148).

        Returns ``[B, T, 4]`` = ``(dtheta/dt, vx, vz, root_y)`` where the angles
        come from ``atan2(sin, cos)`` of the stored heading.

        **Easy to get wrong:** ``normalized`` governs *both* ends. The global
        root block is un-normalised on the way in, and the resulting 4-vector is
        re-normalised with ``local_root_stats`` on the way out
        (``base.py:145-146``)::

            if normalized:
                local_root_motion = self.local_root_stats.normalize(local_root_motion)

        So the local-root condition handed to the Stage-A decoder and to the
        Stage-B body stage is in *normalised* space, not metres and m/s.

        ``lengths`` is forwarded to the two velocity helpers, where it decides
        **which** frame receives the repeated velocity (``lengths - 1``, not the
        tensor's last frame). Omitting it means "the whole sequence is valid".
        """
        if normalized:
            root_features = self.global_root_stats.unnormalize(root_features)
        root_pos = root_features[..., :3]
        heading = root_features[..., 3:5]
        cos = heading[..., 0]
        sin = heading[..., 1]
        heading_angle = mx.arctan2(sin, cos)
        rot_vel = _pad_last(diff_angles(heading_angle, self.fps), lengths)
        # reference: compute_vel_xyz(root_pos[..., None, :], ...)[..., 0, [0, 2]]
        vel4 = compute_vel_xyz(root_pos[:, :, None, :], self.fps, lengths=lengths)
        vel3 = vel4.reshape(vel4.shape[0], vel4.shape[1], vel4.shape[3])  # [B, T, 3]
        vel = mx.stack([vel3[..., 0], vel3[..., 2]], axis=-1)  # planar (x, z)
        global_root_y = root_pos[..., 1]
        local_root = mx.concatenate([rot_vel[..., None], vel, global_root_y[..., None]], axis=-1)
        if normalized:
            local_root = self.local_root_stats.normalize(local_root)
        return local_root


def _pad_last(x: mx.array, lengths: Optional[mx.array] = None) -> mx.array:
    """Zero-pad one step at the end, then repeat ``x[lengths-2]`` at ``x[lengths-1]``.

    Mirrors the ``einops.pack`` + ``gather/scatter`` tail of ``compute_vel_angle``
    (tools.py:88-110). ``lengths=None`` means the whole sequence is valid, so the
    repeated index is the last frame -- the single-window behaviour.
    """
    pad = mx.zeros(x.shape[:-1] + (1,), dtype=x.dtype)
    padded = mx.concatenate([x, pad], axis=-1)
    return _repeat_diff_at_last(padded, lengths)


def core_preset() -> dict:
    """Core/CoreSkeleton27 constants used by the released Horizon40 checkpoint."""
    return {
        "nbjoints": 27,
        "fps": 20,
        "gen_horizon_len": 40,
        "num_frames_per_token": 4,
        "num_base_steps": 10,
        "latent_dim": 512,
        "latent_embedding_dim": 128,
        "body_dim": 325,
        "motion_rep_dim": 330,
        "local_root_dim": 4,
        "local_motion_rep_dim": 329,
        "nframe_root_dim": 20,
        "hybrid_token_dim": 148,
    }
