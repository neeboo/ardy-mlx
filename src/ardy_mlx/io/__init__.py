"""I/O constants and feature layout for ARDY motion data.

**This module is the A/B interface for numbers.** The single source of truth is
``docs/motion-contract.md``; every constant here carries the reference
implementation location it was read from. Do not invent values here — if the
contract is missing something, raise it in ``docs/status-claude.md``.

Read from ``nv-tlabs/ardy`` @ ``693f74d``:

* ``ardy/motion_rep/reps/ardy_motionrep.py:57-64`` — feature blocks and their shapes
* ``ardy/motion_rep/reps/base.py:40-54`` — block ordering, slicing, dimension sums
* ``checkpoints/ARDY-Core-RP-20FPS-Horizon40/config.yaml`` — per-checkpoint hyperparameters
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

__all__ = [
    "BLOCK_ORDER",
    "ROOT_BLOCK_NAMES",
    "BODY_BLOCK_NAMES",
    "FOOT_CONTACT_ORDER",
    "NUM_FRAMES_PER_TOKEN",
    "FSQ_CHANNELS",
    "FSQ_LEVELS",
    "LATENT_BODY_DIM",
    "ROOT_BLOCK_DIM",
    "HYBRID_TOKEN_DIM",
    "LOCAL_ROOT_DIM",
    "NUM_BASE_STEPS",
    "CFG_TYPE",
    "DEFAULT_CFG_WEIGHT",
    "TEXT_FEAT_DIM",
    "MAX_WINDOW_SECONDS",
    "FeatureLayout",
    "PRESETS",
    "default_history_frames",
]

#: Feature block order. This *is* the concatenation order of ``frames (T, D)``.
#: ardy/motion_rep/reps/ardy_motionrep.py:57-64 (dict insertion order) and :169-181 (einops.pack).
BLOCK_ORDER = (
    "root_pos",                # [3]        metres, world root position (x, y, z)
    "global_root_heading",     # [2]        (cos theta, sin theta)
    "local_joints_positions",  # [J-1, 3]   metres, root-local, ground-aligned
    "global_rot_data",         # [J, 6]     6D continuous rotation of GLOBAL joint rotations
    "velocities",              # [J, 3]     metres/second, global
    "foot_contacts",           # [4]        binary
)

#: ``last_root_feature == "global_root_heading"`` (ardy_motionrep.py:65) => root block is the first 5 dims.
ROOT_BLOCK_NAMES = ("root_pos", "global_root_heading")

#: Everything after the root block (base.py:48).
BODY_BLOCK_NAMES = ("local_joints_positions", "global_rot_data", "velocities", "foot_contacts")

#: feet.py:23-25,52 — ``torch.cat((feet_l, feet_r))`` => left first.
FOOT_CONTACT_ORDER = ("left_heel", "left_toe", "right_heel", "right_toe")

#: config.yaml:25 — 4 frames per token ("patch size").
NUM_FRAMES_PER_TOKEN = 4

#: config.yaml:22-23 — FSQ levels; ``latent_embedding_dim = num_fsq_levels`` (fsq.py:85)
#: and an int ``fsq_level_list`` expands to ``[64] * num_fsq_levels`` (fsq.py:117-118).
FSQ_CHANNELS = 128
FSQ_LEVELS = 64

#: The body latent width == FSQ channels. Corroborated by stats/post_quantization/mean.npy shape (128,).
LATENT_BODY_DIM = FSQ_CHANNELS

#: local_root_size_dict (ardy_motionrep.py:66-70): rot_vel[1] + vel[2] + global_root_y[1].
LOCAL_ROOT_DIM = 4

#: The root block flattened over a token; 20 for both released skeletons.
ROOT_BLOCK_DIM = 5

#: ardy_model.py:479 — ``nframe_root_dim + latent_embedding_dim`` == 20 + 128.
HYBRID_TOKEN_DIM = ROOT_BLOCK_DIM * NUM_FRAMES_PER_TOKEN + LATENT_BODY_DIM

#: config.yaml:2 — diffusion schedule length (both stages share one schedule).
NUM_BASE_STEPS = 10

#: config.yaml:3 — three-pass "separated" classifier-free guidance.
CFG_TYPE = "separated"

#: scripts/generate.py:90-96 — default ``--cfg_weight`` as (text, constraint).
DEFAULT_CFG_WEIGHT = (2.0, 2.0)

#: config.yaml:49-51 ``llm_shape`` — Llama-3-8B hidden size; text features are [B, L, 4096].
TEXT_FEAT_DIM = 4096

#: scripts/generate.py:106-115 — "ARDY is trained on windows of at most 10 s".
MAX_WINDOW_SECONDS = 10


@dataclass(frozen=True)
class FeatureLayout:
    """Slice arithmetic for ``frames (T, D)`` for one skeleton.

    All values are derived, not hardcoded, so a new skeleton cannot silently
    disagree with the contract: ``motion_rep_dim`` is the sum of the block sizes.
    """

    nbjoints: int
    fps: int
    horizon: int
    skeleton: str

    @property
    def block_dims(self) -> Dict[str, int]:
        """Per-block feature count (for one frame), in ``BLOCK_ORDER``."""
        return {
            "root_pos": 3,
            "global_root_heading": 2,
            "local_joints_positions": (self.nbjoints - 1) * 3,
            "global_rot_data": self.nbjoints * 6,
            "velocities": self.nbjoints * 3,
            "foot_contacts": 4,
        }

    @property
    def slices(self) -> Dict[str, slice]:
        """Half-open slice of every block inside a ``(T, D)`` row."""
        out: Dict[str, slice] = {}
        start = 0
        for name in BLOCK_ORDER:
            stop = start + self.block_dims[name]
            out[name] = slice(start, stop)
            start = stop
        return out

    @property
    def motion_rep_dim(self) -> int:
        """``D`` of ``frames (T, D)`` (base.py:46)."""
        return sum(self.block_dims.values())

    @property
    def root_block_dim(self) -> int:
        """``global_root_dim`` (base.py:50)."""
        return sum(self.block_dims[n] for n in ROOT_BLOCK_NAMES)

    @property
    def body_dim(self) -> int:
        """``body_dim`` (base.py:49)."""
        return self.motion_rep_dim - self.root_block_dim

    @property
    def local_motion_rep_dim(self) -> int:
        """``local_root_dim + body_dim`` (base.py:53)."""
        return LOCAL_ROOT_DIM + self.body_dim

    @property
    def root_slice(self) -> slice:
        return slice(0, self.root_block_dim)

    @property
    def body_slice(self) -> slice:
        return slice(self.root_block_dim, self.motion_rep_dim)

    @property
    def nframe_root_dim(self) -> int:
        """Root block flattened over ``NUM_FRAMES_PER_TOKEN`` frames."""
        return self.root_block_dim * NUM_FRAMES_PER_TOKEN

    @property
    def dim_token(self) -> int:
        """Denoiser hybrid token width: root block flattened + body latent."""
        return self.nframe_root_dim + LATENT_BODY_DIM

    @property
    def max_window_frames(self) -> int:
        """Trained window budget, floored to a whole number of tokens (generate.py:114)."""
        return (int(MAX_WINDOW_SECONDS * self.fps) // NUM_FRAMES_PER_TOKEN) * NUM_FRAMES_PER_TOKEN

    @property
    def default_history_frames(self) -> int:
        """Batch-path ``--history_frames`` default (generate.py:115)."""
        return default_history_frames(self.fps, self.horizon, NUM_FRAMES_PER_TOKEN)


def default_history_frames(fps: float, gen_horizon_len: int, num_frames_per_token: int) -> int:
    """Longest history that still fits the model's trained window.

    Mirror of ``scripts/generate.py:106-115``. Note this is the *batch* default
    (the maximum). The interactive GUI slider defaults to the minimum instead
    (README.md:194).
    """
    max_window_len = (int(MAX_WINDOW_SECONDS * fps) // num_frames_per_token) * num_frames_per_token
    return ((max_window_len - gen_horizon_len) // num_frames_per_token) * num_frames_per_token


#: The four released checkpoints (README.md:67-70). Only the first ships locally today.
PRESETS = {
    "ARDY-Core-RP-20FPS-Horizon40": FeatureLayout(27, 20, 40, "CoreSkeleton27"),
    "ARDY-Core-RP-20FPS-Horizon8": FeatureLayout(27, 20, 8, "CoreSkeleton27"),
    "ARDY-G1-RP-25FPS-Horizon52": FeatureLayout(34, 25, 52, "G1Skeleton34"),
    "ARDY-G1-RP-25FPS-Horizon8": FeatureLayout(34, 25, 8, "G1Skeleton34"),
}

# Self-check against the contract's independently verified numbers
# (docs/motion-contract.md S1.3). D=330 for Core was confirmed three ways: block
# sum, tokenizer io dims, and the denoiser constraints projection dims.
assert PRESETS["ARDY-Core-RP-20FPS-Horizon40"].motion_rep_dim == 330
assert PRESETS["ARDY-Core-RP-20FPS-Horizon40"].body_dim == 325
assert PRESETS["ARDY-Core-RP-20FPS-Horizon40"].local_motion_rep_dim == 329
assert PRESETS["ARDY-Core-RP-20FPS-Horizon40"].dim_token == 148
assert PRESETS["ARDY-Core-RP-20FPS-Horizon40"].default_history_frames == 160
assert PRESETS["ARDY-G1-RP-25FPS-Horizon52"].motion_rep_dim == 414
assert PRESETS["ARDY-G1-RP-25FPS-Horizon52"].body_dim == 409
