"""Embedding loading, output assembly and the on-disk layout (contract S7).

The contract freezes the artifacts:

* ``out/<model>/<prompt-hash>-<seed>.npz`` with members ``frames``,
  ``posed_joints``, ``root_positions``, ``foot_contacts``, ``offsets``, ``fps``,
  ``text``;
* ``out/<model>/<prompt-hash>-<seed>.json`` sidecar with ``model``, ``skeleton``,
  ``fps``, ``horizon``, ``num_frames_per_token``, ``feature_dim``,
  ``num_frames``, ``duration_s``, ``seed``, ``num_steps``, ``cfg_weight``,
  ``postprocess``, ``text_embedding_sha256``, ``git_sha``, ``reference``,
  ``created_utc``.

``<prompt-hash>`` is the **UTF-8 raw prompt** SHA-256, first 16 hex chars
(decision 0002 ruling 4); ``<seed>`` is decimal, or the literal ``noseed`` when
no seed was given.

Only the text side of ``generate.py`` is replaced here: v1 consumes a
precomputed embedding file instead of running LLM2Vec + Llama-3-8B
(README "Key cost-cutting decision").
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from ..io import (
    NUM_FRAMES_PER_TOKEN,
    PRESETS,
    TEXT_FEAT_DIM,
    FeatureLayout,
)

__all__ = [
    "REFERENCE",
    "TextEmbedding",
    "compute_feature_layout",
    "duration_to_frames",
    "git_sha",
    "load_text_embedding",
    "output_stem",
    "prompt_hash",
    "prompt_hash_bytes",
    "save_outputs",
    "sha256_file",
    "synthesize_posed_joints",
]

#: The reference implementation the contract was read from (contract S7 sidecar).
REFERENCE = {"repo": "nv-tlabs/ardy", "commit": "693f74d"}


@dataclass(frozen=True)
class TextEmbedding:
    """A precomputed ``[B, L, 4096]`` text embedding plus its mask and provenance."""

    embeddings: mx.array
    mask: mx.array
    array: str
    source_path: Optional[str]
    sha256: str

    @property
    def batch_size(self) -> int:
        return self.embeddings.shape[0]

    @property
    def num_tokens(self) -> int:
        return self.embeddings.shape[1]


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """SHA-256 of a file's bytes (contract S7 ``text_embedding_sha256``)."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def load_text_embedding(
    path: str | Path,
    array: str = "embedding",
    mask_array: Optional[str] = None,
) -> TextEmbedding:
    """Load a precomputed text embedding from ``.npy`` or ``.npz``.

    Accepted layouts:

    * ``.npy`` with shape ``[L, 4096]`` (single prompt) -> batch of 1;
    * ``.npy`` with shape ``[B, L, 4096]``;
    * ``.npz`` containing ``embedding`` (or ``--embedding-array``) with either
      shape, plus an optional boolean mask of shape ``[L]`` or ``[B, L]``.

    The contract only fixes the *tensor* contract for the text side
    (``[B, L, 4096]`` float32 + ``[B, L]`` bool mask, S3.3); the file layout of
    the precomputed embedding is new here, so both shapes are accepted and the
    choice is recorded in the sidecar.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"text embedding file not found: {path}")
    if path.suffix == ".npy":
        raw = np.load(path, allow_pickle=False)
        masks: Dict[str, np.ndarray] = {}
    elif path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as handle:
            names = list(handle.files)
            if array not in names:
                raise KeyError(
                    f"{path} has no array {array!r}; available: {sorted(names)}"
                )
            raw = handle[array]
            masks = {name: handle[name] for name in names if name != array}
    else:
        raise ValueError(f"unsupported embedding container {path.suffix!r}; use .npy or .npz")

    raw = np.asarray(raw, dtype=np.float32)
    if raw.shape[-1] != TEXT_FEAT_DIM:
        raise ValueError(
            f"text embedding last dim must be {TEXT_FEAT_DIM}, got shape {raw.shape}"
        )
    if raw.ndim == 2:
        raw = raw[None, ...]
    if raw.ndim != 3:
        raise ValueError(f"text embedding must be [L, {TEXT_FEAT_DIM}] or [B, L, {TEXT_FEAT_DIM}], got {raw.shape}")

    batch, length = raw.shape[0], raw.shape[1]
    mask: Optional[np.ndarray] = None
    if mask_array is not None:
        if mask_array not in masks:
            raise KeyError(
                f"{path} has no mask array {mask_array!r}; available: {sorted(masks)}"
            )
        mask = np.asarray(masks[mask_array])
    elif "mask" in masks:
        mask = np.asarray(masks["mask"])
    if mask is None:
        mask = np.ones((batch, length), dtype=bool)
    else:
        if mask.ndim == 1:
            mask = np.broadcast_to(mask[None, :], (batch, length))
        if mask.shape != (batch, length):
            raise ValueError(
                f"text mask shape {mask.shape} does not match embedding {(batch, length)}"
            )
        mask = mask.astype(bool)

    return TextEmbedding(
        embeddings=mx.array(raw),
        mask=mx.array(mask),
        array=array,
        source_path=str(path),
        sha256=sha256_file(path),
    )


def prompt_hash(prompt: str) -> str:
    """First 16 hex chars of ``sha256(prompt.encode("utf-8"))`` (contract S7)."""
    return prompt_hash_bytes(prompt.encode("utf-8"))


def prompt_hash_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()[:16]


def output_stem(prompt: str, seed: Optional[int]) -> str:
    """``<prompt-hash>-<seed>``; ``<seed>`` is ``noseed`` when unset (contract S7)."""
    seed_field = "noseed" if seed is None else str(int(seed))
    return f"{prompt_hash(prompt)}-{seed_field}"


def duration_to_frames(duration_s: float, fps: float) -> int:
    """``ceil(duration * fps)`` frames.

    The contract does not fix the duration->frames rounding rule (the reference
    derives the frame count from the batch layout instead), so this is recorded as
    an open interface point in ``docs/status-claude.md``.
    """
    if duration_s <= 0:
        raise ValueError(f"duration must be > 0, got {duration_s}")
    return int(math.ceil(duration_s * fps))


def compute_feature_layout(
    model: str,
    num_frames: int,
    history_frames: Optional[int] = None,
) -> Tuple[FeatureLayout, int, int]:
    """Resolve a model name to ``(layout, history_frames, num_autoregressive_steps)``.

    ``num_frames`` is the number of **generated** frames, i.e. what the reference
    ``scripts/generate.py`` calls ``--duration``-derived length and what the batch
    path produces before its ``crop_history_length`` context is considered.

    ``history_frames=None`` resolves to the **batch** default (the maximum) from
    ``io.default_history_frames`` -- contract S5.2: 160 for Core/20fps/H40. This
    is deliberately *not* the GUI slider's min default.

    The step count follows the reference's own arithmetic
    (``ardy_model.py:569-570`` with a zero-length initial history, which is what
    the batch path uses): ``ceil(num_frames / horizon)``.
    """
    if model not in PRESETS:
        raise KeyError(
            f"unknown model {model!r}; known presets: {sorted(PRESETS)}"
        )
    if num_frames < 1:
        raise ValueError(f"num_frames must be >= 1, got {num_frames}")
    layout = PRESETS[model]
    resolved_history = (
        layout.default_history_frames
        if history_frames is None
        else int(history_frames)
    )
    if resolved_history % NUM_FRAMES_PER_TOKEN != 0:
        raise ValueError(
            f"history frames must be a multiple of {NUM_FRAMES_PER_TOKEN}, got {resolved_history}"
        )
    steps = int(math.ceil(num_frames / layout.horizon))
    return layout, resolved_history, steps


def synthesize_posed_joints(
    frames: mx.array,
    layout: FeatureLayout,
) -> mx.array:
    """Recover ``posed_joints (T, J, 3)`` from ``local_joints_positions`` (contract S1.5).

    The contract puts ``posed_joints`` -- **world** joint positions -- in the npz
    "so downstream does not have to redo FK". Recovering them exactly needs
    forward kinematics over ``local_rot_mats`` plus the skeleton offsets, which
    lives on A's side (``src/ardy_mlx/model/**``) and is not available yet.

    This function therefore returns only the *root-local* joint positions that
    are already stored in the feature vector, with the root joint inserted as
    zeros, and the caller surfaces that limitation in the sidecar as
    ``posed_joints_source``. It is a **placeholder**, not the contract's
    ``posed_joints``; recorded as open in ``docs/status-claude.md``.
    """
    joints = layout.nbjoints
    local = frames[..., layout.slices["local_joints_positions"]].reshape(
        frames.shape[0], frames.shape[1], joints - 1, 3
    )
    root = frames[..., layout.slices["root_pos"]].reshape(frames.shape[0], frames.shape[1], 1, 3)
    return mx.concatenate([root, local], axis=2)


def git_sha(repo_root: Optional[str | Path] = None) -> str:
    """``git rev-parse HEAD`` for the sidecar; ``"unknown"`` when unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root) if repo_root else None,
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001 - sidecar field is best-effort
        return "unknown"


def _sidecar(
    *,
    model: str,
    layout: FeatureLayout,
    num_frames: int,
    duration_s: Optional[float],
    seed: Optional[int],
    num_steps: int,
    cfg_weight: Sequence[float],
    postprocess: bool,
    embedding_sha256: str,
    embedding_source: Optional[str],
    repo_root: Optional[str | Path],
    backend: str = "placeholder",
    extra: Optional[Dict[str, object]] = None,
) -> dict:
    payload: Dict[str, object] = {
        "model": model,
        "skeleton": layout.skeleton,
        "fps": layout.fps,
        "horizon": layout.horizon,
        "num_frames_per_token": NUM_FRAMES_PER_TOKEN,
        "feature_dim": layout.motion_rep_dim,
        "num_frames": int(num_frames),
        "duration_s": float(duration_s) if duration_s is not None else None,
        "seed": None if seed is None else int(seed),
        "num_steps": int(num_steps),
        "cfg_weight": [float(v) for v in cfg_weight],
        "postprocess": bool(postprocess),
        "text_embedding_sha256": embedding_sha256,
        "git_sha": git_sha(repo_root),
        "reference": dict(REFERENCE),
        "created_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    # Extra provenance is additive only: the contract's keys above are never
    # renamed or removed, so a downstream reader of the frozen fields is safe.
    payload["extra"] = {
        "text_embedding_source": embedding_source,
        "backend": backend,
        "posed_joints_source": "local_joints_positions (root-local); real FK not implemented yet",
    }
    if extra:
        payload["extra"].update(extra)
    return payload


def save_outputs(
    *,
    out_dir: str | Path,
    model: str,
    layout: FeatureLayout,
    frames: mx.array,
    posed_joints: mx.array,
    root_positions: mx.array,
    foot_contacts: mx.array,
    offsets: Sequence[int],
    prompt: str,
    seed: Optional[int],
    num_steps: int,
    cfg_weight: Sequence[float],
    postprocess: bool,
    duration_s: Optional[float],
    embedding_sha256: str,
    embedding_source: Optional[str] = None,
    backend: str = "placeholder",
    extra_sidecar: Optional[Dict[str, object]] = None,
) -> Tuple[Path, Path]:
    """Write the contract's ``.npz`` + ``.json`` pair; returns both paths (S7)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = output_stem(prompt, seed)
    npz_path = out_dir / f"{stem}.npz"
    json_path = out_dir / f"{stem}.json"

    np.savez(
        npz_path,
        frames=np.asarray(frames, dtype=np.float32),
        posed_joints=np.asarray(posed_joints, dtype=np.float32),
        root_positions=np.asarray(root_positions, dtype=np.float32),
        foot_contacts=np.asarray(foot_contacts).astype(bool),
        offsets=np.asarray(offsets, dtype=np.int64),
        fps=np.int32(layout.fps),
        text=np.array(prompt),
    )

    sidecar = _sidecar(
        model=model,
        layout=layout,
        num_frames=frames.shape[1],
        duration_s=duration_s,
        seed=seed,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        postprocess=postprocess,
        embedding_sha256=embedding_sha256,
        embedding_source=embedding_source,
        repo_root=None,
        backend=backend,
        extra=extra_sidecar,
    )
    json_path.write_text(json.dumps(sidecar, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    return npz_path, json_path
