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
import time
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
    "POSED_JOINTS_SOURCE",
    "REFERENCE",
    "MlxRunner",
    "TextEmbedding",
    "build_mlx_runner",
    "compute_feature_layout",
    "duration_to_frames",
    "git_sha",
    "load_text_embedding",
    "mlx_parameter_count",
    "output_stem",
    "prompt_hash",
    "prompt_hash_bytes",
    "resolve_history_frames",
    "resolve_noise_windows",
    "save_outputs",
    "sha256_file",
    "sha256_text",
    "synthesize_posed_joints",
]

#: The reference implementation the contract was read from (contract S7 sidecar).
REFERENCE = {"repo": "nv-tlabs/ardy", "commit": "693f74d"}

#: ``extra.posed_joints_source`` for the real-FK path. The npz's ``posed_joints``
#: is now A's ``CoreSkeleton27`` FK over ``global_rot_data`` + ``root_pos``
#: (contract S11 N14); the old placeholder wording was "real FK not implemented".
POSED_JOINTS_SOURCE = "skeleton FK (CoreSkeleton27) on global_rot_data + root_pos"


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


def sha256_text(text: str) -> str:
    """Full SHA-256 of a prompt's UTF-8 bytes (the long form of :func:`prompt_hash`).

    The viewer accepts a prompt selector that is a full 64-hex digest — what
    ``shasum`` prints — as well as the 16-char cache name. Both are computed here, on
    the same encoding, so the two forms cannot disagree about which prompt they name.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


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


def resolve_history_frames(model: str, history_frames: Optional[int] = None) -> int:
    """Resolve ``--history-frames`` to a concrete crop length (contract S5.2).

    ``None`` means "the caller did not pass the flag" and resolves to the
    preset's batch-path default -- ``PRESETS[model].default_history_frames``,
    which is 160 for Core/20fps/H40 and 196 for G1/25fps/H52. The value is read
    from the preset, never hardcoded, so every checkpoint gets its own default.

    The distinction matters because ``None`` must **not** be forwarded to the
    model: A's batch path reads it as ``crop_history_length=None``, which
    silently disables history cropping, the ``first_heading_angle`` refresh and
    the chunked decode -- measured ``4.42e-02`` from the golden clip, 44x the
    frozen ``1e-3`` tolerance (contract S11 N16).
    """
    if model not in PRESETS:
        raise KeyError(f"unknown model {model!r}; known presets: {sorted(PRESETS)}")
    if history_frames is None:
        return int(PRESETS[model].default_history_frames)
    return int(history_frames)


def resolve_noise_windows(
    injected: np.ndarray,
    window_shape: Sequence[int],
    windows: int,
    source: str = "--noise-file",
) -> list:
    """Validate an injected ``x_T`` against this run's per-window noise shape.

    ``window_shape`` is what the model actually asks the RNG for, e.g.
    ``(1, 10, 148)`` for Core (batch, generation tokens, token dim). Accepted
    layouts:

    * ``window_shape`` exactly -- a single window, valid only when the run has
      exactly one autoregressive window;
    * ``(W,) + window_shape`` -- one entry per window, sliced by step index.

    Everything else raises, and so does a file that carries fewer windows than
    the run: **silently falling back to MLX's RNG is forbidden in this repo**.
    The old check compared ``injected.shape[1:]`` against ``shape[1:]``, so a
    stacked per-window file (``(2, 1, 10, 148)``) never matched and every window
    quietly used random noise -- the whole clip landed ``4.42`` from golden
    (contract S11 N16). A wrong noise draw produces plausible-looking motion, so
    a warning is not enough; the run must fail.

    Returns the per-window arrays in step order (a one-element list for the
    single-window form).
    """
    shape = tuple(int(v) for v in window_shape)
    held = tuple(int(v) for v in np.shape(injected))
    if windows < 1:
        raise ValueError(f"windows must be >= 1, got {windows}")

    if held == shape:
        if windows > 1:
            raise ValueError(
                f"{source} holds one window of shape {held} but this run has {windows} "
                f"windows; refusing to inject only window 0 (the rest would silently use "
                f"MLX's RNG). Pass a stacked array of shape ({windows},) + {shape}."
            )
        return [np.asarray(injected)]

    if len(held) == len(shape) + 1 and held[1:] == shape:
        if held[0] < windows:
            raise ValueError(
                f"{source} holds {held[0]} windows of shape {held} but this run has "
                f"{windows}; refusing a partial injection (the missing windows would "
                f"silently use MLX's RNG)."
            )
        if held[0] > windows:
            print(
                f"[ardy_generate] note: {source} holds {held[0]} windows; "
                f"using the first {windows}"
            )
        return [np.asarray(injected[i]) for i in range(windows)]

    raise ValueError(
        f"{source} has shape {held}, expected {shape} (one window) or "
        f"({windows},) + {shape} (one entry per autoregressive window)."
    )


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
    resolved_history = resolve_history_frames(model, history_frames)
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

    **This is the pre-FK placeholder, kept only as a diagnostic / negative
    control.** It is what the npz carried before A's skeleton FK landed, and it
    is ``0.132`` (2 s) / ``0.236`` (4 s) away from the golden ``posed_joints``.
    The CLI now exports :func:`ardy_mlx.model.skeleton.posed_joints_from_features`
    instead (contract S11 N14); A's alignment harness still calls this to show the
    comparison can fail.

    It returns only the *root-local* joint positions that are already stored in
    the feature vector, with the root joint inserted as zeros; recovering the
    contract's **world** joints needs forward kinematics over the skeleton
    offsets, which is A's side.
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
        "posed_joints_source": POSED_JOINTS_SOURCE,
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


# --------------------------------------------------------------------------- #
# the MLX batch runner -- one loader for both entry points
# --------------------------------------------------------------------------- #
def mlx_parameter_count(weight_store) -> int:
    """Number of array elements in a :class:`WeightStore` (for the viewer HUD)."""
    return int(weight_store.total_elements())


@dataclass
class MlxRunner:
    """A loaded Core checkpoint plus a cached text embedding, ready to generate.

    This is the **shape** of the one verified path to a real clip: A's
    :class:`~ardy_mlx.model.ardy.ArdyBatch` (the thing ``scripts/ardy_align.py``
    aligns) driven by the reference's own window arithmetic. Nothing here samples
    anything itself -- the window loop, the recentring, the requantisation and the
    detokenisation all stay in ``model/ardy.py``.

    It exists so the CLI (``scripts/ardy_generate.py --backend mlx``) and the
    browser viewer (``scripts/ardy_viewer.py``) load the weights through **one**
    function instead of two copies that can drift apart. The sidecar's
    ``text_embedding_sha256``, the noise shape validation and the "a shape
    mismatch must fail loudly" rule are the same in both.
    """

    model: str
    checkpoint_dir: Path
    embedding: TextEmbedding
    layout: FeatureLayout
    batch: object  # ArdyBatch; left untyped so this module does not import model/ at import time
    parameter_count: int
    load_seconds: Dict[str, float]
    device: str

    @property
    def noise_window_shape(self) -> Tuple[int, int, int]:
        """The per-window ``x_T`` shape the model asks the RNG for.

        ``(batch, generation tokens, hybrid token dim)`` -- ``(1, 10, 148)`` for
        Core/20fps/H40. Every injected-noise path is validated against this before
        anything is spent on a weight load or a diffusion step.
        """
        return (
            self.embedding.batch_size,
            self.layout.horizon // NUM_FRAMES_PER_TOKEN,
            self.layout.dim_token,
        )

    def generate(
        self,
        *,
        num_frames: int,
        num_steps: int = 10,
        cfg_weight: Sequence[float] = (2.0, 2.0),
        history_frames: Optional[int] = None,
        noise_fn=None,
        seed: Optional[int] = None,
        windows: Optional[int] = None,
    ) -> Dict[str, object]:
        """Run the batch path; returns ``frames (1, T, 330)`` and the chunk offsets.

        ``noise_fn`` is forwarded verbatim to ``ArdyBatch.generate`` (per step and
        per window); ``seed`` only picks the MLX key when ``noise_fn`` is absent.
        ``windows`` is the number of autoregressive windows the caller already
        validated its noise against -- when it is omitted the model's own
        ``ceil(num_frames / horizon)`` arithmetic is used, and a mismatch with a
        caller-supplied noise stack would show up as a loud shape error rather
        than as a quiet fallback to MLX's RNG.
        """
        if num_frames < 1:
            raise ValueError(f"num_frames must be >= 1, got {num_frames}")
        resolved_history = resolve_history_frames(self.model, history_frames)
        if windows is None:
            windows = int(math.ceil(num_frames / self.layout.horizon))

        if noise_fn is None:
            base_key = mx.random.key(int(seed) if seed is not None else 0)
            step_keys = list(mx.random.split(base_key, max(windows, 1)))

            def noise_fn(step, shape):  # noqa: ANN001 - MLX callback signature
                return mx.random.normal(shape, key=step_keys[min(step, len(step_keys) - 1)]).astype(
                    mx.float32
                )

        out = self.batch.generate(
            self.embedding.embeddings,
            self.embedding.mask,
            num_frames=num_frames,
            num_steps=num_steps,
            cfg_weight=tuple(cfg_weight),
            first_heading_angle=mx.zeros((self.embedding.batch_size,)),
            crop_history_length=resolved_history,
            noise_fn=noise_fn,
        )
        return {
            "frames": out["frames"].astype(mx.float32),
            "offsets": list(out["offsets"]),
            "history_frames": resolved_history,
            "windows": windows,
        }


def build_mlx_runner(
    model: str,
    embedding: TextEmbedding,
    checkpoint_dir: Optional[str | Path] = None,
    device: str = "gpu",
    repo_root: Optional[str | Path] = None,
) -> MlxRunner:
    """Load ``checkpoints/<model>/mlx/*.safetensors`` and assemble :class:`ArdyBatch`.

    Both entry points call this: ``scripts/ardy_generate.py --backend mlx`` and
    ``scripts/ardy_viewer.py``. It is the *only* place the tokenizer, the denoiser
    and the motion-rep stats are loaded.

    ``device`` selects the MLX default device for the whole process. The caller is
    responsible for **not** touching MLX from another thread: the MLX GPU stream is
    thread-local, and a worker thread that starts evaluating (or aborts) on the
    stream a previous thread opened is what the viewer's single-threaded HTTP
    handler exists to prevent.
    """
    from ..model.ardy import ArdyBatch
    from ..model.denoiser import AutoLatentTwostageDenoiser
    from ..model.fsq import FSQVAETransformer
    from ..model.motion_rep import MotionRep
    from ..model.nn import WeightStore

    if model not in PRESETS:
        raise KeyError(f"unknown model {model!r}; known presets: {sorted(PRESETS)}")
    root = Path(repo_root) if repo_root is not None else Path(__file__).resolve().parents[3]
    ckpt = Path(checkpoint_dir) if checkpoint_dir is not None else (root / "checkpoints" / model)
    if not ckpt.is_dir():
        raise FileNotFoundError(f"checkpoint directory not found: {ckpt}")
    tokenizer_weights = ckpt / "mlx" / "tokenizer.mlx.safetensors"
    denoiser_weights = ckpt / "mlx" / "denoiser.mlx.safetensors"
    for path in (tokenizer_weights, denoiser_weights):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing; run scripts/ardy_convert.py first (docs/status-deepseek.md)"
            )

    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    started = time.time()
    motion_rep = MotionRep(ckpt / "stats" / "motion")
    tokenizer_store = WeightStore.load(tokenizer_weights)
    tokenizer = FSQVAETransformer(tokenizer_store, motion_rep).load_stats(ckpt / "stats")
    mid = time.time()
    denoiser_store = WeightStore.load(denoiser_weights)
    denoiser = AutoLatentTwostageDenoiser(denoiser_store, motion_rep)
    batch_pipe = ArdyBatch(tokenizer, denoiser, motion_rep)
    ended = time.time()

    return MlxRunner(
        model=model,
        checkpoint_dir=ckpt,
        embedding=embedding,
        layout=PRESETS[model],
        batch=batch_pipe,
        parameter_count=mlx_parameter_count(tokenizer_store) + mlx_parameter_count(denoiser_store),
        load_seconds={
            "tokenizer_s": mid - started,
            "denoiser_s": ended - mid,
            "total_s": ended - started,
        },
        device=device,
    )
