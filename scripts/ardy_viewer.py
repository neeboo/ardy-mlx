#!/usr/bin/env python
"""ardy-mlx browser viewer: a single-threaded HTTP server in front of the MLX batch path.

    ./.venv/bin/python scripts/ardy_viewer.py --port 8850

Then open http://127.0.0.1:8850/ . The page picks a **cached text embedding**,
sets a seed and a duration, and asks the server to sample a clip with the real
checkpoint; it can also overlay the golden reference clip for that exact duration.

Why this file exists
--------------------
``ARDY v1`` consumes a **precomputed text embedding**; there is no ported text
encoder (README "Key cost-cutting decision"). So the prompt list on the page can
only ever be the embeddings that already exist on disk, addressed by the hash of
the prompt text. That constraint is made explicit rather than papered over:

* a prompt whose embedding is not cached fails with
  ``no_cached_embedding_for_prompt`` and the server **never** substitutes a
  different prompt's embedding (this repository's hard rule: a silent fallback
  that produces plausible-looking motion is the worst possible failure);
* a duration that is not a whole number of generation horizons fails with
  ``duration_not_horizon_multiple`` (ARDY's window arithmetic only has a defined
  answer at whole horizons);
* a requested golden reference clip that does not exist fails with
  ``no_golden_reference_for_duration`` -- again no substitution.

Single-threaded, on purpose
---------------------------
The MLX GPU stream is thread-local. A worker thread that evaluates on a stream
opened by another thread is what aborts the process (this is exactly the failure
``motionbricks-mlx`` hit). So:

* the HTTP server is a plain :class:`http.server.HTTPServer` -- one request at a
  time, in the thread that built the model;
* generation is additionally serialised behind a lock, so even a client that
  pipelines requests cannot re-enter MLX;
* it binds **127.0.0.1 only** and does **no authentication**. It is a local
  development viewer on a loopback socket; do not expose it.

Usage
-----
======================================  ==========================================
``GET  /``                              the viewer page
``GET  /<file>``                        ``viewer/`` assets (whitelisted suffixes)
``GET  /api/health``                    device, parameter count, cold start,
                                        cached embeddings, skeleton, golden refs
``POST /api/plan``                      sample one clip (JSON body)
``GET  /api/plan``                      the same, from query parameters
``GET  /api/reference``                 the golden clip for one exact duration
======================================  ==========================================

``POST /api/plan`` body::

    {"prompt": "<prompt-hash | prompt text>", "seed": 0, "duration": 2.0,
     "noise": "golden" | "random", "diffusion_steps": 10, "cfg_weight": [2.0, 2.0],
     "include_frames": false}

See ``docs/VIEWER.md`` for the full response schema.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import mimetypes
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import mlx.core as mx  # noqa: E402

from ardy_mlx.io import DEFAULT_CFG_WEIGHT, NUM_BASE_STEPS, NUM_FRAMES_PER_TOKEN, PRESETS  # noqa: E402
from ardy_mlx.model.skeleton import (  # noqa: E402
    CORE_FOOT_JOINT_NAMES,
    core_skeleton,
    posed_joints_from_features,
)
from ardy_mlx.sampling.diffusion import resolve_num_steps  # noqa: E402
from ardy_mlx.sampling.pipeline import (  # noqa: E402
    TextEmbedding,
    build_mlx_runner,
    duration_to_frames,
    load_text_embedding,
    prompt_hash,
    resolve_noise_windows,
    sha256_file,
)

DEFAULT_MODEL = "ARDY-Core-RP-20FPS-Horizon40"
DEFAULT_PORT = 8850
HOST = "127.0.0.1"

#: Where cached text embeddings live. The file name is the **prompt hash** --
#: ``prompt_hash()``, i.e. the first 16 hex chars of ``sha256(prompt.encode())``,
#: the same convention ``save_outputs`` uses for its output file names (contract
#: S7 / decision 0002 ruling 4). Using the full 64-char digest here instead would
#: give one prompt two names in one repository.
EMBEDDINGS_DIR = REPO_ROOT / "artifacts" / "embeddings"

#: The golden (reference-implementation) runs that exist, by duration. There are
#: exactly two, and that is a fact about the repository, not a configuration
#: knob: 2.0 s (one generation horizon = 40 frames) and 4.0 s (two = 80 frames).
GOLDEN_RUNS: Dict[float, Path] = {
    2.0: REPO_ROOT / "artifacts" / "golden-run",
    4.0: REPO_ROOT / "artifacts" / "golden-run-4s",
}

#: The static files the viewer is allowed to serve -- an **exact whitelist**, not
#: a suffix rule. "any ``.css`` under ``viewer/``" is a rule that serves the first
#: scratch file anyone drops in the directory; the page needs exactly these files.
STATIC_FILES = (
    "index.html",
    "app.js",
    "rig.js",
    "style.css",
    "vendor/three.module.min.js",
    "vendor/three.core.min.js",
    "vendor/THREE-LICENSE.txt",
)

VIEWER_DIR = REPO_ROOT / "viewer"

#: ``git_sha`` for the sidecar-shaped part of the payload; best effort.
REFERENCE = {"repo": "nv-tlabs/ardy", "commit": "693f74d"}


# --------------------------------------------------------------------------- #
# errors -- every failure carries a stable machine-readable name
# --------------------------------------------------------------------------- #
class ViewerError(Exception):
    """A request-level failure with a stable ``error`` name for the page."""

    def __init__(self, name: str, message: str, status: int = 400, detail: Optional[str] = None):
        super().__init__(message)
        self.name = name
        self.message = message
        self.status = status
        self.detail = detail


def _empty_embedding() -> TextEmbedding:
    """A one-token stand-in, used only to build the runner when no prompt is cached.

    ``build_mlx_runner`` builds the tokenizer and denoiser, which need the input
    width and not the text; the real embedding is swapped in per request. This is
    never used to generate: :meth:`ViewerApp._embedding_for` resolves a cached
    prompt (or raises ``no_cached_embedding_for_prompt``) before any sampling.
    """
    from ardy_mlx.io import TEXT_FEAT_DIM

    return TextEmbedding(
        embeddings=mx.zeros((1, 1, TEXT_FEAT_DIM), dtype=mx.float32),
        mask=mx.ones((1, 1), dtype=mx.bool),
        array="embedding",
        source_path=None,
        sha256="0" * 64,
    )


def _display_path(path: Path) -> str:
    """A repository-relative path when possible, the absolute path otherwise.

    The viewer normally runs from a checkout, so every path it reports is relative
    to the repository root. A test (or a ``--embeddings-dir`` outside the tree)
    must not crash path formatting, and a wrong-but-pretty path would be worse than
    an absolute one.
    """
    try:
        return str(Path(path).relative_to(REPO_ROOT))
    except ValueError:
        return str(Path(path))


# --------------------------------------------------------------------------- #
# cached embeddings
# --------------------------------------------------------------------------- #
class EmbeddingCache:
    """The prompts this server can actually run, keyed by prompt hash.

    A prompt is available **only** because a file named after its hash exists.
    Nothing here can encode text, and nothing here guesses: a lookup miss is a
    :class:`ViewerError` (``no_cached_embedding_for_prompt``), not a fallback.
    """

    def __init__(self, directory: Path = EMBEDDINGS_DIR):
        self.directory = Path(directory)
        self._entries: Dict[str, dict] = {}

    # -- discovery -------------------------------------------------------
    def _sidecar(self, npy_path: Path) -> Optional[dict]:
        sidecar = npy_path.with_suffix(".json")
        if not sidecar.is_file():
            return None
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def refresh(self) -> List[dict]:
        """Rescan the directory; returns the sorted entry list."""
        entries: Dict[str, dict] = {}
        if self.directory.is_dir():
            for path in sorted(self.directory.glob("*.npy")):
                stem = path.stem
                if not re.fullmatch(r"[0-9a-f]{16}", stem):
                    # Not one of ours (a scratch file, a README placeholder). It
                    # is reported as absent rather than guessed at: an embedding
                    # whose name is not a prompt hash cannot be addressed.
                    continue
                sidecar = self._sidecar(path)
                text = None
                if sidecar and isinstance(sidecar.get("prompt"), str):
                    text = sidecar["prompt"]
                try:
                    array = np.load(path, allow_pickle=False)
                    tokens = int(array.shape[1]) if array.ndim == 3 else int(array.shape[0])
                    shape = list(array.shape)
                except (OSError, ValueError) as exc:  # unreadable -> listed, but not usable
                    tokens, shape = None, None
                    text = text or f"<unreadable: {exc}>"
                entries[stem] = {
                    "prompt_hash": stem,
                    "text": text if text is not None else f"<unknown prompt for {stem}>",
                    "has_prompt_text": text is not None,
                    "path": _display_path(path),
                    "embedding_sha256": sha256_file(path),
                    "shape": shape,
                    "tokens": tokens,
                    "source": (sidecar or {}).get("source"),
                    "note": (sidecar or {}).get("note"),
                }
        self._entries = entries
        return self.list()

    def list(self) -> List[dict]:
        return [self._entries[key] for key in sorted(self._entries)]

    # -- lookup ----------------------------------------------------------
    def resolve(self, selector: str) -> dict:
        """Resolve a prompt hash **or** an exact prompt text to a cached entry.

        Text is resolved by hashing it, so the text path and the hash path agree
        by construction. If a selector is neither a cached hash nor the exact text
        of a cached embedding, this raises -- it never picks "the closest" one.
        """
        if selector is None or selector == "":
            raise ViewerError(
                "no_prompt_given",
                "a prompt is required: pass the prompt hash of a cached embedding, or its exact text",
            )
        if selector in self._entries:
            return self._entries[selector]

        lower = selector.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", lower):
            # A full SHA-256 of the prompt text (what `shasum` prints). Resolved by
            # hashing the *stored* texts, so an arbitrary 64-hex string can never be
            # mistaken for a prompt it does not actually describe.
            for entry in self._entries.values():
                if hashlib.sha256(entry["text"].encode("utf-8")).hexdigest() == lower:
                    return entry

        digest = prompt_hash(selector)
        if digest in self._entries:
            return self._entries[digest]

        available = ", ".join(sorted(self._entries)) or "(none)"
        try:
            where = self.directory.relative_to(REPO_ROOT)
        except ValueError:
            where = self.directory
        raise ViewerError(
            "no_cached_embedding_for_prompt",
            f"no cached embedding for this prompt; expected {where}/{digest}.npy",
            detail=(
                f"selector={selector[:120]!r} resolves to prompt hash {digest}. "
                f"Available prompt hashes: {available}. v1 has no text encoder, so an "
                f"uncached prompt cannot be run -- see embeddings/README.md."
            ),
        )


# --------------------------------------------------------------------------- #
# golden reference runs
# --------------------------------------------------------------------------- #
def golden_npz(duration_s: float) -> Path:
    if not duration_s:
        raise ViewerError("no_duration_given", "duration is required")
    path = GOLDEN_RUNS.get(float(duration_s))
    if path is None:
        known = ", ".join(f"{d:.1f}s" for d in sorted(GOLDEN_RUNS)) or "(none)"
        raise ViewerError(
            "no_golden_reference_for_duration",
            f"no golden reference clip for {float(duration_s):.2f} s; the reference runs on disk are: {known}",
            status=404,
            detail=(
                "Golden clips come from the two authorised reference runs (artifacts/golden-run, "
                "artifacts/golden-run-4s). The viewer will not show a different duration's clip."
            ),
        )
    out_dir = path / "out"
    if not out_dir.is_dir():
        raise ViewerError(
            "golden_reference_missing",
            f"golden run directory {path} exists but has no out/ directory",
            status=404,
        )
    candidates = sorted(out_dir.glob("*.npz"))
    if not candidates:
        raise ViewerError(
            "golden_reference_missing",
            f"no golden .npz under {out_dir}",
            status=404,
            detail="The reference npz is not committed; it must be present locally.",
        )
    return candidates[0]


def load_golden(duration_s: float) -> dict:
    """Read one golden npz and normalise it into the viewer's clip payload."""
    path = golden_npz(duration_s)
    with np.load(path, allow_pickle=False) as data:
        joints = np.asarray(data["posed_joints"], dtype=np.float32)
        roots = np.asarray(data["root_positions"], dtype=np.float32)
        contacts = np.asarray(data["foot_contacts"]).astype(np.uint8)
        fps = int(np.asarray(data["fps"])) if "fps" in data.files else PRESETS[DEFAULT_MODEL].fps
        text = str(np.asarray(data["text"])) if "text" in data.files else ""
    return {
        "kind": "golden",
        "posed_joints_b64": encode_float32(joints),
        "root_positions_b64": encode_float32(roots),
        "foot_contacts_b64": encode_bytes(contacts.reshape(-1)),
        "frames": int(joints.shape[0]),
        "joints": int(joints.shape[1]),
        "fps": fps,
        "duration_s": float(duration_s),
        "model": DEFAULT_MODEL,
        "prompt": text,
        "npz_path": _display_path(path),
        "npz_sha256": sha256_file(path),
        "source": {
            "repo": REFERENCE["repo"],
            "commit": REFERENCE["commit"],
            "postprocess": False,
            "note": (
                "the reference implementation's own output for this duration "
                "(one authorised run, --no-postprocess); it is not an MLX product"
            ),
        },
    }


# --------------------------------------------------------------------------- #
# golden noise
# --------------------------------------------------------------------------- #
def golden_noise_files(duration_s: float) -> List[Path]:
    """The reference ``x_T`` files recorded for this duration, one per window.

    2.0 s has no ``randn_*.npy`` because its single window's ``x_T`` *is*
    ``denoise_step0_x_in.npy`` (the first denoise input is the injected noise);
    4.0 s recorded ``randn_0`` / ``randn_1``. Nothing is synthesised: if a
    duration has no recorded ``x_T``, "golden" noise is refused by name.
    """
    path = GOLDEN_RUNS.get(float(duration_s))
    if path is None:
        return []
    raw = path / "raw"
    if not raw.is_dir():
        return []
    randn = sorted(raw.glob("randn_*.npy"), key=lambda p: int(p.stem.split("_")[-1]))
    if randn:
        return randn
    step0 = raw / "denoise_step0_x_in.npy"
    return [step0] if step0.is_file() else []


def build_noise_fn(noise: str, duration_s: float, runner, seed: Optional[int]):
    """Return ``(noise_fn, sidecar)`` for the requested noise mode.

    ``golden`` injects the reference's own ``x_T`` per window, validated against
    the runner's per-window shape with the repository's shared
    :func:`resolve_noise_windows` (the same function ``--noise-file`` goes
    through). A mismatch raises instead of quietly using MLX's RNG: a wrong draw
    produces plausible motion and is therefore the one error that must never be
    silent (contract S11 N16).
    """
    if noise == "random":
        return None, {
            "mode": "random",
            "seed": None if seed is None else int(seed),
            "source_files": [],
            "note": (
                "MLX's own RNG. CUDA's Philox stream is not reproducible in MLX, so a random "
                "clip is a valid sample but is not a reference reproduction."
            ),
        }

    if noise != "golden":
        raise ViewerError(
            "unknown_noise_mode",
            f"noise must be 'golden' or 'random', got {noise!r}",
        )

    files = golden_noise_files(duration_s)
    if not files:
        raise ViewerError(
            "no_golden_noise_for_duration",
            f"no recorded reference x_T for {float(duration_s):.2f} s",
            detail=(
                "golden noise is read from artifacts/golden-run*/raw (randn_*.npy, or "
                "denoise_step0_x_in.npy for a single-window run). Nothing is synthesised."
            ),
        )

    shape = runner.noise_window_shape
    # One recorded file per autoregressive window; each is validated on its own so
    # a truncated or mis-shaped dump fails where it is read, not mid-generation.
    per_window: List[np.ndarray] = []
    for path in files:
        stack = np.load(path, allow_pickle=False)
        per_window.extend(resolve_noise_windows(stack, shape, 1, source=str(path)))
    if not per_window:
        raise ViewerError(
            "no_golden_noise_for_duration",
            f"the recorded x_T files for {float(duration_s):.2f} s carry no window",
            status=500,
        )

    def noise_fn(step, requested_shape):
        if tuple(requested_shape) != shape:
            raise ViewerError(
                "noise_shape_mismatch",
                f"model asked for noise shape {tuple(requested_shape)} but this run validated {shape}",
                status=500,
            )
        return mx.array(per_window[min(step, len(per_window) - 1)])

    return noise_fn, {
        "mode": "golden",
        "seed": None if seed is None else int(seed),
        "windows": len(per_window),
        "source_files": [_display_path(path) for path in files],
        "note": (
            "the reference implementation's own x_T is injected, so this clip reproduces the "
            "reference up to float32 reduction order (measured 1.9e-05 for 2.0 s)"
        ),
    }


# --------------------------------------------------------------------------- #
# encoding helpers
# --------------------------------------------------------------------------- #
def encode_float32(array: np.ndarray) -> str:
    """Little-endian float32 bytes, base64. Bit-exact; no decimal round trip."""
    return base64.b64encode(np.ascontiguousarray(array, dtype="<f4").tobytes()).decode("ascii")


def encode_bytes(array: np.ndarray) -> str:
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode("ascii")


def duration_to_windows(duration_s: float, layout) -> int:
    """Autoregressive windows this duration needs: ``ceil(frames / horizon)``."""
    frames = duration_to_frames(duration_s, layout.fps)
    return max(int(np.ceil(frames / layout.horizon)), 1)


def validate_duration(duration_s: float, layout) -> int:
    """``duration_s`` must be a whole number of horizons; returns the frame count.

    ARDY's batch path only has a defined answer at whole horizons: with
    ``crop_history_length`` fixed, a partial final window is either padded or
    dropped, and the two choices give different clips from the same request. There
    is no honest way to pick one for the user, so this is an error -- named, with
    the nearest legal values in the message.

    The check runs on **frames**, not on seconds: a horizon of 40 frames at 20 fps
    is 2.0 s, and ``duration * fps`` reintroduces float error for values like
    ``0.1`` that ``ceil`` would then turn into the wrong frame count.
    """
    if duration_s is None or not np.isfinite(duration_s) or duration_s <= 0:
        raise ViewerError(
            "invalid_duration",
            f"duration must be a positive number of seconds, got {duration_s!r}",
        )
    step_s = layout.horizon / layout.fps
    frames = duration_to_frames(duration_s, layout.fps)
    if frames % layout.horizon != 0:
        lower = max(1, int(frames // layout.horizon)) * step_s
        raise ViewerError(
            "duration_not_horizon_multiple",
            (
                f"duration {duration_s:g} s is not a whole number of {layout.horizon}-frame "
                f"horizons ({step_s:g} s at {layout.fps} fps); nearest legal values are "
                f"{lower:g} s and {lower + step_s:g} s"
            ),
            detail=(
                f"{duration_s:g} s is {frames} frames, and {layout.horizon} does not divide "
                f"{frames}. Refusing to round the clip for you: a partial final window is "
                f"either padded or dropped, and those are different clips."
            ),
        )
    return frames


# --------------------------------------------------------------------------- #
# the application
# --------------------------------------------------------------------------- #
class ViewerApp:
    """Owns the model, the embedding cache and the golden runs.

    Constructed once, used from one thread. :meth:`plan` takes a lock so that a
    second request can never re-enter MLX while the first is mid-generation.
    """

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        device: str = "gpu",
        embeddings_dir: Path = EMBEDDINGS_DIR,
        checkpoint_dir: Optional[Path] = None,
        allow_cpu: bool = False,
    ):
        self.model = model
        self.device = device
        self.layout = PRESETS[model]
        self.allow_cpu = allow_cpu
        self.embeddings = EmbeddingCache(embeddings_dir)
        self.checkpoint_dir = checkpoint_dir
        self.lock = threading.Lock()
        self.runner = None
        self.error: Optional[str] = None
        self.boot_seconds: Optional[float] = None
        self.embedding_cache: Dict[str, object] = {}

    # -- startup ---------------------------------------------------------
    def start(self) -> None:
        """Load the checkpoint once. A failure is recorded, not raised.

        The server still answers ``/api/health`` (with the error in it) and still
        serves the page: a viewer that cannot say *why* it has no model is worse
        than one that says it plainly.

        The weight load is deliberately independent of the embedding cache: the
        runner is built with whichever cached embedding happens to exist (or an
        empty stand-in), and :meth:`plan` swaps the requested prompt's embedding in
        per request. A machine with the checkpoint but no cached prompt therefore
        answers health honestly -- ``model`` loaded, ``embeddings`` empty, and
        ``no_cached_embedding_for_prompt`` on every plan -- instead of claiming the
        model is broken.
        """
        if self.device == "cpu" and not self.allow_cpu:
            self.error = (
                "the MLX viewer only supports the GPU device in v1; pass --allow-cpu to "
                "override (the golden comparison numbers are GPU)"
            )
            return
        started = time.time()
        try:
            entries = self.embeddings.refresh()
            embedding = self._embedding_for(entries[0]["prompt_hash"]) if entries else _empty_embedding()
            self.runner = build_mlx_runner(
                self.model,
                embedding,
                checkpoint_dir=self.checkpoint_dir,
                device=self.device,
                repo_root=REPO_ROOT,
            )
            # One evaluation so the Metal pipeline is actually built before the
            # first request is timed; a "cold start" that excludes the first
            # kernel compile is a number that flatters the server.
            mx.eval(mx.zeros((1,)))
        except Exception as exc:  # noqa: BLE001 - the health endpoint reports it
            self.error = f"{type(exc).__name__}: {exc}"
            self.runner = None
        finally:
            self.boot_seconds = time.time() - started

    # -- embeddings ------------------------------------------------------
    def _embedding_for(self, prompt_hash: str):
        entry = self.embeddings.resolve(prompt_hash)
        cached = self.embedding_cache.get(entry["prompt_hash"])
        if cached is not None:
            return cached
        embedding = load_text_embedding(entry["path"])
        self.embedding_cache[entry["prompt_hash"]] = embedding
        return embedding

    # -- endpoints -------------------------------------------------------
    def health(self) -> dict:
        self.embeddings.refresh()
        entries = self.embeddings.list()
        skeleton = core_skeleton()
        step_s = self.layout.horizon / self.layout.fps
        return {
            "status": "ok" if self.runner is not None else "unavailable",
            "error": self.error,
            "model": self.model,
            "device": {
                "name": str(mx.default_device()),
                "backend": self.device,
                "platform": sys.platform,
                "mlx_version": getattr(mx, "__version__", "unknown"),
            },
            "parameters": (self.runner.parameter_count if self.runner is not None else None),
            "cold_start_s": self.boot_seconds,
            "load_seconds": (self.runner.load_seconds if self.runner is not None else None),
            "fps": self.layout.fps,
            "horizon": self.layout.horizon,
            "num_frames_per_token": NUM_FRAMES_PER_TOKEN,
            "motion_rep_dim": self.layout.motion_rep_dim,
            "duration_step_s": step_s,
            "duration_min_s": step_s,
            "num_base_steps": NUM_BASE_STEPS,
            "default_cfg_weight": list(DEFAULT_CFG_WEIGHT),
            "skeleton": {
                "name": skeleton.name,
                "jointNames": list(skeleton.joint_names),
                "jointParents": list(skeleton.joint_parents),
                "rootIndex": skeleton.root_index,
                "neutralJoints": skeleton.neutral_joints.tolist(),
                "footJointNames": list(CORE_FOOT_JOINT_NAMES),
                "footJointIdx": list(skeleton.foot_joint_idx),
            },
            "embeddings": entries,
            "embeddings_dir": _display_path(self.embeddings.directory),
            "golden_references": [
                {
                    "duration_s": duration,
                    "path": _display_path(path),
                    "available": (path / "out").is_dir() and any((path / "out").glob("*.npz")),
                }
                for duration, path in sorted(GOLDEN_RUNS.items())
            ],
            "limits": {
                "text_encoder": "not ported in v1: prompts must be pre-cached embeddings",
                "postprocess": "MotionCorrection (C++ IK/trajectory) is not built; foot skating remains",
                "rng": "MLX's RNG is not CUDA's Philox stream; only golden noise reproduces the reference",
                "auth": "none: loopback-only development viewer",
                "threading": "one request at a time (the MLX GPU stream is thread-local)",
            },
            "served_by": f"scripts/ardy_viewer.py ({HOST})",
        }

    def plan(
        self,
        prompt: str,
        seed: Optional[int] = None,
        duration: Optional[float] = None,
        noise: str = "golden",
        diffusion_steps: Optional[int] = None,
        cfg_weight=None,
        include_frames: bool = False,
    ) -> dict:
        if self.runner is None:
            raise ViewerError(
                "model_unavailable",
                self.error or "the MLX checkpoint is not loaded",
                status=503,
            )
        entry = self.embeddings.resolve(prompt)
        embedding = self._embedding_for(entry["prompt_hash"])
        frames_wanted = validate_duration(duration, self.layout)
        num_steps = resolve_num_steps(diffusion_steps, NUM_BASE_STEPS)
        weights = self._cfg_weight(cfg_weight)
        windows = duration_to_windows(duration, self.layout)

        noise_fn, noise_sidecar = build_noise_fn(noise, duration, self.runner, seed)

        runner = self.runner
        original = runner.embedding
        try:
            # The runner caches one embedding; swap in this prompt's. Cheap (an
            # mx.array of [1, L, 4096]) and it keeps the MLX call in one place.
            runner.embedding = embedding
            with self.lock:
                started = time.perf_counter()
                out = runner.generate(
                    num_frames=frames_wanted,
                    num_steps=num_steps,
                    cfg_weight=weights,
                    history_frames=None,
                    noise_fn=noise_fn,
                    seed=seed,
                    windows=windows,
                )
                generated = out["frames"]
                mx.eval(generated)
                generate_ms = (time.perf_counter() - started) * 1000.0
        finally:
            runner.embedding = original

        posed = posed_joints_from_features(generated)
        mx.eval(posed)
        joints = np.asarray(posed, dtype=np.float32)[0]
        roots = np.asarray(generated[..., 0:3], dtype=np.float32)[0]
        contacts = (np.asarray(generated[..., -4:], dtype=np.float32) > 0.5)[0]

        payload = {
            "kind": "plan",
            "model": self.model,
            "frames": int(joints.shape[0]),
            "joints": int(joints.shape[1]),
            "fps": self.layout.fps,
            "duration_s": float(frames_wanted / self.layout.fps),
            "duration_requested_s": float(duration),
            "windows": int(windows),
            "diffusion_steps": int(num_steps),
            "cfg_weight": list(weights),
            "seed": None if seed is None else int(seed),
            "generate_ms": generate_ms,
            "posed_joints_b64": encode_float32(joints),
            "root_positions_b64": encode_float32(roots),
            "foot_contacts_b64": encode_bytes(contacts.reshape(-1).astype(np.uint8)),
            "foot_contact_order": ["left_heel", "left_toe", "right_heel", "right_toe"],
            "foot_joint_names": list(CORE_FOOT_JOINT_NAMES),
            "embedding": {
                "text": entry["text"],
                "prompt_hash": entry["prompt_hash"],
                "embedding_sha256": entry["embedding_sha256"],
                "embedding_path": entry["path"],
                "has_prompt_text": entry["has_prompt_text"],
            },
            "noise": noise_sidecar,
            "num_autoregressive_steps": max(len(out["offsets"]) - 1, 0),
            "offsets": [int(v) for v in out["offsets"]],
            "history_frames": int(out["history_frames"]),
            "postprocess": False,
            "reference": REFERENCE,
            "served_by": f"{HOST} scripts/ardy_viewer.py",
        }
        if include_frames:
            payload["frames_330_b64"] = encode_float32(np.asarray(generated, dtype=np.float32)[0])
        return payload

    @staticmethod
    def _cfg_weight(values) -> Tuple[float, float]:
        if values is None:
            return tuple(DEFAULT_CFG_WEIGHT)
        if isinstance(values, (int, float)):
            values = [values]
        if len(values) == 1:
            return (float(values[0]), 0.0)
        if len(values) == 2:
            return (float(values[0]), float(values[1]))
        raise ViewerError(
            "invalid_cfg_weight",
            f"cfg_weight takes 1 or 2 values (text, constraint), got {len(values)}",
        )

    def reference(self, duration: Optional[float], prompt: Optional[str] = None) -> dict:
        if duration is None:
            raise ViewerError("no_duration_given", "duration is required")
        payload = load_golden(float(duration))
        if prompt:
            entry = self.embeddings.resolve(prompt)
            if not payload["prompt"] or payload["prompt"] != entry["text"]:
                raise ViewerError(
                    "golden_reference_prompt_mismatch",
                    (
                        f"the golden clip for {float(duration):.2f} s was generated from "
                        f"{payload['prompt']!r}, not {entry['text']!r}"
                    ),
                    status=404,
                    detail="Comparing two different prompts would not be a comparison.",
                )
        return payload


# --------------------------------------------------------------------------- #
# HTTP
# --------------------------------------------------------------------------- #
class ViewerHandler(BaseHTTPRequestHandler):
    server_version = "ardy-mlx-viewer/1.0"
    protocol_version = "HTTP/1.0"

    # The URL space is a whitelist: `/` plus the five viewer files plus the three
    # API routes. Nothing else is reachable, so the static handler cannot be
    # walked out of `viewer/`.
    ROUTES = ("/api/health", "/api/plan", "/api/reference")

    def log_message(self, fmt, *args):  # noqa: A003 - stdlib signature
        sys.stderr.write("[ardy_viewer] %s - %s\n" % (self.address_string(), fmt % args))

    # -- plumbing --------------------------------------------------------
    def _send_json(self, payload: dict, status: int = 200) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        # A HEAD response carries the headers and no body (RFC 9110 §9.3.2).
        if self.command != "HEAD":
            self.wfile.write(body)

    def _send_error_json(self, error: ViewerError) -> None:
        self._send_json(
            {"error": error.name, "message": error.message, "detail": error.detail},
            status=error.status,
        )

    def _read_json_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        if length > 1 << 20:
            raise ViewerError("body_too_large", "request body over 1 MiB", status=413)
        raw = self.rfile.read(length)
        try:
            body = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ViewerError("invalid_json", f"could not parse the request body: {exc}") from None
        if not isinstance(body, dict):
            raise ViewerError("invalid_json", "the request body must be a JSON object")
        return body

    # -- verbs -----------------------------------------------------------
    # Only GET and POST exist. The others are answered with a named 405 rather
    # than the stdlib's bare 501: "this viewer never writes anything" should be
    # visible in the response, not inferred from a missing method.
    def do_GET(self):  # noqa: N802 - stdlib signature
        self._dispatch("GET")

    def do_POST(self):  # noqa: N802 - stdlib signature
        self._dispatch("POST")

    def do_PUT(self):  # noqa: N802
        self._dispatch("PUT")

    def do_PATCH(self):  # noqa: N802
        self._dispatch("PATCH")

    def do_DELETE(self):  # noqa: N802
        self._dispatch("DELETE")

    def do_HEAD(self):  # noqa: N802
        self._dispatch("HEAD")

    def do_OPTIONS(self):  # noqa: N802
        self._dispatch("OPTIONS")

    def _dispatch(self, method: str) -> None:
        parsed = urlparse(self.path)
        query = parse_qs(parsed.query)
        app: ViewerApp = self.server.app  # type: ignore[attr-defined]
        try:
            if parsed.path in self.ROUTES:
                if parsed.path == "/api/health":
                    if method != "GET":
                        raise ViewerError("method_not_allowed", "use GET /api/health", status=405)
                    self._send_json(app.health())
                elif parsed.path == "/api/plan":
                    if method not in ("GET", "POST"):
                        raise ViewerError(
                            "method_not_allowed",
                            f"{method} is not allowed on /api/plan; use GET or POST",
                            status=405,
                        )
                    params = self._read_json_body() if method == "POST" else {}
                    params = {**{k: v[-1] for k, v in query.items()}, **params}
                    self._send_json(app.plan(
                        prompt=params.get("prompt"),
                        seed=_opt_int(params.get("seed"), "seed"),
                        duration=_opt_float(params.get("duration"), "duration"),
                        noise=str(params.get("noise", "golden")),
                        diffusion_steps=_opt_int(params.get("diffusion_steps"), "diffusion_steps"),
                        cfg_weight=_opt_cfg(params.get("cfg_weight")),
                        include_frames=_opt_bool(params.get("include_frames")),
                    ))
                else:  # /api/reference
                    if method != "GET":
                        raise ViewerError("method_not_allowed", "use GET /api/reference", status=405)
                    self._send_json(app.reference(
                        duration=_opt_float(query.get("duration", [None])[-1], "duration"),
                        prompt=query.get("prompt", [None])[-1],
                    ))
                return
            if method != "GET":
                raise ViewerError("method_not_allowed", f"{method} is not allowed here", status=405)
            self._serve_static(parsed.path)
        except ViewerError as exc:
            self._send_error_json(exc)
        except Exception as exc:  # noqa: BLE001 - a named 500 beats a silent reset
            self._send_json(
                {
                    "error": "internal_error",
                    "message": f"{type(exc).__name__}: {exc}",
                    "detail": traceback.format_exc(limit=6),
                },
                status=500,
            )

    def _serve_static(self, path: str) -> None:
        name = "index.html" if path in ("/", "") else path.lstrip("/")
        # Membership in an exact list, before any path arithmetic happens. A
        # traversal probe is therefore rejected for being an unknown *name*, not
        # for resolving somewhere odd -- the check cannot be defeated by an
        # encoding, a symlink or a suffix.
        if name not in STATIC_FILES:
            raise ViewerError("not_found", f"no such path: {path}", status=404)
        target = (VIEWER_DIR / name).resolve()
        # Defence in depth: even a whitelisted name must resolve inside the
        # viewer directory (a symlinked `app.js` would otherwise escape it).
        if not target.is_relative_to(VIEWER_DIR.resolve()) or not target.is_file():
            raise ViewerError("not_found", f"no such file: {name}", status=404)
        body = target.read_bytes()
        content_type = mimetypes.guess_type(str(target))[0] or "application/octet-stream"
        if content_type.startswith("text/") or target.suffix in (".js", ".css", ".map"):
            content_type += "; charset=utf-8"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def _opt_int(value, field: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ViewerError("invalid_parameter", f"{field} must be an integer, got {value!r}") from None


def _opt_float(value, field: str) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ViewerError("invalid_parameter", f"{field} must be a number, got {value!r}") from None


def _opt_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in ("1", "true", "yes", "on")


def _opt_cfg(value):
    if value is None or value == "":
        return None
    if isinstance(value, list):
        return value
    return [float(part) for part in str(value).split(",")]


class ViewerServer(HTTPServer):
    """A single-threaded HTTP server with the app attached.

    ``HTTPServer`` (not ``ThreadingHTTPServer``) is the whole point: MLX's GPU
    stream is thread-local, and serving a request in a worker thread while another
    thread holds the stream is what aborts the process.
    """

    allow_reuse_address = True

    def __init__(self, address, app: ViewerApp):
        self.app = app
        super().__init__(address, ViewerHandler)


# --------------------------------------------------------------------------- #
# entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ardy_viewer.py",
        description="Serve the ardy-mlx browser viewer (loopback only, no authentication).",
    )
    parser.add_argument("--host", default=HOST, help=f"bind address (default {HOST})")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port (default {DEFAULT_PORT})")
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        choices=sorted(PRESETS),
        help=f"checkpoint preset (default {DEFAULT_MODEL})",
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default="gpu")
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="permit --device cpu (v1 is verified on the GPU; CPU is for smoke tests)",
    )
    parser.add_argument(
        "--embeddings-dir",
        default=str(EMBEDDINGS_DIR),
        help="directory of cached embeddings, named <prompt-hash>.npy",
    )
    parser.add_argument("--checkpoint-dir", default=None, help="checkpoint root override")
    parser.add_argument(
        "--no-preload",
        action="store_true",
        help="start the HTTP server before loading weights (the page sees model_unavailable)",
    )
    parser.add_argument(
        "--exit-after",
        type=float,
        default=None,
        metavar="SECONDS",
        help=(
            "shut the server down this many seconds after it starts listening. Off by "
            "default; used by the tests, so a spawned server cannot outlive its test run."
        ),
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.host not in ("127.0.0.1", "localhost", "::1"):
        print(
            f"error: refusing to bind {args.host}: this viewer has no authentication and no TLS. "
            "It is a loopback development tool.",
            file=sys.stderr,
        )
        return 2

    app = ViewerApp(
        model=args.model,
        device=args.device,
        embeddings_dir=Path(args.embeddings_dir),
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        allow_cpu=args.allow_cpu,
    )

    # Bind before loading so a port clash is reported immediately instead of after
    # a 765 MB weight load.
    server = ViewerServer((args.host, args.port), app)
    url = f"http://{args.host}:{args.port}/"

    def run():
        """Load the weights and serve -- all in this one thread.

        This is the whole threading story, and it is why the server is not started
        on the calling thread: MLX's GPU stream is **thread-local**. Creating the
        model on one thread and driving it from another fails with
        ``RuntimeError: There is no Stream(gpu, 1) in current thread`` (measured;
        see ``tests/test_viewer_service.py``). One thread that both loads and
        serves is the arrangement that works, and it is the same arrangement the
        viewer had when the model was loaded before serving -- just with the load
        moved to the thread that owns the socket.
        """
        try:
            if not args.no_preload:
                app.start()
            if app.error:
                print(f"[ardy_viewer] WARNING: model unavailable: {app.error}", flush=True)
            elif app.runner is not None:
                print(
                    f"[ardy_viewer] model {app.model} loaded in {app.boot_seconds:.2f}s "
                    f"({app.runner.parameter_count:,} parameters, device {app.device})",
                    flush=True,
                )
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        except Exception as exc:  # noqa: BLE001 - the server must not die silently
            print(f"[ardy_viewer] fatal: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
            traceback.print_exc()
        finally:
            server.server_close()

    print(f"[ardy_viewer] serving {url}  (viewer dir: {VIEWER_DIR})", flush=True)
    if not args.no_preload:
        print("[ardy_viewer] loading weights before the first request...", flush=True)
    worker = threading.Thread(target=run, name="ardy-viewer", daemon=False)
    worker.start()
    if args.exit_after:
        # The shutdown must come from another thread: ``serve_forever`` blocks until
        # it is asked to stop, and only ``shutdown()`` (from elsewhere) can ask.
        def stop_later():
            time.sleep(args.exit_after)
            print(f"[ardy_viewer] --exit-after {args.exit_after}s reached; stopping", flush=True)
            server.shutdown()

        threading.Thread(target=stop_later, name="ardy-viewer-exit", daemon=True).start()
    try:
        worker.join()
    except KeyboardInterrupt:
        print("[ardy_viewer] interrupted", flush=True)
        server.shutdown()
        worker.join(timeout=10)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
