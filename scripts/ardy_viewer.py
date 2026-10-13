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
import json
import mimetypes
import os
import re
import sys
import threading
import time
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from collections import OrderedDict
from dataclasses import dataclass, replace
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
from ardy_mlx.router.jev import (  # noqa: E402
    ACTION_LIBRARY,
    JevRouter,
    decide as decide_route,
    known_intents_revision,
)
from ardy_mlx.router.jev import DEFAULT_DEADLINE_MS as DEFAULT_ROUTER_DEADLINE_MS  # noqa: E402
from ardy_mlx.sampling.diffusion import resolve_num_steps  # noqa: E402
from ardy_mlx.sampling.local_encoder import (  # noqa: E402
    DEFAULT_LLM2VEC_REPO,
    LOCAL_ENCODER_SUBDIR,
    REFERENCE_ENCODER_ID,
    LocalEncoder,
    LocalEncoderError,
)
from ardy_mlx.sampling.pipeline import (  # noqa: E402
    TextEmbedding,
    build_mlx_runner,
    duration_to_frames,
    load_text_embedding,
    prompt_hash,
    resolve_noise_windows,
    sha256_file,
    sha256_text,
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
# cached embeddings, keyed by (prompt_hash, encoder_id)
# --------------------------------------------------------------------------- #
#: The largest cache we will hash through on every ``/api/health``. Past this, the
#: directory is summarised rather than re-hashed (see :meth:`EmbeddingCache.list`).
#: Hashing is cheap per file and the check is what keeps the health endpoint
#: honest about *these* bytes, but "cheap" stops being true at scale.
_MAX_FINGERPRINT_ENTRIES = 512


def _slug(text: str) -> str:
    """A filesystem-safe directory name for an encoder id.

    The encoder id is the cache key's second half and contains spaces, ``+`` and
    commas (it spells out the quantization plan). The directory name is that string
    with anything awkward folded to ``-``; the **sidecar** keeps the id verbatim, so a
    directory name is a convenience and never the authority on what a vector is.
    """
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text)).strip("-")
    return slug or "local"


def _entry_priority(entry: dict) -> int:
    """Which vector a prompt should be conditioned on, lower being better.

    A reference vector **always** wins. It is the conditioning signal the golden
    clip was produced from, so using it is what keeps a viewer clip comparable with
    the reference at all; a locally-encoded vector is a different (valid) condition
    and would break that comparison silently. This ordering is the whole reason the
    two live in separate directories instead of one.
    """
    if entry.get("source") == "reference":
        return 0
    if entry.get("source") == "local":
        return 1
    return 2


class EmbeddingCache:
    """Every text vector this server can condition on, keyed by what produced it.

    A prompt can have more than one vector, and the difference is not cosmetic:

    * the **reference** vector (bf16, encoded on the reference machine) is the
      conditioning signal the golden clip came from. It lives at
      ``<embeddings-dir>/<prompt-hash>.npy``, the historical convention.
    * a **locally encoded** vector (the MLX encoder in ``llm2vec-mlx``) lives at
      ``<embeddings-dir>/mlx/<prompt-hash>.npy``, so the two never overwrite each
      other. It is a different model artifact and gives a genuinely different —
      equally valid — clip.

    The cache key is therefore ``(prompt_hash, encoder_id)``, where ``encoder_id``
    names the encoder *and its exact precision* (``mlx-llm2vec-4-bit-group-32-…``).
    Two vectors only ever share a slot when both halves match; a sidecar that cannot
    say what produced its vector is labelled rather than promoted.

    Nothing here invents a vector. A lookup miss raises
    :class:`ViewerError` ``no_cached_embedding_for_prompt`` and names both
    directories and the file that was looked for. Local encoding is a separate,
    explicit step (:meth:`ardy_viewer.ViewerApp._resolve_embedding`) that fails by
    name too — this class never substitutes.
    """

    def __init__(self, directory: Path = EMBEDDINGS_DIR):
        self.directory = Path(directory)
        self.by_key: Dict[Tuple[str, str], dict] = {}
        self.by_prompt: Dict[str, Dict[str, dict]] = {}
        self.text_index: Dict[str, str] = {}
        self.skipped: List[dict] = []
        self._fingerprints: "OrderedDict[Tuple[str, int, int], str]" = OrderedDict()

    # -- paths -----------------------------------------------------------
    def local_variants_dirs(self) -> List[Path]:
        """Every sub-directory of the cache: one per local encoder.

        The layout is ``<cache>/<prompt-hash>.npy`` for a reference vector and
        ``<cache>/<encoder-slug>/<prompt-hash>.npy`` for a local one. The directory
        *is* the encoder half of the cache key, which is why one prompt can hold a
        reference vector and several precisions of local vector at once without any
        of them overwriting another — the failure mode a single flat ``mlx/``
        directory has the moment two precisions are both in use.
        """
        if not self.directory.is_dir():
            return []
        return sorted(
            path for path in self.directory.iterdir()
            if path.is_dir() and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", path.name)
        )

    def local_dir_for(self, encoder_id: str) -> Path:
        """Where one encoder's local vectors live: ``<cache>/<slug>/``."""
        return self.directory / _slug(encoder_id)

    @property
    def local_variants_dir(self) -> Path:
        """The default local encoder's directory (kept for direct callers/tests)."""
        return self.directory / LOCAL_ENCODER_SUBDIR

    def _display(self, path: Path) -> str:
        try:
            return str(Path(path).relative_to(REPO_ROOT))
        except ValueError:
            return str(Path(path))

    def _scan_roots(self) -> List[Tuple[Path, str]]:
        """``(directory, source)`` pairs to scan, in precedence order.

        The top level holds **reference** vectors and nothing else: a local vector
        belongs in a sub-directory, so a file at the top level is a reference vector
        by construction (the historical convention, and the reason every existing
        cache keeps working). Each sub-directory is one local encoder. There is no
        recursion past one level — a deeper directory is not a cache layout and is
        reported as skipped rather than guessed at.
        """
        if not self.directory.is_dir():
            return []
        return [(self.directory, "reference")] + [
            (path, "local") for path in self.local_variants_dirs()
        ]

    # -- discovery -------------------------------------------------------
    def _fingerprint(self, path: Path, stat: os.stat_result) -> str:
        """Cached ``sha256`` of a file, keyed by identity *and* mtime/size.

        ``/api/health`` reports these digests and the tests poll it, so it is called
        far more often than the cache changes. The stat fields are part of the key on
        purpose: replacing a file with different bytes of the same length still
        changes ``st_mtime_ns``, so a stale digest cannot survive an edit.
        """
        key = (str(path), int(stat.st_mtime_ns), int(stat.st_size))
        digest = self._fingerprints.get(key)
        if digest is None:
            digest = sha256_file(path)
            self._fingerprints[key] = digest
            while len(self._fingerprints) > _MAX_FINGERPRINT_ENTRIES:
                self._fingerprints.popitem(last=False)
        else:
            self._fingerprints.move_to_end(key)
        return digest

    def _sidecar(self, npy_path: Path) -> Optional[dict]:
        sidecar = npy_path.with_suffix(".json")
        if not sidecar.is_file():
            return None
        try:
            loaded = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return loaded if isinstance(loaded, dict) else None

    def _scan_entry(self, path: Path, source: str, encoder_default: str) -> dict:
        """Normalise one ``.npy`` (+ sidecar) into a cache entry.

        Every field a caller reports is derived here from the file's own sidecar, so
        ``/api/health`` and ``/api/plan`` cannot disagree about where a vector came
        from. Fields the sidecar does not carry are reported as ``None`` — an
        explicit unknown, never a plausible default.
        """
        stat = path.stat()
        sidecar = self._sidecar(path) or {}
        text = sidecar.get("prompt") if isinstance(sidecar.get("prompt"), str) else None
        encoder_id = sidecar.get("encoder_id")
        if not isinstance(encoder_id, str) or not encoder_id:
            # A sidecar without an id: the reference convention (the historical files
            # predate the field), or a locally-encoded file whose provenance is simply
            # not recorded. The latter is *labelled unknown*, not attributed.
            if source == "reference" and isinstance(sidecar.get("encoder"), str):
                encoder_id = sidecar["encoder"]
            elif source == "reference":
                encoder_id = REFERENCE_ENCODER_ID
            else:
                encoder_id = "unknown-encoder"
        quantization = sidecar.get("quantization")
        if not isinstance(quantization, str):
            quantization = None
        token_count = sidecar.get("num_tokens")
        if not isinstance(token_count, int):
            token_count = None

        try:
            array = np.load(path, allow_pickle=False)
            shape = [int(dim) for dim in array.shape]
            tokens = int(array.shape[1]) if array.ndim >= 3 else (int(array.shape[0]) if array.ndim else 0)
        except (OSError, ValueError) as exc:  # unreadable -> listed, but not usable
            shape, tokens = None, None
            text = text or f"<unreadable: {exc}>"

        return {
            "prompt_hash": path.stem,
            "encoder_id": encoder_id,
            "source": source,
            "text": text if text is not None else f"<unknown prompt for {path.stem}>",
            "has_prompt_text": text is not None,
            "path": self._display(path),
            "path_abs": str(path),
            "embedding_sha256": self._fingerprint(path, stat),
            "shape": shape,
            "tokens": tokens,
            "num_tokens": token_count,
            "quantization": quantization,
            "encoder": sidecar.get("encoder"),
            "note": sidecar.get("note"),
            "reference_source": sidecar.get("reference_source"),
            "created_utc": sidecar.get("created_utc"),
            "git_sha": sidecar.get("git_sha"),
            "weights_sha256": sidecar.get("weights_sha256"),
            "manifest_sha256": sidecar.get("manifest_sha256"),
            "merged_bf16_sha256": sidecar.get("merged_bf16_sha256"),
            "bits": sidecar.get("bits"),
            "group_size": sidecar.get("group_size"),
        }

    def refresh(self) -> List[dict]:
        """Rescan the cache directory (root = reference, ``mlx/`` = local).

        Returns :meth:`list`. Both directories are read every time: the viewer
        installs a freshly-encoded vector by writing there, and a refresh is what
        makes it visible to the next request.
        """
        found: List[dict] = []
        skipped: List[dict] = []
        for root, source in self._scan_roots():
            for path in sorted(root.glob("*.npy")):
                stem = path.stem
                if not re.fullmatch(r"[0-9a-f]{16}", stem):
                    # Not one of ours (a scratch file, a README placeholder). It is
                    # reported as skipped rather than guessed at: an embedding whose
                    # name is not a prompt hash cannot be addressed.
                    skipped.append(
                        {
                            "path": self._display(path),
                            "source": source,
                            "reason": "file name is not a 16-hex prompt hash",
                        }
                    )
                    continue
                try:
                    entry = self._scan_entry(path, source, REFERENCE_ENCODER_ID)
                except OSError as exc:
                    skipped.append(
                        {"path": self._display(path), "source": source, "reason": str(exc)}
                    )
                    continue
                # The directory is the authority on which encoder a local vector
                # belongs to; the sidecar is a label that can be edited by hand. When
                # they disagree the file is still addressable, but it is labelled by
                # where it lives and the disagreement is reported.
                recorded = entry["encoder_id"]
                directory_encoder = root.name
                if source == "local" and recorded != directory_encoder:
                    entry["encoder_id"] = directory_encoder
                    entry["encoder_id_conflict"] = recorded
                found.append(entry)

        reference_encoder = next(
            (entry["encoder_id"] for entry in found if entry["source"] == "reference"),
            REFERENCE_ENCODER_ID,
        )

        by_prompt: Dict[str, Dict[str, dict]] = {}
        for entry in sorted(found, key=lambda item: (_entry_priority(item), item["encoder_id"])):
            by_prompt.setdefault(entry["prompt_hash"], {})[entry["encoder_id"]] = entry

        by_key: Dict[Tuple[str, str], dict] = {}
        text_index: Dict[str, str] = {}
        for digest, variants in by_prompt.items():
            for variant in variants.values():
                by_key[(digest, variant["encoder_id"])] = variant
            best = next(iter(variants.values()))
            if best["has_prompt_text"]:
                text_index[best["text"]] = digest

        self.by_key = by_key
        self.by_prompt = by_prompt
        self.text_index = text_index
        self.skipped = skipped
        self.reference_encoder_id = reference_encoder
        return self.list()

    # -- listing ---------------------------------------------------------
    def list(self) -> List[dict]:
        """Every prompt once, its preferred (reference-first) vector.

        One prompt with both a reference and a local vector appears **once**, as the
        reference — that is the entry a request will actually be conditioned on. The
        alternatives stay addressable through :meth:`variants` and
        :meth:`resolve_cached`.
        """
        entries: List[dict] = []
        for digest in sorted(self.by_prompt):
            variants = self.by_prompt[digest]
            best_id = min(variants, key=lambda key: _entry_priority(variants[key]))
            entries.append(variants[best_id])
        return entries

    def variants(self, prompt_hash: str) -> List[dict]:
        """Every cached vector for one prompt, best first (reference, then local)."""
        variants = self.by_prompt.get(prompt_hash, {})
        return [variants[key] for key in sorted(variants, key=lambda k: _entry_priority(variants[k]))]

    def local_variants(self, prompt_hash: str) -> List[dict]:
        return [entry for entry in self.variants(prompt_hash) if entry["source"] == "local"]

    # -- lookup ----------------------------------------------------------
    def _selector_digest(self, selector: str) -> Optional[str]:
        """Resolve a selector that names a prompt (hash, full sha256, or text).

        Text is resolved through an index built from the *stored* sidecar texts, so
        the by-hash path and the by-text path agree by construction and an arbitrary
        64-hex string can never be mistaken for a prompt it does not describe.
        """
        if selector in self.by_prompt:
            return selector
        lower = selector.strip().lower()
        if re.fullmatch(r"[0-9a-f]{64}", lower):
            for digest, variants in self.by_prompt.items():
                for entry in variants.values():
                    if entry["has_prompt_text"] and sha256_text(entry["text"]) == lower:
                        return digest
            return None
        if selector in self.text_index:
            return self.text_index[selector]
        digest = prompt_hash(selector)
        return digest if digest in self.by_prompt else None

    def resolve_cached(self, prompt_hash: str, encoder_id: Optional[str] = None) -> dict:
        """The cached entry for exactly ``(prompt_hash, encoder_id)``.

        With no ``encoder_id``, the prompt's **preferred** vector is returned — the
        reference when one exists, otherwise the best local one. With an
        ``encoder_id``, only that exact key matches: asking for one encoder's vector
        can never hand back another's.
        """
        variants = self.by_prompt.get(prompt_hash)
        if not variants:
            return None  # type: ignore[return-value]
        if encoder_id is not None:
            return variants.get(encoder_id)
        best_id = min(variants, key=lambda key: _entry_priority(variants[key]))
        return variants[best_id]

    def resolve(self, selector: str, encoder_id: Optional[str] = None) -> dict:
        """Resolve a prompt selector to a **cached** entry, or fail by name.

        This is the cache-only lookup; it never encodes and never substitutes. A
        miss is ``no_cached_embedding_for_prompt``, naming both directories and the
        file that was looked for. With ``encoder_id`` given, the miss is instead
        ``encoder_variant_not_cached``, because the prompt *is* available — just not
        from that encoder, and silently swapping in the other one is the failure this
        whole cache layout exists to prevent.
        """
        if selector is None or selector == "":
            raise ViewerError(
                "no_prompt_given",
                "a prompt is required: pass the prompt hash of a cached embedding, or its exact text",
            )
        digest = self._selector_digest(selector)
        if digest is not None:
            entry = self.resolve_cached(digest, encoder_id)
            if entry is not None:
                return entry
            variants = [variant["encoder_id"] for variant in self.variants(digest)]
            raise ViewerError(
                "encoder_variant_not_cached",
                (
                    f"prompt {digest} is cached, but not from encoder {encoder_id!r}; "
                    f"the cached encoder(s) are: {', '.join(variants) or '(none)'}"
                ),
                detail=(
                    "Refusing to condition on a different encoder's vector for the same "
                    "prompt: the two are different artifacts and produce different clips."
                ),
            )

        digest = prompt_hash(selector)
        within = self.directory.relative_to(REPO_ROOT) if self._under_repo(self.directory) else self.directory
        available = ", ".join(sorted(self.by_prompt)) or "(none)"
        raise ViewerError(
            "no_cached_embedding_for_prompt",
            f"no cached embedding for this prompt; expected {within}/{digest}.npy",
            detail=(
                f"selector={selector[:120]!r} resolves to prompt hash {digest}. "
                f"Available prompt hashes: {available}. The viewer can encode this "
                f"prompt locally instead (see local_encoder); this error means the "
                f"cache-only lookup was asked for it."
            ),
        )

    @staticmethod
    def _under_repo(path: Path) -> bool:
        try:
            Path(path).relative_to(REPO_ROOT)
            return True
        except ValueError:
            return False

    # -- installation ----------------------------------------------------
    def install(self, result) -> dict:
        """Copy an :class:`EncodeResult` into the cache, sidecar and all.

        The vector the encoder wrote is moved with :func:`os.replace` (atomic within a
        filesystem), so a reader can never see a half-written ``.npy``: either the
        entry is absent or it is complete. The sidecar is written *after* the vector
        and carries the full quantization name, so a cached entry always explains
        itself.

        The bytes are not re-derived or re-quantized here. This is a copy of the
        encoder's own output plus its provenance — the same bytes the subprocess
        wrote, addressed by ``(prompt_hash, encoder_id)``.
        """
        target_dir = self.local_dir_for(result.encoder_id)
        target_dir.mkdir(parents=True, exist_ok=True)
        vector_target = target_dir / f"{result.prompt_hash}.npy"
        staged = Path(result.vector_path)
        record = result.as_sidecar()

        # A file in this encoder's own directory can only be this encoder's vector, so
        # a mismatch here means a sidecar was edited or a directory was renamed by
        # hand. Refuse rather than silently relabel somebody else's bytes.
        if vector_target.is_file():
            existing = self._sidecar(vector_target) or {}
            if existing.get("encoder_id") not in (None, record.get("encoder_id")):
                raise ViewerError(
                    "local_encoder_failed",
                    (
                        f"{vector_target} already holds a vector from encoder "
                        f"{existing.get('encoder_id')!r}; refusing to overwrite it with "
                        f"{record.get('encoder_id')!r}"
                    ),
                    status=500,
                    detail=(
                        "Two encoders' vectors for one prompt are two cache keys, in two "
                        "directories. Move the old file aside if that label is wrong."
                    ),
                )
        os.replace(staged, vector_target)
        sidecar_target = vector_target.with_suffix(".json")
        sidecar_target.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

        self.refresh()
        entry = self.resolve_cached(result.prompt_hash, result.encoder_id)
        if entry is None:  # pragma: no cover - install() just wrote it
            raise ViewerError(
                "local_encoder_failed",
                f"the encoded vector was written to {vector_target} but did not come back from a rescan",
                status=500,
            )
        return entry

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
# continuation: what the next prompt starts from
# --------------------------------------------------------------------------- #
# A prompt change is not a restart. The page sends no ``reset``, so a request that
# arrives while a clip is on screen is generated *from* that clip: its trailing
# frames become the window's clean history, the accumulated translation and the
# heading are carried over, and the served clip opens on the frame the page is
# already showing. ``ArdySampler`` is the loop that hands one window to
# ``make_step_fn`` -- the same window kernel the multi-window golden comparison
# validated -- so nothing here re-implements the window advance
# (``docs/decisions/0007-streaming-demo.md``).

#: What ``/api/plan``'s ``continuation`` block means, in one sentence a reader can
#: check against the frames.
CONTINUATION_NOTE = (
    "a prompt change continues the clip that is on screen: the trailing frames are the "
    "window's clean history, the accumulated root translation and the heading are kept, "
    "and frame 0 of this clip is the previous clip's own last frame (carried over verbatim, "
    "not regenerated). Pass reset=true to start from noise instead."
)

#: Why a continuation was asked for and did not happen.
CONTINUATION_NO_PREVIOUS = "no_previous_clip"
CONTINUATION_RESET = "reset_requested"
CONTINUATION_MODEL_CHANGED = "model_changed"
CONTINUATION_UNSUPPORTED = "runner_has_no_continuation_path"

#: Golden noise reproduces the reference only from its own beginning: the recorded
#: x_T belongs to the reference run's *first* window with the reference run's own
#: history, so injecting it into a window whose history is the page's current clip
#: is not a reproduction of anything. A continuation therefore falls back to MLX's
#: own RNG, by name, and says so in the payload and on the page.
CONTINUATION_NOISE_FALLBACK = "golden noise does not survive a continuation"

#: How far the first generated frame may sit from the frame on screen before the
#: request is refused instead of answered. One frame is 50 ms at 20 fps, so a
#: quarter of a metre is 5 m/s -- faster than any motion in this checkpoint, and
#: well above the 60 mm/frame these prompts actually produce. It is a *carry-over*
#: invariant: a wrong anchor or a mixed unit lands here, loudly, rather than in a
#: clip that silently teleports.
CONTINUATION_SEAM_LIMIT_M = 0.25


def continuation_history_frames(layout) -> int:
    """How many frames of the previous clip condition the next window.

    The checkpoint's own crop budget, never a new constant: ``default_history_frames``
    (160 for Core/20fps/H40) is what the window logic was validated with.
    """
    return int(layout.default_history_frames)


@dataclass
class ContinuationState:
    """The clip on screen, in the form the next window needs.

    ``frames`` are **normalised** 330-dim explicit features in world coordinates --
    the space ``MotionRep.normalize``/``unnormalize`` bracket and the space the
    window loop hands back and forth. ``joints``/``roots``/``contacts`` are the last
    frame exactly as it was served, so the next request can open on the frame the
    page is already showing rather than on a new one.
    """

    frames: np.ndarray        # [1, T, 330] normalised explicit, world frame
    joints: np.ndarray        # [27, 3]     metric world joint positions
    roots: np.ndarray         # [3]         metric world root position
    contacts: np.ndarray      # [4]         foot contacts, 0/1
    prompt_hash: str
    prompt_text: str
    model: str


def continuation_decision(
    previous: Optional[ContinuationState],
    *,
    reset: bool,
    can_continue: bool,
    model: str,
) -> dict:
    """Whether this request continues the previous clip, and if not, why. **Pure.**

    The order matters: an explicit ``reset`` is honoured before anything else, so
    the negative control ("a forced reset must jump back to the origin") is the
    same code path a browser would take, not a claim about it.
    """
    if reset:
        return {"requested": False, "applied": False, "reason": CONTINUATION_RESET}
    if previous is None:
        return {"requested": True, "applied": False, "reason": CONTINUATION_NO_PREVIOUS}
    if previous.model != model:
        return {"requested": True, "applied": False, "reason": CONTINUATION_MODEL_CHANGED}
    if not can_continue:
        return {"requested": True, "applied": False, "reason": CONTINUATION_UNSUPPORTED}
    return {"requested": True, "applied": True, "reason": None}


def _with_injected_noise(step_fn, noise_fn):
    """Give the sampler loop the caller's ``x_T`` instead of its own draw.

    ``ArdySampler`` draws a fresh ``randn`` per window internally; the loop's own
    step function is the one place a window's noise can be replaced without
    touching the loop. The wrapper keeps the loop's contract -- one
    ``DenoiseResult`` per ``DenoiseRequest`` -- and changes nothing else.
    """

    def step_with_noise(request):
        shape = tuple(int(v) for v in np.asarray(request.generation_noise).shape)
        return step_fn(replace(request, generation_noise=noise_fn(int(request.step), shape)))

    step_with_noise.batch = getattr(step_fn, "batch", None)
    step_with_noise.reset = getattr(step_fn, "reset", None)
    return step_with_noise


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
        local_encoder: Optional[LocalEncoder] = None,
        router: Optional[object] = None,
        router_enabled: bool = True,
        router_deadline_ms: int = DEFAULT_ROUTER_DEADLINE_MS,
    ):
        self.model = model
        self.device = device
        self.layout = PRESETS[model]
        self.allow_cpu = allow_cpu
        self.embeddings = EmbeddingCache(embeddings_dir)
        self.checkpoint_dir = checkpoint_dir
        # ``None`` means "no local encoding configured" -- every prompt must then
        # come out of the cache, exactly as v1 behaved. The CLI resolves the sibling
        # ``llm2vec-mlx`` checkout and passes an encoder; a test that wants the
        # cached-only semantics simply does not.
        self.local_encoder = local_encoder
        # The router is *advisory*: it reads the sentence and names a route, and
        # ``_route_request`` is the only place that turns that into an action. Three
        # ways in, and the difference is visible in every payload:
        #   * ``router`` given        -> use it (the tests inject a fake here);
        #   * ``router_enabled`` True -> build one from the environment; it reports
        #     ``router_unavailable`` by name when no API key is set;
        #   * ``router_enabled`` False-> ``--no-router``: ``router_disabled``.
        if router is not None:
            self.router = router
        elif router_enabled:
            self.router = JevRouter.from_environment(deadline_ms=router_deadline_ms)
        else:
            self.router = JevRouter.disabled_router(deadline_ms=router_deadline_ms)
        #: The text of the last clip this viewer planned, which is what the next
        #: request's ``context.current_action`` reports. The server owns it rather
        #: than the page, because it is the server that knows what it last drew.
        self.last_action: Optional[str] = None
        #: The clip on screen, in the form the *next* window needs (see
        #: :class:`ContinuationState`). ``None`` until one has been planned, and
        #: replaced -- never silently reused across a ``reset``.
        self.continuation: Optional[ContinuationState] = None
        self.lock = threading.Lock()
        self.runner = None
        self.error: Optional[str] = None
        self.boot_seconds: Optional[float] = None
        #: ``(prompt_hash, encoder_id)`` -> loaded :class:`TextEmbedding`.
        self.embedding_cache: Dict[Tuple[str, str], object] = {}
        #: ``(prompt_hash, encoder_id)`` -> :class:`LocalEncoderError` that is known to
        #: be reproducible, so a second request answers in microseconds and the page
        #: shows the original text rather than a fresh, differently-worded one.
        self.encoder_failures: Dict[Tuple[str, str], LocalEncoderError] = {}

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
            embedding = self._embedding_for(entries[0]) if entries else _empty_embedding()
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
    def _encoder_id(self) -> Optional[str]:
        """The encoder every local encode is attributed to, or ``None`` when disabled."""
        return self.local_encoder.status().encoder_id if self.local_encoder is not None else None

    def _cache_key(self, prompt_hash: str, encoder_id: str) -> Tuple[str, str]:
        return (prompt_hash, encoder_id)

    def _embedding_for(self, entry: dict):
        """The loaded :class:`TextEmbedding` for one cache entry (keyed by both halves)."""
        key = self._cache_key(entry["prompt_hash"], entry["encoder_id"])
        cached = self.embedding_cache.get(key)
        if cached is not None:
            return cached
        embedding = load_text_embedding(entry["path"])
        self.embedding_cache[key] = embedding
        return embedding

    def encoder_status(self, *, refresh: bool = True) -> Optional[dict]:
        if self.local_encoder is None:
            return None
        return self.local_encoder.status(refresh=refresh).as_dict()

    def _encoded_provenance(self) -> dict:
        """What to report when the vector came from this machine."""
        status = self.local_encoder.status() if self.local_encoder is not None else None
        return {
            "kind": "local-mlx",
            "label": "local MLX (this machine)",
            "encoder_id": status.encoder_id if status else None,
            "quantization": status.quantization if status else None,
            "note": (
                "encoded on this machine by the MLX port of the LLM2Vec text encoder "
                "(the sibling llm2vec-mlx checkout, reached through a subprocess). The "
                "conditioning vector is close to the reference bf16 one but not "
                "identical, so this clip is a different, equally valid sample -- it is "
                "not aligned with the golden clip."
            ),
        }

    def _reference_provenance(self, entry: dict) -> dict:
        return {
            "kind": "reference",
            "label": "reference bf16 (golden)",
            "encoder_id": entry["encoder_id"],
            "quantization": entry.get("quantization") or "bfloat16 (no quantization)",
            "source": entry.get("reference_source") or entry.get("source"),
            "note": (
                "the reference machine's own bf16 vector -- the conditioning signal the "
                "golden clip was generated from; used preferentially so a viewer clip "
                "stays comparable with it"
            ),
        }

    def _variant_provenance(self, entry: dict) -> dict:
        if entry["source"] == "reference":
            return self._reference_provenance(entry)
        if entry["source"] == "local":
            provenance = self._encoded_provenance()
            return {**provenance, "encoder_id": entry["encoder_id"]}
        return {
            "kind": "unknown",
            "label": f"unlabelled ({entry['source']})",
            "encoder_id": entry["encoder_id"],
            "quantization": entry.get("quantization"),
            "note": "this cache entry does not record what produced it",
        }

    @staticmethod
    def _priority(entry: dict) -> int:
        return _entry_priority(entry)

    def _local_error(self, name: str, message: str, *, detail=None, status: int = 400):
        raise ViewerError(name, message, status=status, detail=detail)

    def _raise_encoder_failure(self, failure: LocalEncoderError, prompt_hash: str):
        """Turn a :class:`LocalEncoderError` into the viewer's named failure.

        The encoder's own words travel in ``detail`` unchanged: this repository does
        not paraphrase a failure it does not own, and the page shows the raw text.
        """
        detail_parts = [f"prompt hash: {prompt_hash}"]
        if failure.exit_code is not None:
            detail_parts.append(f"encoder exit code: {failure.exit_code}")
        if failure.detail:
            detail_parts.append(failure.detail)
        status = 503 if failure.name == "local_encoder_unavailable" else 502
        raise ViewerError(
            failure.name,
            failure.message,
            status=status,
            detail="\n".join(detail_parts),
        )

    def _encode_locally(self, prompt: str, prompt_hash: str) -> dict:
        """Run the local encoder on one prompt, install the vector, return its entry.

        Called only when no cached vector for *this* prompt exists. The temporary
        directory is private to the call and removed afterwards, so a failed encode
        leaves nothing behind for a later request to trip over.
        """
        import tempfile

        with tempfile.TemporaryDirectory(prefix="ardy-encode-") as staging:
            try:
                result = self.local_encoder.encode(
                    prompt, prompt_hash=prompt_hash, out_dir=Path(staging)
                )
            except Exception as exc:  # noqa: BLE001 - the adapter may not have run at all
                # The adapter is supposed to return its failures; if it ever raises,
                # that is still a named encoder failure and never a substitution.
                result = LocalEncoderError(
                    "local_encoder_failed",
                    f"the local encoder adapter raised {type(exc).__name__}: {exc}",
                )
            if isinstance(result, LocalEncoderError):
                key = self._cache_key(prompt_hash, self._encoder_id() or "unavailable")
                if result.soft:
                    self.encoder_failures[key] = result
                self._raise_encoder_failure(result, prompt_hash)
            # ``install`` writes the sidecar and rescans, so the returned entry's
            # embedding_sha256 is of the bytes that are now on disk -- not of the
            # staging file, which no longer exists.
            entry = self.embeddings.install(result)
            return entry, float(result.encode_ms)

    def _reusable_variant(self, prompt_hash: str) -> Optional[dict]:
        """The cached vector for ``prompt_hash`` this viewer may condition on.

        A **reference** vector always qualifies — that is the vector the golden clip
        came from, and preferring it is the whole point of the precedence rule. A
        **local** vector only qualifies when it was produced by the encoder this
        viewer is *currently* configured for: a vector from a different precision is a
        different artifact, and quietly conditioning on it while the HUD names the
        configured encoder is exactly the silent mix the ``(prompt_hash, encoder_id)``
        key exists to prevent. A non-matching local vector is therefore a miss, and the
        request either encodes afresh or fails by name.
        """
        best = self.embeddings.resolve_cached(prompt_hash)
        if best is None or best["source"] == "reference":
            return best
        configured = self._encoder_id()
        if configured is None:
            # No local encoder configured, so there is nothing to encode with and no
            # precision to match: the labelled local vector is reused as itself, and
            # the payload reports its own encoder_id rather than the viewer's.
            return best
        for variant in self.embeddings.variants(prompt_hash):
            if variant["source"] != "local":
                continue
            if variant["encoder_id"] == configured:
                return variant
        return None

    def _embedding_for_request(self, prompt: str) -> Tuple[dict, dict]:
        """Resolve a prompt to exactly one cache entry, or fail by name.

        The order is fixed and is the whole of the source-precedence rule:

        1. a **reference** vector for this prompt, if one is cached;
        2. the best **local** vector for this prompt, if one is cached;
        3. otherwise **encode here** -- and only if the local encoder is configured
           and usable. A failure at this step is named (``local_encoder_unavailable``
           or ``local_encoder_failed``) and ends the request.

        There is no fourth branch. In particular, step 3 never falls back to a
        *different* prompt's vector, and it never falls back to another encoder's
        vector for the same prompt: both would produce a perfectly convincing clip
        under the wrong conditioning signal.
        """
        if prompt is None or prompt == "":
            raise ViewerError(
                "no_prompt_given",
                "a prompt is required: pass the prompt hash of a cached embedding, or its exact text",
            )

        digest = self.embeddings._selector_digest(prompt)
        if digest is not None:
            cached = self._reusable_variant(digest)
            if cached is not None:
                return cached, {"kind": "cache", "encode_ms": None}

        if self.local_encoder is None:
            if digest is not None and self.embeddings.variants(digest):
                # The prompt is known to the cache but only from encoders this viewer
                # is not configured for. Say that, rather than "no cached embedding".
                variants = [variant["encoder_id"] for variant in self.embeddings.variants(digest)]
                self._local_error(
                    "encoder_variant_not_cached",
                    (
                        f"prompt {digest} is cached, but not from encoder "
                        f"{self._encoder_id()!r}; the cached encoder(s) are: "
                        f"{', '.join(variants) or '(none)'}"
                    ),
                    detail=(
                        "Refusing to condition on a different encoder's vector for the same "
                        "prompt: the two are different artifacts and produce different clips."
                    ),
                )
            self.embeddings.resolve(prompt)  # raises no_cached_embedding_for_prompt

        key = self._cache_key(prompt_hash(prompt), self._encoder_id() or "unavailable")

        with self.lock:
            # Re-check under the lock: a pipelined duplicate request must not encode
            # the same prompt twice (or be answered with a different error).
            digest = self.embeddings._selector_digest(prompt)
            if digest is not None:
                cached = self._reusable_variant(digest)
                if cached is not None:
                    return cached, {"kind": "cache", "encode_ms": None}
            failure = self.encoder_failures.get(key)
            if failure is not None:
                self._raise_encoder_failure(failure, key[0])

            status = self.local_encoder.status()
            if not status.available:
                failure = LocalEncoderError(
                    "local_encoder_unavailable",
                    status.reason or "the local text encoder is not usable",
                    detail=json.dumps(status.checks, indent=2, sort_keys=True),
                )
                self.encoder_failures[key] = failure
                self._raise_encoder_failure(failure, key[0])

            entry, encode_ms = self._encode_locally(prompt, key[0])
        return entry, {"kind": "encoded", "encode_ms": encode_ms}

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
            "embeddings_detail": {
                "count": len(entries),
                # A prompt with both a reference and a local vector has two entries
                # keyed by (prompt_hash, encoder_id); ``embeddings`` shows the one a
                # request will use, this shows everything that is addressable.
                "variants": sum(len(self.embeddings.variants(digest)) for digest in self.embeddings.by_prompt),
                "cache_keys": [
                    [prompt_hash, encoder_id]
                    for (prompt_hash, encoder_id) in sorted(self.embeddings.by_key)
                ],
                "by_source": {
                    source: sum(1 for entry in self.embeddings.by_key.values() if entry["source"] == source)
                    for source in ("reference", "local")
                },
                "local_dirs": [
                    _display_path(path) for path in self.embeddings.local_variants_dirs()
                ],
                "skipped": list(self.embeddings.skipped),
            },
            "encoder": self.encoder_status(refresh=True)
            or {
                "enabled": False,
                "available": False,
                "state": "disabled",
                "reason": (
                    "no local text encoder is configured; start the viewer with "
                    "--llm2vec-repo (or leave the default sibling checkout) to enable it"
                ),
                "encoder_id": None,
                "quantization": None,
                "bits": None,
                "repo": None,
                "interpreter": None,
                "encode_script": None,
                "weights_dir": None,
                "manifest": None,
                "probe_ms": None,
                "checks": {},
            },
            "golden_references": [
                {
                    "duration_s": duration,
                    "path": _display_path(path),
                    "available": (path / "out").is_dir() and any((path / "out").glob("*.npz")),
                }
                for duration, path in sorted(GOLDEN_RUNS.items())
            ],
            # Which durations have a *recorded x_T* (a different fact from which
            # have a golden clip). The page uses this to decide, before sending
            # anything, whether "golden" noise is even possible for the duration it
            # is about to ask for -- instead of discovering it with a 400.
            "golden_noise": [
                {
                    "duration_s": duration,
                    "available": bool(files),
                    "windows": len(files),
                    "source_files": [_display_path(path) for path in files],
                }
                for duration in sorted(GOLDEN_RUNS)
                for files in (golden_noise_files(duration),)
            ],
            "continuation": {
                # A prompt change is not a restart: the next plan is generated from
                # the clip on screen unless it is explicitly reset.
                "supported": self.continuation_supported(),
                "active": self.continuation is not None,
                "history_frames": continuation_history_frames(self.layout),
                "frames": (int(self.continuation.frames.shape[1]) if self.continuation else None),
                "prompt_hash": (self.continuation.prompt_hash if self.continuation else None),
                "prompt_text": (self.continuation.prompt_text if self.continuation else None),
                "note": CONTINUATION_NOTE,
            },
            "limits": {
                "text_encoder": (
                    "a local MLX text encoder is available (the sibling llm2vec-mlx checkout, "
                    "reached by subprocess); an uncached prompt is encoded here, about half a "
                    "second, once per prompt, then cached. It is a close but different artifact "
                    "from the reference bf16 encoder, so a locally-conditioned clip is NOT "
                    "aligned with the golden clip -- it is a different, equally valid sample. "
                    "Reference bf16 vectors are used preferentially when a prompt has one."
                ),
                "tokenization": (
                    "prompts are encoded and cached whole; there is no token-level conditioning "
                    "and no prompt interpolation"
                ),
                "postprocess": "MotionCorrection (C++ IK/trajectory) is not built; foot skating remains",
                "rng": "MLX's RNG is not CUDA's Philox stream; only golden noise reproduces the reference",
                "auth": "none: loopback-only development viewer",
                "threading": "one request at a time (the MLX GPU stream is thread-local)",
                "encoder_latency": (
                    "the local encoder runs in its own process, so the first prompt after a "
                    "cache miss pays its cold start (about 1-2 s) before the ~0.4 s encode"
                ),
            },
            "router": {
                # The same six named fields a plan carries -- state, router_id, route,
                # probabilities, latency_ms, cache_hit -- plus what a reader needs to
                # tell a guess from a reading. Before the first plan, ``last`` is null
                # and the top-level ``state`` still says whether the router *would*
                # run, which is the fact health exists to report.
                **self.router.status(),
                "library_entries": len(self.library_entries()),
                "known_intents": self.known_intents(),
            },
            "served_by": f"scripts/ardy_viewer.py ({HOST})",
        }

    # -- routing: the sentence -> a route -> an action --------------------
    def library_entries(self) -> List[dict]:
        """The motions this viewer can load *directly*, as ``{prompt_hash, text}``.

        This is the cache, filtered to what a request could actually resolve: one
        preferred entry per prompt. It is what ``context.library_entries`` carries and
        what ``route == library`` loads from, so the answer the model picks from and
        the entries the code can load are the same list by construction.
        """
        entries = []
        for entry in self.embeddings.list():
            text = entry.get("text")
            digest = entry.get("prompt_hash")
            if text and digest:
                entries.append({"prompt_hash": digest, "text": text})
        entries.sort(key=lambda item: item["prompt_hash"])
        return entries

    def known_intents(self) -> List[str]:
        """The intents the engine already knows, as the model is told them.

        The library *is* the set of motions this engine can already produce, so this
        is its texts. It is derived rather than configured: an intent list that could
        drift from the cache would let ``known`` answer yes about an ability that is
        no longer on disk.
        """
        return [entry["text"] for entry in self.library_entries()]

    def _utterance_for_request(self, prompt: Optional[str]) -> Optional[str]:
        """The sentence Jev is asked about -- from the cache when a hash was sent.

        A prompt hash carries no text of its own, so the cached sidecar's ``prompt``
        is what the user actually said. When a hash has no cached text there is
        nothing to judge, and ``None`` here becomes ``router_no_utterance`` rather
        than a judgment about a hex string.
        """
        if prompt is None or prompt == "":
            return None
        digest = self.embeddings._selector_digest(prompt)
        if digest is None:
            return str(prompt)
        entry = self.embeddings.resolve_cached(digest)
        if entry is not None and entry.get("text"):
            return str(entry["text"])
        for variant in self.embeddings.variants(digest):
            if variant.get("text"):
                return str(variant["text"])
        return None

    def _open_judgment(self, prompt: Optional[str]):
        """Start the judgment and hand back the handle, the library and the intents.

        Split from :meth:`_read_judgment` on purpose. The caller resolves the embedding
        *between* the two, so Jev and the encoder are in flight at the same time and
        the judgment's wall clock is largely hidden behind work the request had to do
        anyway. Resolving here instead -- which is what this used to do -- serialises
        them and turns every uncached prompt into a deadline wait.
        """
        library = self.library_entries()
        intents = [entry["text"] for entry in library]
        utterance = self._utterance_for_request(prompt)
        handle = self.router.submit(
            utterance,
            current_action=self.last_action,
            known_intents=intents,
            library_entries=library,
            prompt_hash=prompt_hash(utterance) if utterance else "",
        )
        return handle, library, intents

    def _read_judgment(self, handle, library: List[dict], intents: List[str]) -> Tuple[dict, dict, Optional[dict]]:
        """Collect the judgment, map it to an action, and report both.

        Returns ``(router_block, decision, library_entry)``. Every path through this
        method ends in a ``router`` block that says what happened; there is no branch
        that returns a route without a state.
        """
        verdict = handle.resolve()
        router = self.router
        decision = decide_route(
            verdict.judgment,
            state=verdict.state,
            deadline_ms=getattr(router, "deadline_ms", DEFAULT_ROUTER_DEADLINE_MS),
            library_entries=library,
            detail=verdict.detail,
        )
        block = verdict.router_block(
            deadline_ms=getattr(router, "deadline_ms", DEFAULT_ROUTER_DEADLINE_MS),
            model=getattr(router, "model", None),
            sdk_version=getattr(router, "sdk_version", None),
        )
        # The revision the judgment was conditioned on travels with the payload: it is
        # the third part of the cache key, and a reader can see *which* library the
        # answer was chosen against.
        block["known_intents_rev"] = known_intents_revision(intents, library)
        return block, decision.as_dict(), decision.library_entry

    def plan(
        self,
        prompt: str,
        seed: Optional[int] = None,
        duration: Optional[float] = None,
        noise: str = "golden",
        diffusion_steps: Optional[int] = None,
        cfg_weight=None,
        include_frames: bool = False,
        reset: bool = False,
    ) -> dict:
        if self.runner is None:
            raise ViewerError(
                "model_unavailable",
                self.error or "the MLX checkpoint is not loaded",
                status=503,
            )
        request_started = time.perf_counter()
        # The judgment starts here and the encoder starts two lines later, so the two
        # overlap. The verdict is collected *after* the encode, which is where the
        # deadline has been ticking down; a judgment that misses it degrades by name.
        handle, library, intents = self._open_judgment(prompt)
        # Resolving the prompt may encode it here (a ~0.4 s subprocess on a cache
        # miss); the time that takes is reported separately from sampling, so the page
        # can show which half of the wall clock was the encoder.
        entry, embedding_source = self._embedding_for_request(prompt)
        router_block, decision, library_entry = self._read_judgment(handle, library, intents)
        if decision["action"] == ACTION_LIBRARY and library_entry:
            # "Load the library entry" is a real substitution, and it is the one thing
            # the route changes about the request: the clip is conditioned on the
            # entry's own cached vector, not on the sentence that pointed at it.
            entry, embedding_source = self._embedding_for_request(library_entry["prompt_hash"])

        if not decision["starts_new_action"]:
            # ``non_motion``, or a sentence the model does not read as a motion
            # request. Nothing is generated and the clip on screen is left alone: the
            # one outcome this viewer must never produce is a body that moves because
            # of a sentence that did not ask it to.
            return {
                "kind": "hold",
                "reason": decision["reason"],
                "router": router_block,
                "decision": decision,
                "duration_s": float(duration) if duration is not None else None,
                "seed": seed,
                "prompt": entry.get("text"),
                "held_action": self.last_action,
                "timing": {
                    "encode_ms": embedding_source.get("encode_ms"),
                    "generate_ms": None,
                    "total_ms": (time.perf_counter() - request_started) * 1000.0,
                    "embedding_source": embedding_source.get("kind"),
                    "cache_hit": embedding_source.get("kind") == "cache",
                    "router_ms": router_block.get("latency_ms"),
                },
                "served_by": f"{HOST} scripts/ardy_viewer.py",
            }

        embedding = self._embedding_for(entry)
        frames_wanted = validate_duration(duration, self.layout)
        num_steps = resolve_num_steps(diffusion_steps, NUM_BASE_STEPS)
        weights = self._cfg_weight(cfg_weight)
        windows = duration_to_windows(duration, self.layout)

        # Whether this request continues the clip on screen or starts a new one.
        # Decided before the noise is built, because a continuation changes what
        # "golden noise" can honestly mean (see ``CONTINUATION_NOISE_FALLBACK``).
        continuation_choice = continuation_decision(
            self.continuation,
            reset=reset,
            can_continue=self.continuation_supported(),
            model=self.model,
        )
        noise_requested = noise
        noise_sidecar_extra: Dict[str, object] = {}
        if continuation_choice["applied"] and noise == "golden":
            noise = "random"
            noise_sidecar_extra = {
                "requested": noise_requested,
                "fallback_from": noise_requested,
                "fallback_reason": CONTINUATION_NOISE_FALLBACK,
            }

        noise_fn, noise_sidecar = build_noise_fn(noise, duration, self.runner, seed)
        noise_sidecar = {**noise_sidecar, **noise_sidecar_extra}

        runner = self.runner
        original = runner.embedding
        carried: Optional[ContinuationState] = self.continuation if continuation_choice["applied"] else None
        served_norm: Optional[np.ndarray] = None
        try:
            # The runner caches one embedding; swap in this prompt's. Cheap (an
            # mx.array of [1, L, 4096]) and it keeps the MLX call in one place.
            runner.embedding = embedding
            with self.lock:
                started = time.perf_counter()
                if continuation_choice["applied"]:
                    # Generated *from* the clip on screen: the trailing frames are
                    # the window's clean history and the accumulated translation
                    # and heading are kept. Nothing is reset, and the clip is not
                    # replayed from frame 0.
                    generated, served_norm, run_info = self._generate_continued(
                        runner=runner,
                        embedding=embedding,
                        previous=carried,
                        frames_wanted=frames_wanted,
                        num_steps=num_steps,
                        weights=weights,
                        seed=seed,
                        noise_fn=noise_fn,
                    )
                    mx.eval(generated)
                    generate_ms = (time.perf_counter() - started) * 1000.0
                else:
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
                    run_info = {
                        "offsets": [int(value) for value in out["offsets"]],
                        "history_frames": int(out["history_frames"]),
                    }
                    mx.eval(generated)
                    generate_ms = (time.perf_counter() - started) * 1000.0
        finally:
            runner.embedding = original

        posed = posed_joints_from_features(generated)
        mx.eval(posed)
        joints = np.asarray(posed, dtype=np.float32)[0]
        roots = np.asarray(generated[..., 0:3], dtype=np.float32)[0]
        contacts = (np.asarray(generated[..., -4:], dtype=np.float32) > 0.5)[0]

        # The seam, measured on the frames that are about to be served. Frame 0 is
        # the previous clip's own last frame, so the first number is a measurement
        # of the carry-over rather than a promise; the second is the model's own
        # next step, which is the one to compare against a reset.
        seam = None
        if continuation_choice["applied"] and carried is not None:
            seam = {
                "carried_frame_joint_mm": float(np.abs(joints[0] - carried.joints).max() * 1000.0),
                "carried_frame_root_mm": float(np.linalg.norm(roots[0] - carried.roots) * 1000.0),
                "model_first_joint_mm": float(np.abs(joints[1] - carried.joints).max() * 1000.0),
                "model_first_root_mm": float(np.linalg.norm(roots[1] - carried.roots) * 1000.0),
                "previous_last_root": [float(value) for value in carried.roots],
                "model_first_root": [float(value) for value in roots[1]],
            }

        # The state the *next* prompt continues from: the clip exactly as it was
        # served, in the space the window loop hands back and forth. Recorded from
        # what the page will draw -- not from an intermediate tensor -- so "what
        # the next clip starts from" and "what is on screen" cannot disagree.
        if self.continuation_supported():
            motion_rep = runner.batch.motion_rep
            if served_norm is None:
                served_norm = np.asarray(motion_rep.normalize(generated), dtype=np.float32)
            self.continuation = ContinuationState(
                frames=np.asarray(served_norm, dtype=np.float32),
                joints=np.array(joints[-1], dtype=np.float32, copy=True),
                roots=np.array(roots[-1], dtype=np.float32, copy=True),
                contacts=np.array(contacts[-1], dtype=np.uint8, copy=True),
                prompt_hash=entry["prompt_hash"],
                prompt_text=entry["text"],
                model=self.model,
            )

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
                # Which vector this clip was actually conditioned on, and what made it.
                # The page renders this verbatim in the HUD: a locally-encoded vector
                # must never be presentable as the reference one.
                "source": entry["source"],
                "source_kind": (
                    "reference" if entry["source"] == "reference"
                    else "local-mlx" if entry["source"] == "local"
                    else f"unlabelled ({entry['source']})"
                ),
                "encoder_id": entry["encoder_id"],
                "quantization": entry.get("quantization"),
                "encoder": entry.get("encoder"),
                "reference_source": entry.get("reference_source"),
                "bits": entry.get("bits"),
                "group_size": entry.get("group_size"),
                "provenance": (
                    self._reference_provenance(entry) if entry["source"] == "reference"
                    else self._variant_provenance(entry)
                ),
            },
            "noise": noise_sidecar,
            "num_autoregressive_steps": max(len(run_info["offsets"]) - 1, 0),
            "offsets": list(run_info["offsets"]),
            "history_frames": int(run_info["history_frames"]),
            # What the next ``/api/plan`` without ``reset`` will start from. The
            # page sends no reset flag, so this is the normal path, not an option.
            "continuation": {
                "requested": bool(continuation_choice["requested"]),
                "applied": bool(continuation_choice["applied"]),
                "reason": continuation_choice["reason"],
                "history_frames": int(run_info["history_frames"]) if continuation_choice["applied"] else None,
                "seam": seam,
                "previous": (
                    {"prompt_hash": carried.prompt_hash, "prompt_text": carried.prompt_text}
                    if (continuation_choice["applied"] and carried is not None) else None
                ),
                "note": CONTINUATION_NOTE,
            },
            "postprocess": False,
            "reference": REFERENCE,
            "timing": {
                # ``encode_ms`` is the encoder subprocess (``None`` when the vector came
                # out of the cache, in which case nothing was encoded); ``total_ms`` is
                # the whole request as the server saw it, excluding only the HTTP
                # framing. The two are measured at different points on purpose: the
                # page can show "0.9 s, of which 0.4 s was the encoder" without
                # subtracting anything itself.
                "encode_ms": embedding_source.get("encode_ms"),
                "generate_ms": generate_ms,
                "total_ms": (time.perf_counter() - request_started) * 1000.0,
                "embedding_source": embedding_source.get("kind"),
                "cache_hit": embedding_source.get("kind") == "cache",
                "router_ms": router_block.get("latency_ms"),
            },
            # Both blocks travel on every planned clip: ``router`` is what was read
            # (or the name of the reason nothing was), ``decision`` is what the viewer
            # does about it, including the bubble sentence and the source it names.
            "router": router_block,
            "decision": decision,
            "served_by": f"{HOST} scripts/ardy_viewer.py",
        }
        # The next request's ``context.current_action``. Recorded from the clip that
        # was actually drawn, which is the only honest answer to "what is happening".
        self.last_action = entry.get("text")
        if include_frames:
            payload["frames_330_b64"] = encode_float32(np.asarray(generated, dtype=np.float32)[0])
        return payload

    # -- continuation ----------------------------------------------------
    def continuation_supported(self) -> bool:
        """Whether this runner can generate *from* a state instead of from noise.

        ``scripts/ardy_viewer.py`` builds :class:`~ardy_mlx.sampling.pipeline.MlxRunner`,
        whose ``batch`` is the assembled ``ArdyBatch``. A stand-in runner (the
        viewer's tests, and anything else that only implements ``generate``) has no
        batch, and the page is told so by name rather than being handed a clip that
        silently restarted.
        """
        batch = getattr(self.runner, "batch", None)
        return all(
            hasattr(batch, attribute)
            for attribute in ("denoiser", "tokenizer", "motion_rep", "gen_horizon_len")
        )

    def _generate_continued(
        self,
        *,
        runner,
        embedding,
        previous: Optional[ContinuationState],
        frames_wanted: int,
        num_steps: int,
        weights,
        seed: Optional[int],
        noise_fn,
    ):
        """One clip generated from the clip on screen (never from frame 0).

        Two things are carried over, and they are carried separately because they
        live in different places:

        * **the pose and the recentred state** -- the previous clip's trailing
          frames become the window's clean history, exactly as window *n+1* of a
          multi-window run would have been handed window *n*'s output;
        * **the world position and the heading** -- the model's own loop only ever
          accumulates translation *within* one call, so the accumulated root
          translation comes from the frame the page is showing (``previous.roots``)
          and the heading is recomputed by the loop from the seed's first frame.

        The served clip opens on the previous clip's last frame, carried over
        verbatim: the model's first new frame is one frame *after* it, so
        ``[carried, generated...]`` is the gapless continuation and the clip keeps
        the requested length.
        """
        from ardy_mlx.model.ardy import make_step_fn
        from ardy_mlx.sampling.autoregressive import ArdyAutoregressiveConfig, ArdySampler
        from ardy_mlx.sampling.window import shift_root_xy

        if previous is None:  # pragma: no cover - the decision never lets this through
            raise ViewerError(
                "no_continuation_state",
                "a continuation was requested but nothing is on screen to continue from",
            )
        batch = runner.batch
        motion_rep = batch.motion_rep
        history = min(continuation_history_frames(self.layout), int(previous.frames.shape[1]))
        history -= history % NUM_FRAMES_PER_TOKEN
        if history < NUM_FRAMES_PER_TOKEN:
            raise ViewerError(
                "continuation_state_too_short",
                f"the clip on screen holds {previous.frames.shape[1]} frame(s); a window needs "
                f"at least {NUM_FRAMES_PER_TOKEN}",
                status=500,
            )
        tail = np.asarray(previous.frames[:, -history:, :], dtype=np.float32)
        # The window logic expects the newest frame to be the origin the next one is
        # relative to, so the seed is the tail re-centred on its own last frame. The
        # world position is put back from the same frame at the end. **Everything
        # here is in the normalised space the frames are carried in** -- mixing in a
        # metric root would offset the new clip by the stats' mean (and is exactly
        # what ``CONTINUATION_SEAM_LIMIT_M`` below is there to catch).
        anchor = np.asarray(previous.frames[0, -1, [0, 2]], dtype=np.float32)
        seed_frames = tail.copy()
        seed_frames[0, :, 0] -= anchor[0]
        seed_frames[0, :, 2] -= anchor[1]

        config = ArdyAutoregressiveConfig(
            layout=self.layout,
            num_frames=history + frames_wanted,
            history_frames=history,
            num_steps=num_steps,
            cfg_weight=weights,
            seed=seed,
            text_embeddings=embedding.embeddings,
            text_mask=embedding.mask,
        )
        step_fn = make_step_fn(
            batch,
            embedding.embeddings,
            embedding.mask,
            num_steps=num_steps,
            cfg_weight=weights,
            crop_history_length=history,
        )
        if noise_fn is not None:
            step_fn = _with_injected_noise(step_fn, noise_fn)
        sampler = ArdySampler(config, step_fn, initial_history_frames=mx.array(seed_frames))
        result = sampler.run()

        # ``result.translated()`` is the new motion in the seed's own frame; adding
        # the previous clip's last root puts it back where the page is looking.
        shift = mx.array(np.array([[anchor[0], 0.0, anchor[1]]], dtype=np.float32))
        world_norm = shift_root_xy(result.translated(), shift)
        carried_norm = mx.array(tail[:, -1:, :])
        # The stats are per feature channel, so the root is read *after* the whole
        # frame is un-normalised rather than by slicing first.
        seam_gap_m = float(
            np.abs(
                np.asarray(motion_rep.unnormalize(world_norm[:, :1, :]), dtype=np.float32)[0, 0, 0:3]
                - np.asarray(motion_rep.unnormalize(carried_norm), dtype=np.float32)[0, 0, 0:3]
            ).max()
        )
        if seam_gap_m > CONTINUATION_SEAM_LIMIT_M:
            # The first generated frame is 50 ms after the frame the page is
            # showing. No motion in this checkpoint crosses a quarter of a metre in
            # one frame (that is 5 m/s); a seam beyond it is a bug in the carry-over
            # -- a wrong anchor, a mixed unit, a lost translation -- and shipping it
            # would be exactly the silent "the character teleported" this feature
            # exists to prevent.
            raise ViewerError(
                "continuation_seam_too_far",
                f"the first generated frame sits {seam_gap_m * 1000:.1f} mm from the frame on "
                f"screen (the limit is {CONTINUATION_SEAM_LIMIT_M * 1000:.0f} mm for one frame)",
                detail=(
                    "the carried state, not the model, is at fault: the world position or the "
                    "re-centred history did not survive the request boundary. Nothing is served "
                    "rather than serving a clip that jumps."
                ),
                status=500,
            )
        served_norm = mx.concatenate(
            [carried_norm, world_norm[:, : max(frames_wanted - 1, 0), :]], axis=1
        )
        generated = motion_rep.unnormalize(served_norm)
        run_info = {
            "offsets": [int(value) for value in result.offsets],
            "history_frames": history,
        }
        return generated, np.asarray(served_norm, dtype=np.float32), run_info

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
                        reset=_opt_bool(params.get("reset")),
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
        "--llm2vec-repo",
        default=os.environ.get("ARDY_LLM2VEC_REPO", str(DEFAULT_LLM2VEC_REPO)),
        help=(
            "the sibling llm2vec-mlx checkout to encode with, run through its own .venv "
            f"(default {DEFAULT_LLM2VEC_REPO}; never imported, only subprocessed)"
        ),
    )
    parser.add_argument(
        "--no-local-encoder",
        action="store_true",
        help=(
            "disable local text encoding: every prompt must then have a cached embedding, "
            "exactly as v1 behaved"
        ),
    )
    parser.add_argument(
        "--encoder-bits",
        type=int,
        default=4,
        choices=(4, 8),
        help=(
            "quantization of the local encoder's weights (default 4). The full precision "
            "plan -- 4-bit is group 32 with the last two layers and embed_tokens at 8 bits "
            "-- is read from the artifact manifest and recorded in every sidecar"
        ),
    )
    parser.add_argument(
        "--encoder-weights-dir",
        default=None,
        help="override the encoder's weights directory (default <repo>/weights)",
    )
    parser.add_argument(
        "--encoder-interpreter",
        default=None,
        help="override the interpreter used to run the encoder (default <repo>/.venv/bin/python)",
    )
    parser.add_argument(
        "--encoder-timeout",
        type=float,
        default=600.0,
        metavar="SECONDS",
        help="give up on one encoder subprocess after this long (default 600)",
    )
    parser.add_argument(
        "--no-router",
        action="store_true",
        help=(
            "turn semantic routing off entirely: no judgment is made and nothing leaves "
            "this machine (by default the router runs when TYPESAFE_API_KEY or JEV_API_KEY "
            "is in this server's environment)"
        ),
    )
    parser.add_argument(
        "--router-deadline-ms",
        type=int,
        default=DEFAULT_ROUTER_DEADLINE_MS,
        metavar="MS",
        help=(
            "how long a plan may wait for a judgment before degrading to the plain vector "
            f"route, by name (default {DEFAULT_ROUTER_DEADLINE_MS}; the first call of a "
            "process pays Jev's cold start, about 1.7 s)"
        ),
    )
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

    encoder = None
    if not args.no_local_encoder:
        encoder = LocalEncoder(
            repo=Path(args.llm2vec_repo),
            bits=args.encoder_bits,
            weights_dir=Path(args.encoder_weights_dir) if args.encoder_weights_dir else None,
            interpreter=Path(args.encoder_interpreter) if args.encoder_interpreter else None,
            timeout_s=args.encoder_timeout,
        )
        status = encoder.status(refresh=True)
        if status.available:
            print(
                f"[ardy_viewer] local encoder: {status.encoder_id} "
                f"({status.quantization}) via {status.interpreter}",
                flush=True,
            )
        else:
            # Not fatal, and said plainly: the viewer still serves every cached prompt,
            # and an uncached one now fails with local_encoder_unavailable instead of
            # the old no_cached_embedding_for_prompt. Either way it never substitutes.
            print(f"[ardy_viewer] WARNING: local encoder unavailable: {status.reason}", flush=True)

    app = ViewerApp(
        model=args.model,
        device=args.device,
        embeddings_dir=Path(args.embeddings_dir),
        checkpoint_dir=Path(args.checkpoint_dir) if args.checkpoint_dir else None,
        allow_cpu=args.allow_cpu,
        local_encoder=encoder,
        router_enabled=not args.no_router,
        router_deadline_ms=args.router_deadline_ms,
    )
    # Say which of the three router conditions this process is in, before the first
    # request: a viewer that quietly stopped asking Jev would otherwise look identical
    # to one that asked and got no answer.
    print(f"[ardy_viewer] router: {app.router.status()['state']} "
          f"({app.router.status()['reason'] or app.router.router_id})", flush=True)

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
