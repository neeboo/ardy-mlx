"""Local text encoding, across a **file interface** to ``llm2vec-mlx``.

ARDY consumes a precomputed ``[1, L, 4096]`` text embedding. The encoder that
produces it — LLM2Vec + Meta-Llama-3-8B (MNTP + supervised LoRA) on MLX — lives in
its own repository, ``~/dev/llm2vec-mlx``. This module does **not** import it, copy
it, or vendor it. It runs the other repository's own entry point in the other
repository's own virtualenv ::

    <llm2vec-mlx>/.venv/bin/python <llm2vec-mlx>/scripts/encode.py \\
        --text "<prompt>" --bits 4 --out <tmp>/<prompt-hash>.npy --quiet

and reads back the two artefacts that command writes: the ``.npy`` vector and its
``.json`` sidecar. That is the whole coupling — a subprocess and two files.

Why a subprocess rather than an import
-------------------------------------
Three reasons, in order of how much they matter:

1. **The two repositories are separate.** ``ardy-mlx`` must not acquire a
   ``sys.path`` dependency on a sibling checkout, and ``llm2vec-mlx`` must not grow
   an API that only exists to serve this viewer. A file interface keeps the seam
   visible: if the other repository changes, this adapter fails by name instead of
   breaking in a way nobody can trace.
2. **The encoder is a second, much larger model.** It must not share a process (or
   an MLX GPU stream, or a footprint) with the ARDY checkpoint. A subprocess makes
   that separation structural rather than careful.
3. **It is honest about the cost.** The cold start of the encoder process is a
   number the viewer reports (``encode_ms``), not something hidden behind an
   in-process call.

Failures are **named and never substituted**
-------------------------------------------
Every way this can go wrong leaves through a :class:`LocalEncoderError` that says
what happened, in the vocabulary the viewer already uses:

``local_encoder_unavailable``
    The interpreter, the script or the weight manifest is not on disk, so no
    encoding is possible at all. This is decided *before* the subprocess is
    spawned (see :meth:`LocalEncoder.probe`).
``local_encoder_failed``
    The encoder ran and refused: a non-zero exit, a missing vector, an unreadable
    sidecar, a wrong shape, a prompt over the model's ``max_length``. The original
    ``stderr`` tail is carried in ``detail``.

Both are *soft* in one precise sense: :func:`encode` returns them rather than
raising, because the caller may legitimately want to record the failure and carry
on. What is **not** available is a silent fall back to a different prompt's vector,
or to a vector from a different encoder. There is no code path in this module that
returns a vector it did not just read out of the file the subprocess wrote.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

__all__ = [
    "LocalEncoderError",
    "EncodeResult",
    "EncoderStatus",
    "LocalEncoder",
    "EncoderConfigError",
    "DEFAULT_LLM2VEC_REPO",
    "DEFAULT_BITS",
    "SUPPORTED_BITS",
    "REFERENCE_ENCODER_ID",
    "LOCAL_ENCODER_SUBDIR",
    "quantization_name",
    "encoder_id_for",
    "interpreter_candidates",
]

#: The sibling checkout this adapter shells out to. Overridable by
#: ``--llm2vec-repo`` / ``ARDY_LLM2VEC_REPO``; never imported.
DEFAULT_LLM2VEC_REPO = Path.home() / "dev" / "llm2vec-mlx"

#: The bit width the viewer encodes at. 4-bit is the artifact whose latency and
#: footprint fit inside the ARDY process; 8-bit is available on request, and both
#: are named in full in every sidecar (see :func:`quantization_name`).
DEFAULT_BITS = 4
SUPPORTED_BITS = (4, 8)

#: ``encoder`` field of a **reference** vector's sidecar (``llm2vec_mlx``'s
#: ``recipe.ENCODER_REFERENCE``). A sidecar with no ``encoder`` field at all is also
#: read as a reference vector: the historical cache files predate the field.
REFERENCE_ENCODER_ID = "reference-llm2vec-bf16"

#: Sub-directory of the viewer's embedding cache that holds locally-encoded vectors,
#: matching ``llm2vec_mlx.encoder.cache.MLX_SUBDIR``. The split is what makes one
#: prompt able to hold both a reference and a local vector without either
#: overwriting the other.
LOCAL_ENCODER_SUBDIR = "mlx"


class EncoderConfigError(ValueError):
    """The adapter was configured with something it cannot mean."""


# --------------------------------------------------------------------------- #
# results and failures
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LocalEncoderError:
    """A named, non-fatal failure to produce a vector locally.

    ``soft`` is ``True`` for a failure the caller should cache and reuse (retrying
    would hit the same wall: an over-length prompt, a missing manifest) and ``False``
    for one that may clear on its own (a timeout, an out-of-disk write). The viewer
    caches only the soft ones.
    """

    name: str
    message: str
    detail: Optional[str] = None
    soft: bool = True
    exit_code: Optional[int] = None


@dataclass
class EncodeResult:
    """One ratio of the file interface: vector ``path``, plus what produced it."""

    prompt: str
    prompt_hash: str
    encoder_id: str
    vector_path: Path
    embedding_sha256: str
    shape: List[int]
    encode_ms: float
    quantization: Optional[str]
    sidecar: Dict[str, Any] = field(default_factory=dict)

    def as_sidecar(self) -> Dict[str, Any]:
        """The provenance record written next to the cached vector."""
        record: Dict[str, Any] = {
            "prompt": self.prompt,
            "prompt_hash": self.prompt_hash,
            "encoder_id": self.encoder_id,
            "embedding_sha256": self.embedding_sha256,
            "shape": list(self.shape),
            "dtype": "float32",
            "num_tokens": int(self.shape[1]) if len(self.shape) >= 2 else None,
            "quantization": self.quantization,
            "local": True,
            "encoded_by": "llm2vec-mlx/scripts/encode.py (subprocess)",
        }
        record.update(self.sidecar)
        # The cache key is this viewer's, so it wins over the sibling repository's
        # shorter spelling (``mlx-llm2vec-4bit``): a cached file has to be addressable
        # by the key the viewer looked it up with, or a rescan would file the vector
        # under a name nothing ever asks for and re-encode the prompt every time.
        record["encoder_id"] = self.encoder_id
        record["encoder"] = self.sidecar.get("encoder", record.get("encoder"))
        return record


@dataclass
class EncoderStatus:
    """``/api/health``'s view of the local encoder. Never raises."""

    enabled: bool
    available: bool
    state: str
    reason: Optional[str]
    encoder_id: Optional[str]
    quantization: Optional[str]
    bits: Optional[int]
    repo: Optional[str]
    interpreter: Optional[str]
    encode_script: Optional[str]
    weights_dir: Optional[str]
    manifest: Optional[str]
    probe_ms: Optional[float]
    checks: Dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "enabled": self.enabled,
            "available": self.available,
            "state": self.state,
            "reason": self.reason,
            "encoder_id": self.encoder_id,
            "quantization": self.quantization,
            "bits": self.bits,
            "repo": self.repo,
            "interpreter": self.interpreter,
            "encode_script": self.encode_script,
            "weights_dir": self.weights_dir,
            "manifest": self.manifest,
            "probe_ms": self.probe_ms,
            "checks": dict(self.checks),
            "note": (
                "a subprocess of the sibling checkout llm2vec-mlx, reached through its "
                ".venv and scripts/encode.py; nothing is imported, so the two repositories "
                "stay decoupled"
            ),
        }


# --------------------------------------------------------------------------- #
# naming the quantization in full
# --------------------------------------------------------------------------- #
_LAYER_TENSOR = re.compile(r"^layers\.(\d+)\.")


def quantization_name(manifest: Optional[dict]) -> Optional[str]:
    """The **full** precision description of an encoder artifact, from its manifest.

    ``"4-bit"`` is not enough to identify the artifact that clears the acceptance
    floor. The 4-bit recipe is 4-bit affine **at group 32, with the last two
    transformer layers and the token-embedding table held at 8 bits**; plain 4-bit at
    group 32 measures a minimum cosine of 0.9828 against the bf16 reference and does
    **not** clear the frozen 0.985 floor. The pair of numbers therefore names
    different model behaviour, and a sidecar that records only the nominal width
    cannot be compared with one that records the plan.

    This reads the plan out of the artifact's own manifest
    (``recipe.quantization_plan`` + ``quantization.mixed_precision``) and renders it
    as, for example::

        4-bit group 32, last-2 layers + embed_tokens at 8-bit
        8-bit group 64, no exceptions

    ``<bits>`` is the artifact's **nominal** width; the escalated widths are read from
    the plan, so the exception statement never repeats a width the nominal one already
    implies. The exact tensor list stays in the manifest and in every sidecar for
    anyone who wants it — this string is the summary, not the record.

    Returns ``None`` when the manifest cannot supply the nominal width and group size:
    an unnamed artifact is an explicit unknown, never a guess.
    """
    if not manifest:
        return None
    bits = manifest.get("bits")
    group = manifest.get("group_size")
    if bits is None or group is None:
        return None
    quantization = manifest.get("quantization") or {}
    mixed = quantization.get("mixed_precision") or {}
    plan = (manifest.get("recipe") or {}).get("quantization_plan") or {}

    base = f"{int(bits)}-bit group {int(group)}"
    if not mixed:
        return f"{base}, no exceptions"

    # The token embedding is escalated or it is not; either way it is one clause.
    embed_raised = "embed_tokens.weight" in mixed
    layer_extra: Dict[int, int] = {}
    for name, width in mixed.items():
        match = _LAYER_TENSOR.match(name)
        if match:
            layer_extra[int(match.group(1))] = int(width)

    # The layer block, expressed as the plan states it: "everything from layer N up".
    # The count comes from the plan's floor rather than from the number of tensor
    # names, so a partially-escalated layer cannot read as a smaller block.
    parts: List[str] = []
    if layer_extra:
        top = max(layer_extra)
        floor = int(plan.get("__layers_from__", min(layer_extra)))
        span = top - floor + 1
        width = plan.get("__layers_from_width__")
        if width is None:
            widths = set(layer_extra.values())
            width = widths.pop() if len(widths) == 1 else None
        parts.append(f"last-{span} layers")
    if embed_raised:
        parts.append("embed_tokens")

    # One width, stated once, at the end: "…, last-2 layers + embed_tokens at 8-bit".
    widths = set(layer_extra.values())
    if embed_raised:
        widths.add(int(plan.get("embed_tokens", int(bits))))
    escalated = sorted(width for width in widths if width != int(bits))
    if len(escalated) == 1:
        return f"{base}, " + " + ".join(parts) + f" at {escalated[0]}-bit"
    if not escalated:
        # Every exception is at the nominal width already; naming a width here would
        # suggest a difference that does not exist.
        return f"{base}, " + " + ".join(parts) + " (same width)"
    return f"{base}, " + " + ".join(parts) + " at mixed widths"


def encoder_id_for(encoder: str, quantized: Optional[str]) -> str:
    """The cache-key half that names the encoder and its exact precision.

    Two vectors may only share a cache slot when this string is identical. The
    ``quantized`` half is the full :func:`quantization_name`, lower-cased and
    punctuation-collapsed, so a sidecar records both a stable id and, separately, the
    human-readable quantization.
    """
    if not quantized:
        return encoder
    slug = re.sub(r"[^a-z0-9]+", "-", quantized.lower()).strip("-")
    return f"{encoder}-{slug}"


# --------------------------------------------------------------------------- #
# finding the interpreter
# --------------------------------------------------------------------------- #
def interpreter_candidates(repo: Path) -> List[Path]:
    """The interpreters this adapter will try, in order, inside ``repo``.

    **Only the sibling checkout's own virtualenv.** That is the environment the
    encoder was built and accepted in; any other interpreter is a different,
    unmeasured configuration. In particular this viewer's own ``sys.executable`` is
    *not* a candidate: it is a different virtualenv, on a different dependency set,
    and reaching for it would make "the encoder is unavailable" quietly become "the
    encoder ran under the wrong python". Both spellings of the venv entry point are
    tried so an environment that only ships one still works, and every candidate is
    reported in the failure detail.
    """
    venv = Path(repo) / ".venv"
    return [venv / "bin" / "python", venv / "bin" / "python3"]


_BITS_IN_ID = re.compile(r"(\d+)\s*-?\s*bit")


def _bits_from_encoder_id(encoder_id: str) -> Optional[int]:
    """The bit width a sibling encoder id names, e.g. ``mlx-llm2vec-4bit`` -> 4.

    ``None`` when the id does not name one; the caller then falls back to the other
    recorded fields rather than guessing.
    """
    match = _BITS_IN_ID.search(encoder_id)
    return int(match.group(1)) if match else None


# --------------------------------------------------------------------------- #
# the adapter
# --------------------------------------------------------------------------- #
class LocalEncoder:
    """Runs ``llm2vec-mlx``'s ``scripts/encode.py`` and reads back its two files.

    Constructed with paths and nothing else; :meth:`probe` is cheap and safe to call
    at startup (no weights are loaded), :meth:`encode` spawns one subprocess per
    prompt. The object holds no model and no state between calls, so a failure cannot
    poison the next request.
    """

    def __init__(
        self,
        repo: Path = DEFAULT_LLM2VEC_REPO,
        *,
        bits: int = DEFAULT_BITS,
        weights_dir: Optional[Path] = None,
        interpreter: Optional[Path] = None,
        timeout_s: float = 600.0,
        encoder: str = "mlx-llm2vec",
    ):
        if int(bits) not in SUPPORTED_BITS:
            raise EncoderConfigError(
                f"bits must be one of {SUPPORTED_BITS}, got {bits!r}"
            )
        self.repo = Path(repo).expanduser()
        self.bits = int(bits)
        self.weights_dir = Path(weights_dir).expanduser() if weights_dir else self.repo / "weights"
        self.interpreter = Path(interpreter).expanduser() if interpreter else None
        self.timeout_s = float(timeout_s)
        self.encoder = encoder
        self._status: Optional[EncoderStatus] = None

    # -- paths -----------------------------------------------------------
    @property
    def encode_script(self) -> Path:
        return self.repo / "scripts" / "encode.py"

    @property
    def manifest_path(self) -> Path:
        return self.weights_dir / f"manifest-{self.bits}bit.json"

    def _resolve_interpreter(self) -> Optional[Path]:
        if self.interpreter is not None:
            return self.interpreter if self.interpreter.is_file() else None
        for candidate in interpreter_candidates(self.repo):
            if candidate.is_file():
                return candidate
        return None

    # -- manifest --------------------------------------------------------
    def read_manifest(self) -> Optional[dict]:
        try:
            return json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None

    def quantization(self) -> Optional[str]:
        return quantization_name(self.read_manifest())

    def group_size(self) -> Optional[int]:
        manifest = self.read_manifest() or {}
        value = manifest.get("group_size")
        return int(value) if value is not None else None

    def encoder_id(self) -> str:
        """``mlx-llm2vec`` + the full quantization, i.e. the cache key's second half."""
        manifest = self.read_manifest()
        if manifest and isinstance(manifest.get("encoder"), str):
            base = manifest["encoder"]
        else:
            base = self.encoder
        return encoder_id_for(base, quantization_name(manifest))

    # -- probing ---------------------------------------------------------
    def probe(self, *, refresh: bool = False) -> EncoderStatus:
        """Can this encoder run at all? Filesystem only — no weights, no subprocess.

        Deliberately *not* a trial encode: loading the encoder costs seconds and
        gigabytes, and doing that at viewer startup to answer a health check would
        make the page slower to be more confident about something the first real
        request discovers anyway. So the states are exact about what is known:

        ``ready``
            interpreter, ``scripts/encode.py`` and the ``<bits>`` manifest are all
            present and readable, so an encode *should* run.
        ``disabled``
            the adapter was not configured (no encoder wired in at all).
        ``unavailable``
            one of those three is missing; ``reason`` names which, and ``checks``
            carries every candidate that was tried.
        """
        started = time.perf_counter()
        checks: Dict[str, Any] = {}
        if not self.repo.is_dir():
            checks["repo"] = {"path": str(self.repo), "exists": False}
            return self._finish_status(
                enabled=True, available=False, state="unavailable",
                reason=f"the llm2vec-mlx checkout is not at {self.repo}",
                checks=checks, started=started,
            )
        checks["repo"] = {"path": str(self.repo), "exists": True}

        script = self.encode_script
        checks["encode_script"] = {"path": str(script), "exists": script.is_file()}
        interpreter = self._resolve_interpreter()
        candidates = (
            [self.interpreter] if self.interpreter is not None
            else interpreter_candidates(self.repo)
        )
        checks["interpreter"] = {
            "path": str(interpreter) if interpreter else None,
            "candidates": [
                {"path": str(c), "exists": c.is_file()} for c in candidates if c is not None
            ],
        }
        manifest = self.manifest_path
        manifest_data = self.read_manifest()
        checks["manifest"] = {
            "path": str(manifest),
            "exists": manifest.is_file(),
            "readable": manifest_data is not None,
        }

        missing: List[str] = []
        if not script.is_file():
            missing.append(f"scripts/encode.py ({script})")
        if interpreter is None:
            missing.append(
                "an interpreter (tried: "
                + ", ".join(str(c) for c in candidates if c is not None)
                + ")"
            )
        if manifest_data is None:
            missing.append(
                f"the {self.bits}-bit weight manifest ({manifest}); "
                f"run llm2vec-mlx's scripts/build_weights.py first"
            )
        if missing:
            return self._finish_status(
                enabled=True, available=False, state="unavailable",
                reason="the local text encoder is not usable: missing " + "; ".join(missing),
                checks=checks, started=started,
            )
        return self._finish_status(
            enabled=True, available=True, state="ready",
            reason=None, checks=checks, started=started, manifest=manifest_data,
        )

    def _finish_status(
        self,
        *,
        enabled: bool,
        available: bool,
        state: str,
        reason: Optional[str],
        checks: Dict[str, Any],
        started: float,
        manifest: Optional[dict] = None,
    ) -> EncoderStatus:
        manifest = manifest if manifest is not None else self.read_manifest()
        status = EncoderStatus(
            enabled=enabled,
            available=available,
            state=state,
            reason=reason,
            encoder_id=encoder_id_for(
                (manifest or {}).get("encoder", self.encoder) if manifest else self.encoder,
                quantization_name(manifest),
            ),
            quantization=quantization_name(manifest),
            bits=self.bits,
            repo=str(self.repo),
            interpreter=str(self._resolve_interpreter()) if self._resolve_interpreter() else None,
            encode_script=str(self.encode_script),
            weights_dir=str(self.weights_dir),
            manifest=str(self.manifest_path),
            probe_ms=(time.perf_counter() - started) * 1000.0,
            checks=checks,
        )
        self._status = status
        return status

    def status(self, *, refresh: bool = False) -> EncoderStatus:
        if self._status is None or refresh:
            return self.probe()
        return self._status

    # -- encoding --------------------------------------------------------
    def encode(
        self,
        prompt: str,
        *,
        prompt_hash: str,
        out_dir: Path,
        timeout_s: Optional[float] = None,
    ) -> "EncodeResult | LocalEncoderError":
        """Encode one prompt into ``out_dir/<prompt-hash>.npy`` (+ sidecar).

        Returns an :class:`EncodeResult` or a :class:`LocalEncoderError`; it never
        raises for an encoder-side problem and it never returns a vector it did not
        read from the file this call's subprocess wrote. ``out_dir`` is expected to be
        a private staging directory — the caller installs the result into the cache
        after validating it, so a failed encode cannot leave a half-written entry
        where a later request would find it.
        """
        status = self.status()
        if not status.available:
            return LocalEncoderError(
                "local_encoder_unavailable",
                status.reason or "the local text encoder is not usable",
                detail=json.dumps(status.checks, indent=2, sort_keys=True),
            )

        interpreter = self._resolve_interpreter()
        if interpreter is None:  # pragma: no cover - probe() already refused
            return LocalEncoderError(
                "local_encoder_unavailable",
                f"no interpreter under {self.repo}/.venv",
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{prompt_hash}.npy"
        command = [
            str(interpreter),
            str(self.encode_script),
            "--text", prompt,
            "--bits", str(self.bits),
            "--out", str(target),
            "--weights-dir", str(self.weights_dir),
            # --quiet suppresses the progress bar and the per-prompt summary; the
            # vector and its sidecar are unaffected, and stderr stays useful.
            "--quiet",
        ]

        started = time.perf_counter()
        try:
            completed = subprocess.run(
                command,
                cwd=str(self.repo),
                capture_output=True,
                text=True,
                timeout=float(timeout_s or self.timeout_s),
                # A closed stdin stops the other process ever blocking on a prompt.
                stdin=subprocess.DEVNULL,
                env=self._child_env(),
                check=False,
            )
        except subprocess.TimeoutExpired:
            return LocalEncoderError(
                "local_encoder_failed",
                f"the local encoder did not finish within {timeout_s or self.timeout_s:g} s",
                detail=f"command: {' '.join(command)}",
                soft=False,
            )
        except OSError as exc:
            return LocalEncoderError(
                "local_encoder_failed",
                f"could not start the local encoder: {type(exc).__name__}: {exc}",
                detail=f"command: {' '.join(command)}",
                soft=False,
            )
        encode_ms = (time.perf_counter() - started) * 1000.0

        if completed.returncode != 0:
            return LocalEncoderError(
                "local_encoder_failed",
                (
                    f"the local encoder exited {completed.returncode} for prompt hash "
                    f"{prompt_hash}"
                ),
                detail=_stderr_tail(completed.stderr, completed.stdout),
                soft=_soft_exit(completed.returncode),
                exit_code=completed.returncode,
            )

        return self._read_result(
            prompt=prompt,
            prompt_hash=prompt_hash,
            vector_path=target,
            encode_ms=encode_ms,
        )

    def _read_result(
        self,
        *,
        prompt: str,
        prompt_hash: str,
        vector_path: Path,
        encode_ms: float,
    ) -> "EncodeResult | LocalEncoderError":
        """Validate what the subprocess left behind, or say exactly what is wrong.

        Every check here exists because the alternative is a plausible-looking vector
        of the wrong shape or a sidecar describing a different artifact. The shape
        check is the one that matters most: ARDY's conditioning input is
        ``[1, L, 4096]`` and a ``(4096,)`` vector would broadcast into something the
        model accepts and nobody can explain.
        """
        if not vector_path.is_file():
            return LocalEncoderError(
                "local_encoder_failed",
                f"the local encoder reported success but wrote no vector at {vector_path}",
            )
        try:
            array = np.load(vector_path, allow_pickle=False)
        except (OSError, ValueError) as exc:
            return LocalEncoderError(
                "local_encoder_failed",
                f"the local encoder's vector at {vector_path} is unreadable: {exc}",
            )
        if array.ndim != 3 or array.shape[0] != 1 or array.shape[-1] != 4096:
            return LocalEncoderError(
                "local_encoder_failed",
                (
                    f"the local encoder produced shape {tuple(array.shape)}; ARDY needs "
                    f"(1, L, 4096)"
                ),
                detail=f"vector: {vector_path}",
            )
        if array.dtype != np.float32:
            return LocalEncoderError(
                "local_encoder_failed",
                f"the local encoder produced {array.dtype}, not float32",
                detail=f"vector: {vector_path}",
                soft=False,
            )
        if array.size and not np.isfinite(array).all():
            return LocalEncoderError(
                "local_encoder_failed",
                f"the local encoder's vector at {vector_path} contains non-finite values",
            )

        sidecar: Dict[str, Any] = {}
        sidecar_path = vector_path.with_suffix(".json")
        if sidecar_path.is_file():
            try:
                loaded = json.loads(sidecar_path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    sidecar = loaded
            except (OSError, json.JSONDecodeError):
                sidecar = {}

        # The vector must have come from the artifact this adapter is configured for. A
        # mismatch means the subprocess used different weights than this object names —
        # a stale weights directory, say — and caching the result under this
        # encoder_id would be exactly the silent mixing the sidecar exists to prevent.
        #
        # The check is on what the *encoder* recorded it used, not on string equality
        # with this adapter's own id. The sibling repository's own id names the bit
        # width ("mlx-llm2vec-4bit") while this viewer's cache key spells out the whole
        # precision plan; both are correct, and demanding equality would reject the
        # real encoder for being more concise than the cache key.
        mismatch = self._sidecar_mismatch(sidecar, sidecar_path)
        if mismatch is not None:
            return mismatch

        from ardy_mlx.sampling.pipeline import sha256_file

        # The cache key this viewer attributes the vector to, and the human-readable
        # precision: this adapter's own summary, not a restatement of the sibling's
        # shorter id.
        expected = self.encoder_id()
        quantized = self.quantization()

        return EncodeResult(
            prompt=prompt,
            prompt_hash=prompt_hash,
            encoder_id=expected,
            vector_path=vector_path,
            embedding_sha256=sha256_file(vector_path),
            shape=[int(dim) for dim in array.shape],
            encode_ms=encode_ms,
            quantization=quantized,
            sidecar=sidecar,
        )

    def _sidecar_mismatch(self, sidecar: dict, sidecar_path: Path):
        """``None`` if the sidecar describes this adapter's artifact, else the refusal.

        The comparison is **semantic**, not a string equality against this viewer's own
        cache id. The two names are legitimately different spellings of the same
        artifact: the sibling repository writes ``mlx-llm2vec-4bit`` (the bit width)
        while the cache key spells out the whole precision plan
        (``mlx-llm2vec-4-bit-group-32-…``). Demanding equality would reject the real
        encoder for being more concise than the cache key, so each recorded field is
        checked against the thing it actually claims:

        * ``bits`` / ``group_size`` / ``mode`` — the parameters that define the artifact.
        * ``encoder`` — the family name (``mlx-llm2vec``); a different family is a
          different encoder, not a different precision of this one.
        * ``encoder_id`` — only its embedded bit width is compared, since that is the
          one part of the name both spellings share.
        * ``quantization`` — a human-readable summary that the sibling repository's
          sidecars do not carry, so a missing string is not evidence of anything.
        """
        if not sidecar:
            # No sidecar at all: the vector's bytes were checked and :meth:`install`
            # records its provenance. Silence here is a missing sidecar, not a
            # contradiction, and inventing a refusal would be its own kind of lie.
            return None
        expected = self.encoder_id()
        manifest = self.read_manifest() or {}
        reported_quantization = sidecar.get("quantization")
        configured_quantization = self.quantization()

        def refuse(message: str) -> LocalEncoderError:
            return LocalEncoderError(
                "local_encoder_failed", message, detail=f"sidecar: {sidecar_path}"
            )

        bits = sidecar.get("bits")
        if bits is not None and int(bits) != self.bits:
            return refuse(
                f"the local encoder produced a {int(bits)}-bit vector, but this viewer "
                f"is configured for {self.bits}-bit ({expected})"
            )
        group = sidecar.get("group_size")
        expected_group = manifest.get("group_size", self.group_size())
        if group is not None and expected_group is not None and int(group) != int(expected_group):
            return refuse(
                f"the local encoder used quantization group {int(group)}, but this viewer "
                f"is configured for group {int(expected_group)} ({expected})"
            )
        family = sidecar.get("encoder")
        if isinstance(family, str) and family and not expected.startswith(family):
            return refuse(
                f"the local encoder reported encoder {family!r}, but this viewer is "
                f"configured for {expected!r}"
            )
        reported_id = sidecar.get("encoder_id")
        if isinstance(reported_id, str) and reported_id:
            reported_bits = _bits_from_encoder_id(reported_id)
            if reported_bits is not None and reported_bits != self.bits:
                return refuse(
                    f"the local encoder reported encoder_id {reported_id!r} "
                    f"({reported_bits}-bit), but this viewer is configured for "
                    f"{self.bits}-bit ({expected})"
                )
        if (
            isinstance(reported_quantization, str)
            and reported_quantization
            and configured_quantization
            and reported_quantization != configured_quantization
        ):
            return refuse(
                f"the local encoder described its artifact as {reported_quantization!r}, "
                f"but this viewer is configured for {configured_quantization!r}"
            )
        return None

    # -- environment -----------------------------------------------------
    def _child_env(self) -> Dict[str, str]:
        """The subprocess environment: the parent's, minus MLX knobs that mislead.

        ``PYTHONPATH`` is dropped rather than forwarded. This repository's own
        ``src/`` is on it when the viewer runs from a checkout, and handing a sibling
        repository a path that contains a *different* ``llm2vec_mlx`` package would
        be a cross-repository import wearing a subprocess costume — precisely the
        coupling this module exists to avoid.
        """
        env = dict(os.environ)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONSTARTUP", None)
        return env


def _soft_exit(returncode: int) -> bool:
    """Exit code 3 is the encoder's own "I refuse" (see ``encode.py``'s docstring).

    A refusal is about the *input* — a missing artifact, a prompt over
    ``max_length`` — and will refuse again identically, so the viewer caches it. Any
    other non-zero code (a crash, a signal, a full disk) is treated as retryable.
    """
    return returncode == 3


def _stderr_tail(stderr: Optional[str], stdout: Optional[str], limit: int = 1200) -> str:
    """The last ``limit`` characters of the child's output, stderr first.

    This is deliberately the *raw* tail rather than a parsed message: the encoder's
    own words are the evidence, and summarising them here would put this repository
    in the business of paraphrasing a failure it does not own.
    """
    parts: List[str] = []
    if stderr and stderr.strip():
        parts.append(stderr.strip())
    if stdout and stdout.strip():
        parts.append(stdout.strip())
    joined = "\n".join(parts)
    if len(joined) > limit:
        return "…" + joined[-limit:]
    return joined
