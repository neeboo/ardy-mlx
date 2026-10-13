"""Shared fixtures for the viewer tests.

``tests/`` is not a package, and pytest's import modes disagree about whether
``conftest`` is importable by name, so nothing here is imported by the test files:
the fixtures are injected, and the one module-level name they all share
(``ardy_viewer``, i.e. ``scripts/ardy_viewer.py``) is imported normally by each file
that needs it. That matters more than it sounds -- ``monkeypatch.setattr`` on a
*different copy* of the viewer module than the one under test is a green test that
proves nothing, and this repository's rule is that a check has to be able to go red.

The synthetic frames below are **not** a stand-in for the model when a test needs
the model: anything asserting alignment skips unless the real checkpoint is on disk
(``tests/test_viewer_service.py``). What they give the API and HTTP tests is a
correctly shaped, finite feature block, so the real ``CoreSkeleton27`` FK and the
real payload encoder run without loading 765 MB of weights.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ardy_mlx.io import NUM_FRAMES_PER_TOKEN, PRESETS  # noqa: E402

#: The shipped Core preset, and a 40x cheaper layout with the same arithmetic.
CORE = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
TINY = CORE.__class__(nbjoints=27, fps=5, horizon=5, skeleton="CoreSkeleton27")

PROMPT = "A person walks forward."
WRONG_PROMPT = "A person runs backwards."


def load_viewer():
    """``scripts/ardy_viewer.py`` under a stable module name.

    ``scripts/`` is not a package either, so this is the by-path import every test
    file uses. Because the name is stable, two files that both call this get the
    **same** module object, and a ``monkeypatch`` in one is visible to the other.
    """
    name = "ardy_viewer"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "scripts" / "ardy_viewer.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


#: The shipped Core preset, and a 40x cheaper layout with the same arithmetic.
CORE = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
TINY = CORE.__class__(nbjoints=27, fps=5, horizon=5, skeleton="CoreSkeleton27")

PROMPT = "A person walks forward."
WRONG_PROMPT = "A person runs backwards."


def relink_viewer_module() -> None:
    """Point every test file's ``ardy_viewer`` name at the one live module.

    ``tests/`` is not a package, so each viewer test file loads
    ``scripts/ardy_viewer.py`` by path under the module name ``ardy_viewer``. The
    first one to do so registers it in ``sys.modules``; this function is then called
    from every file's module scope and rebinds *that file's* globals to the one
    object, so ``ardy_viewer.ViewerError`` is a single class no matter which file the
    collection order reached first. Two copies of the module compare unequal, and the
    resulting failures look like ordinary assertion errors rather than the import bug
    they are.
    """
    module = load_viewer()
    frame = sys._getframe(1)
    frame.f_globals.setdefault("ardy_viewer", module)
    frame.f_globals["ardy_viewer"] = module


def load_script(name: str, relative: str):
    """Import any other ``scripts/*.py`` by path (e.g. ``ardy_generate.py``)."""
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader, f"cannot load {relative}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_frames(num_frames: int, seed: int = 0) -> np.ndarray:
    """A deterministic ``(num_frames, 330)`` feature block, in physical units.

    The layout is the contract's: ``root_pos(3) | heading(2) | local_joints(78) |
    global_rot_data(162) | velocities(81) | foot_contacts(4)``. The rotation block
    is a small perturbation of the identity 6D basis so ``cont6d_to_matrix``
    produces a real rotation, and a foot contact is set on the first frame of every
    5-frame group so a "no contacts anywhere" bug cannot pass silently.
    """
    rng = np.random.default_rng(seed)
    frames = np.zeros((num_frames, CORE.motion_rep_dim), dtype=np.float32)
    t = np.arange(num_frames, dtype=np.float32)
    frames[:, 0] = 0.05 * t
    frames[:, 1] = 0.9 + 0.02 * np.sin(t)
    frames[:, 2] = 0.0
    frames[:, 3] = 1.0
    frames[:, 4] = 0.0
    frames[:, 5 : 5 + 26 * 3] = 0.01 * rng.standard_normal((num_frames, 26 * 3))
    rot = np.zeros((num_frames, 27, 6), dtype=np.float32)
    rot[..., 0] = 1.0
    rot[..., 4] = 1.0
    rot[..., 2] = 0.05 * np.sin(t)[:, None]
    rot[..., 5] = 0.05 * np.cos(t)[:, None]
    frames[:, 83 : 83 + 27 * 6] = rot.reshape(num_frames, -1)
    frames[:, -4:] = 0.0
    frames[::5, -4:] = 1.0
    return frames


class FakeRunner:
    """The parts of :class:`ardy_mlx.sampling.pipeline.MlxRunner` the viewer uses."""

    def __init__(self, layout=CORE, num_windows: int = 1, seed: int = 0):
        self.model = "ARDY-Core-RP-20FPS-Horizon40"
        self.layout = layout
        self.parameter_count = 1234
        self.load_seconds = {"tokenizer_s": 0.0, "denoiser_s": 0.0, "total_s": 0.0}
        self.device = "fake"
        self.embedding = None
        self.num_windows = num_windows
        self.seed = seed
        self.calls: list[dict] = []

    @property
    def noise_window_shape(self):
        return (1, self.layout.horizon // NUM_FRAMES_PER_TOKEN, self.layout.dim_token)

    def generate(self, *, num_frames, num_steps=10, cfg_weight=(2.0, 2.0), history_frames=None,
                 noise_fn=None, seed=None, windows=None):
        self.calls.append(
            {"num_frames": num_frames, "noise_fn": noise_fn, "windows": windows, "seed": seed}
        )
        windows = self.num_windows if windows is None else windows
        noise = None
        if noise_fn is not None:
            noise = np.asarray(noise_fn(0, self.noise_window_shape), dtype=np.float32)
        frames = synthetic_frames(num_frames, seed=self.seed)
        if noise is not None:
            # A deterministic, *tame* coupling: the noise nudges the root's y and z,
            # so two different x_T produce measurably different joints while the
            # numbers stay small enough for the FK to stay well-conditioned.
            level = float(np.abs(noise).mean())
            frames[:, 1] += 0.05 * level
            frames[:, 2] += 0.10 * level
        # ...and the same for the conditioning vector. Without this the fake would
        # ignore its own input, and a test that "the clip changed with the prompt"
        # could never fail -- a green test that proves nothing. The coupling is a
        # mean of the embedding's bytes, so it is deterministic and differs for any
        # two distinct vectors.
        if self.embedding is not None:
            vector = np.asarray(self.embedding.embeddings)
            condition = float(np.abs(vector).mean()) % 1.0
            frames[:, 1] += 0.02 * condition
            frames[:, 2] += 0.04 * condition
        return {
            "frames": frames[None, ...],
            "offsets": [0] + [self.layout.horizon * (i + 1) for i in range(windows)],
            "history_frames": (
                history_frames if history_frames is not None else self.layout.default_history_frames
            ),
            "windows": windows,
        }


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="session")
def viewer():
    """The viewer module, imported once for the whole session."""
    return load_viewer()


@pytest.fixture(scope="session")
def core():
    return CORE


@pytest.fixture(scope="session")
def tiny():
    return TINY


@pytest.fixture(scope="session")
def synthetic():
    """The synthetic-frame generator (see :func:`synthetic_frames`)."""
    return synthetic_frames


@pytest.fixture()
def embeddings_dir(tmp_path):
    """A cache directory holding one prompt, written the way the README says."""
    from ardy_mlx.sampling import pipeline

    directory = tmp_path / "embeddings"
    directory.mkdir()
    digest = pipeline.prompt_hash(PROMPT)
    rng = np.random.default_rng(7)
    np.save(directory / f"{digest}.npy", rng.standard_normal((1, 1, 4096)).astype(np.float32))
    (directory / f"{digest}.json").write_text(
        json.dumps({"prompt": PROMPT, "prompt_hash": digest}), encoding="utf-8"
    )
    return directory


@pytest.fixture()
def app(embeddings_dir, viewer):
    """A real :class:`ardy_viewer.ViewerApp` wired to a :class:`FakeRunner`.

    No checkpoint is loaded, so the tests that use this fixture exercise the real
    request handling, the real payload encoder and the real ``CoreSkeleton27`` FK
    without paying for 765 MB of weights. Anything that must prove alignment against
    the golden clips uses the real checkpoint and lives in
    ``tests/test_viewer_service.py``.
    """
    application = viewer.ViewerApp(embeddings_dir=embeddings_dir)
    application.embeddings.refresh()
    application.runner = FakeRunner()
    application.error = None
    application.boot_seconds = 0.01
    return application


# --------------------------------------------------------------------------- #
# a local text encoder that is not a text encoder
# --------------------------------------------------------------------------- #
#: The full quantization name the 4-bit encoder artifact reports. The tests assert on
#: this literal rather than on a helper's output because it is the string that has to
#: appear in a sidecar: "4-bit" alone names a different, failing artifact (plain 4-bit
#: at group 32 measures 0.9828 minimum cosine against the bf16 reference, below the
#: frozen 0.985 floor).
QUANTIZATION_4BIT = "4-bit group 32, last-2 layers + embed_tokens at 8-bit"
QUANTIZATION_8BIT = "8-bit group 64, no exceptions"
ENCODER_ID_4BIT = "mlx-llm2vec-4-bit-group-32-last-2-layers-embed-tokens-at-8-bit"
ENCODER_ID_8BIT = "mlx-llm2vec-8-bit-group-64-no-exceptions"


def deterministic_vector(prompt: str, *, salt: str = "", tokens: int = 1) -> np.ndarray:
    """A reproducible ``(1, L, 4096)`` vector **derived from the prompt**.

    Derived, not random: two prompts must give two different vectors, or a test that
    "the clip changed when the prompt changed" would pass for the wrong reason. The
    seed is the prompt's own hash, so the same prompt always yields the same bytes and
    ``salt`` distinguishes two encoders' vectors for one prompt.
    """
    from ardy_mlx.sampling import pipeline

    digest = hashlib.sha256(f"{salt}|{pipeline.prompt_hash(prompt)}".encode("utf-8")).digest()
    rng = np.random.default_rng(int.from_bytes(digest[:8], "big"))
    return rng.standard_normal((1, tokens, 4096)).astype(np.float32)


class FakeLocalEncoder:
    """A stand-in for :class:`ardy_mlx.sampling.local_encoder.LocalEncoder`.

    It implements the same two methods the viewer uses (``status`` and ``encode``)
    and nothing else, so the viewer's contract with a local encoder is exercised
    without loading 5 GB of weights or spawning a process. That contract is small on
    purpose: a status that can be *unavailable*, and an encode that returns either a
    result or a named ``LocalEncoderError``.

    What it deliberately does **not** do is reach the real encoder. The test that
    proves the subprocess and the file interface really work lives in
    ``tests/test_viewer_local_encoder.py`` and skips (with a reason) unless the
    sibling ``llm2vec-mlx`` checkout is on disk.
    """

    def __init__(
        self,
        *,
        available: bool = True,
        reason: Optional[str] = None,
        bits: int = 4,
        fail_with: Optional[str] = None,
        fail_message: str = "the local encoder refused this prompt",
        fail_soft: bool = True,
        fail_exit_code: Optional[int] = 3,
        encoder_salt: str = "mlx-4bit",
        tokens: int = 1,
        delay_s: float = 0.0,
    ):
        self._available = available
        self._reason = reason or "the local text encoder is not usable in this test"
        self.bits = bits
        self.fail_with = fail_with
        self.fail_message = fail_message
        self.fail_soft = fail_soft
        self.fail_exit_code = fail_exit_code
        self.encoder_salt = encoder_salt
        self.tokens = tokens
        self.delay_s = delay_s
        self.calls: list[dict] = []

    # -- the same two methods as the real adapter ------------------------
    def quantization(self) -> str:
        return QUANTIZATION_4BIT if self.bits == 4 else QUANTIZATION_8BIT

    def encoder_id(self) -> str:
        return ENCODER_ID_4BIT if self.bits == 4 else ENCODER_ID_8BIT

    def probe(self, *, refresh: bool = False):
        return self.status(refresh=refresh)

    def status(self, *, refresh: bool = False):
        from ardy_mlx.sampling.local_encoder import EncoderStatus

        return EncoderStatus(
            enabled=True,
            available=self._available,
            state="ready" if self._available else "unavailable",
            reason=None if self._available else self._reason,
            encoder_id=self.encoder_id() if self._available else None,
            quantization=self.quantization() if self._available else None,
            bits=self.bits,
            repo=f"/fake/llm2vec-mlx-{self.bits}bit",
            interpreter=f"/fake/llm2vec-mlx-{self.bits}bit/.venv/bin/python",
            encode_script=f"/fake/llm2vec-mlx-{self.bits}bit/scripts/encode.py",
            weights_dir=f"/fake/llm2vec-mlx-{self.bits}bit/weights",
            manifest=f"/fake/llm2vec-mlx-{self.bits}bit/weights/manifest-{self.bits}bit.json",
            probe_ms=0.01,
            checks={"fake": True},
        )

    def encode(self, prompt, *, prompt_hash, out_dir, timeout_s=None):
        from ardy_mlx.sampling.local_encoder import EncodeResult, LocalEncoderError
        from ardy_mlx.sampling.pipeline import sha256_file

        self.calls.append({"prompt": prompt, "prompt_hash": prompt_hash, "out_dir": str(out_dir)})
        if self.delay_s:
            time.sleep(self.delay_s)
        if not self._available:
            return LocalEncoderError(
                "local_encoder_unavailable",
                self._reason,
                detail="the fake encoder is switched off",
            )
        if self.fail_with:
            return LocalEncoderError(
                self.fail_with,
                self.fail_message,
                detail="fake stderr tail: the encoder refused this prompt",
                soft=self.fail_soft,
                exit_code=self.fail_exit_code,
            )

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        target = out_dir / f"{prompt_hash}.npy"
        vector = deterministic_vector(prompt, salt=self.encoder_salt, tokens=self.tokens)
        np.save(target, vector)
        sidecar = {
            "prompt": prompt,
            "prompt_hash": prompt_hash,
            "encoder": "mlx-llm2vec",
            "encoder_id": self.encoder_id(),
            "bits": self.bits,
            "group_size": 32 if self.bits == 4 else 64,
            "quantization": self.quantization(),
            "git_sha": "0" * 40,
        }
        target.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
        return EncodeResult(
            prompt=prompt,
            prompt_hash=prompt_hash,
            encoder_id=self.encoder_id(),
            vector_path=target,
            embedding_sha256=sha256_file(target),
            shape=[1, self.tokens, 4096],
            encode_ms=12.5,
            quantization=self.quantization(),
            sidecar=sidecar,
        )


@pytest.fixture()
def fake_encoder():
    """A working local encoder. Tests that need a *failing* one construct their own."""
    return FakeLocalEncoder()


@pytest.fixture()
def encoder_app(embeddings_dir, viewer, fake_encoder):
    """A :class:`ViewerApp` with a fake local encoder wired in.

    The embedding cache still holds exactly one prompt, so this is the fixture for
    "the cache missed and the encoder answered" (and, with a different fake, "the
    cache missed and the encoder refused").
    """
    application = viewer.ViewerApp(embeddings_dir=embeddings_dir, local_encoder=fake_encoder)
    application.embeddings.refresh()
    application.runner = FakeRunner()
    application.error = None
    application.boot_seconds = 0.01
    return application
