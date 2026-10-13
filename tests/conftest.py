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

import importlib.util
import json
import sys
from pathlib import Path

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
