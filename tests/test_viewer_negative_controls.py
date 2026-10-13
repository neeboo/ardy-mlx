"""Negative controls, or: does each viewer claim actually *break* under a fault?

A green test only means something if the thing it checks can go red. Each test here
introduces one deliberate, plausible fault and asserts that either the server refuses
the request by name, or the produced clip is measurably wrong. The ones about wrong
output are the important ones: a viewer that silently mixes up the prompt, the
duration or the initial noise still renders a completely convincing skeleton, and
nothing on the page would look amiss.

The "must fail loudly" controls need no checkpoint (a fake runner supplies correctly
shaped noise). The controls that need real weights skip with a reason when
``checkpoints/`` is not populated.
"""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def _load_viewer():
    """Load ``scripts/ardy_viewer.py`` under a stable module name.

    ``tests/`` and ``scripts/`` are not packages, and pytest's import modes disagree
    about whether ``conftest`` can be imported by name. This by-path load is what
    every viewer test file uses, and because the module name is fixed, all four files
    -- and the fixtures in ``tests/conftest.py`` -- get the **same** module object.
    Anything else would make ``monkeypatch.setattr`` patch a copy, which is a green
    test that proves nothing.
    """
    import importlib.util

    name = "ardy_viewer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ardy_viewer.py")
    assert spec and spec.loader, "cannot load scripts/ardy_viewer.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ardy_viewer = _load_viewer()

from ardy_mlx.sampling import pipeline  # noqa: E402

#: The shipped Core layout, a 40x cheaper layout with the same arithmetic, and the
#: prompt the one authorised reference run encoded.
CORE = ardy_viewer.PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
TINY = CORE.__class__(nbjoints=27, fps=5, horizon=5, skeleton="CoreSkeleton27")
PROMPT = "A person walks forward."
PROMPT_HASH = pipeline.prompt_hash(PROMPT)
WRONG_PROMPT = "A person runs backwards."
GOLDEN_2S_NOISE = ROOT / "artifacts" / "golden-run" / "raw" / "denoise_step0_x_in.npy"
#: The 4.0 s run's *first denoise input* is its whole 20-token window -- the shape a
#: 2.0 s request must not accept. (Its `randn_0.npy` is a per-window draw of the same
#: 10-token shape as the 2.0 s dump, which is why this control uses this file.)
GOLDEN_4S_WIDE_NOISE = ROOT / "artifacts" / "golden-run-4s" / "raw" / "denoise_step0_x_in.npy"
GOLDEN_2S_NPZ = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"


def synthetic_frames(num_frames: int, seed: int = 0):
    """A deterministic ``(num_frames, 330)`` feature block, in physical units.

    The layout is the contract's: ``root_pos(3) | heading(2) | local_joints(78) |
    global_rot_data(162) | velocities(81) | foot_contacts(4)``. The rotation block is
    a small perturbation of the identity 6D basis so ``cont6d_to_matrix`` produces a
    real rotation, and a contact is set on the first frame of every 5-frame group so
    a "no contacts anywhere" bug cannot pass silently.

    It lives in each file that needs it rather than behind an import: the only thing
    the viewer test files share is the module under test, and a fixture cannot be
    used to build a module-level constant.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    frames = np.zeros((num_frames, CORE.motion_rep_dim), dtype=np.float32)
    t = np.arange(num_frames, dtype=np.float32)
    frames[:, 0] = 0.05 * t
    frames[:, 1] = 0.9 + 0.02 * np.sin(t)
    frames[:, 3] = 1.0
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

    def __init__(self, layout=None, num_windows: int = 1, seed: int = 0):
        self.model = "ARDY-Core-RP-20FPS-Horizon40"
        self.layout = layout or CORE
        self.parameter_count = 1234
        self.load_seconds = {"tokenizer_s": 0.0, "denoiser_s": 0.0, "total_s": 0.0}
        self.device = "fake"
        self.embedding = None
        self.num_windows = num_windows
        self.seed = seed
        self.calls: list[dict] = []

    @property
    def noise_window_shape(self):
        from ardy_mlx.io import NUM_FRAMES_PER_TOKEN

        return (1, self.layout.horizon // NUM_FRAMES_PER_TOKEN, self.layout.dim_token)

    def generate(self, *, num_frames, num_steps=10, cfg_weight=(2.0, 2.0), history_frames=None,
                 noise_fn=None, seed=None, windows=None):
        import numpy as np

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


def test_negative_a_missing_embedding_is_refused_rather_than_substituted(tmp_path):
    """The fault: ask for a prompt that has no cached embedding.

    The tempting wrong behaviour is to run the one prompt that *is* cached, which
    would render a perfectly convincing walking clip under the wrong caption.
    """
    directory = tmp_path / "embeddings"
    directory.mkdir()
    text = "A person walks forward."
    digest = pipeline.prompt_hash(text)
    np.save(directory / f"{digest}.npy", np.zeros((1, 1, 4096), dtype=np.float32))
    (directory / f"{digest}.json").write_text(json.dumps({"prompt": text}), encoding="utf-8")

    app = ardy_viewer.ViewerApp(embeddings_dir=directory)
    app.embeddings.refresh()
    app.runner = FakeRunner(layout=CORE)

    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    assert excinfo.value.name == "no_cached_embedding_for_prompt"
    # And the substitution really is absent: nothing was sampled at all.
    assert app.runner.calls == []
    assert app.embedding_cache == {}


def test_negative_a_partial_horizon_is_refused_rather_than_rounded(tmp_path):
    """The fault: ask for 3.0 s, which is 1.5 horizons.

    Rounding up to 4.0 s or down to 2.0 s would both produce a real clip that is
    not the clip the HUD claims.
    """
    directory = tmp_path / "embeddings"
    directory.mkdir()
    text = "A person walks forward."
    digest = pipeline.prompt_hash(text)
    np.save(directory / f"{digest}.npy", np.zeros((1, 1, 4096), dtype=np.float32))
    (directory / f"{digest}.json").write_text(json.dumps({"prompt": text}), encoding="utf-8")

    app = ardy_viewer.ViewerApp(embeddings_dir=directory)
    app.embeddings.refresh()
    app.runner = FakeRunner(layout=CORE)

    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.plan(prompt=digest, duration=3.0, noise="random")
    assert excinfo.value.name == "duration_not_horizon_multiple"
    assert app.runner.calls == []


def test_negative_a_missing_golden_clip_is_refused_rather_than_nearest(tmp_path):
    """The fault: ask for the 6.0 s reference.

    The tempting wrong behaviour is to show the 4.0 s clip, which is the nearest
    one on disk and would look entirely plausible next to a 6.0 s generation.
    """
    app = ardy_viewer.ViewerApp(embeddings_dir=tmp_path)
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.reference(6.0)
    assert excinfo.value.name == "no_golden_reference_for_duration"
    assert excinfo.value.status == 404


@pytest.mark.skipif(
    not GOLDEN_2S_NOISE.is_file() or not GOLDEN_4S_WIDE_NOISE.is_file(),
    reason="golden-run raw noise dumps are not present (they are weight-derived, never committed)",
)
def test_negative_the_wrong_window_size_noise_is_refused_by_shape(tmp_path):
    """The fault: feed the 4.0 s run's 20-token x_T window to a 2.0 s request.

    This is the mistake the repository has already paid for once (contract S11
    N16): a wrong x_T produces plausible motion, so the check has to be a hard
    shape failure, not a warning. The two real dumps have different token counts,
    which is exactly what the check is for.
    """
    window_2s = np.load(GOLDEN_2S_NOISE, allow_pickle=False)
    window_4s = np.load(GOLDEN_4S_WIDE_NOISE, allow_pickle=False)
    assert window_4s.shape[1] != window_2s.shape[1], (
        "the two dumps must differ in token count for this control to mean anything"
    )

    runner = FakeRunner(layout=CORE)
    # The correctly-shaped 2.0 s window is accepted ...
    accepted = ardy_viewer.resolve_noise_windows(window_2s, runner.noise_window_shape, 1, source="control")
    assert accepted[0].shape == runner.noise_window_shape
    # ... and the 4.0 s window, which is a real reference draw, is not.
    with pytest.raises(ValueError) as excinfo:
        ardy_viewer.resolve_noise_windows(window_4s, runner.noise_window_shape, 1, source="control")
    assert "expected" in str(excinfo.value)
    # And the file list the viewer actually picks per duration follows the run's
    # own window count: one window for 2.0 s, two for 4.0 s.
    assert len(ardy_viewer.golden_noise_files(2.0)) == 1
    assert len(ardy_viewer.golden_noise_files(4.0)) == 2


def test_negative_a_single_window_dump_may_not_stand_in_for_a_two_window_run(tmp_path):
    """The fault: pass one x_T for a two-window run.

    Window 1 would silently fall back to MLX's RNG and only window 0 would be the
    reference. The shared validator refuses the partial injection.
    """
    window = np.zeros((1, 10, 148), dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        ardy_viewer.resolve_noise_windows(window, (1, 10, 148), 2, source="control")
    assert "one window" in str(excinfo.value) and "windows" in str(excinfo.value)


def test_negative_a_short_stacked_dump_is_refused(tmp_path):
    """The fault: two windows of noise for a four-window run."""
    stacked = np.zeros((2, 1, 10, 148), dtype=np.float32)
    with pytest.raises(ValueError) as excinfo:
        ardy_viewer.resolve_noise_windows(stacked, (1, 10, 148), 4, source="control")
    assert "refusing a partial injection" in str(excinfo.value)


# --------------------------------------------------------------------------- #
# controls that need the real weights: the clip must be *wrong*, not just refused
# --------------------------------------------------------------------------- #
REAL_CHECKPOINT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40" / "mlx" / "denoiser.mlx.safetensors"


def _real_app(tmp_path):
    """A real :class:`ViewerApp` on the real checkpoint, with a two-prompt cache.

    Two prompts matter: the wrong-prompt control below needs a *real* second
    embedding that generates a *real* clip, so "the clip changed" is a statement
    about the model and not about a zero array.
    """
    directory = tmp_path / "embeddings"
    directory.mkdir()
    rng = np.random.default_rng(0)
    for text in ("A person walks forward.", WRONG_PROMPT):
        digest = pipeline.prompt_hash(text)
        # The second prompt's embedding is a perturbation of the first's, not
        # noise: a wholly random 4096-vector would put the model outside anything
        # it was trained on and make the comparison meaningless.
        base = rng.standard_normal((1, 1, 4096)).astype(np.float32)
        np.save(directory / f"{digest}.npy", base)
        (directory / f"{digest}.json").write_text(json.dumps({"prompt": text}), encoding="utf-8")
    # Give the *cached golden prompt* its real embedding, so the golden-noise run
    # below is actually comparable with the reference clip.
    source = ROOT / "artifacts" / "golden-run" / "raw" / "text_feat.npy"
    if source.is_file():
        np.save(directory / f"{pipeline.prompt_hash('A person walks forward.')}.npy",
                np.load(source, allow_pickle=False))
    app = ardy_viewer.ViewerApp(embeddings_dir=directory)
    app.start()
    assert app.runner is not None, app.error
    return app, directory


@pytest.mark.skipif(
    not REAL_CHECKPOINT.is_file() or not GOLDEN_2S_NPZ.is_file(),
    reason="converted checkpoint or golden npz not present (both are weight-derived, never committed)",
)
def test_negative_changing_the_prompt_changes_the_clip(tmp_path):
    """The fault: conditioning on the wrong prompt.

    Both runs use golden noise, so the only difference is the text embedding. If
    the clip did *not* move, "the page shows which prompt it used" would be a claim
    about the HUD rather than about the motion.
    """
    app, _ = _real_app(tmp_path)
    forward = ROOT / "artifacts" / "golden-run" / "raw" / "text_feat.npy"
    if not forward.is_file():
        pytest.skip("the golden prompt embedding is not on disk")
    same = app.plan(prompt="A person walks forward.", duration=2.0, noise="golden")
    other = app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="golden")

    import base64

    a = np.frombuffer(base64.b64decode(same["posed_joints_b64"]), dtype="<f4")
    b = np.frombuffer(base64.b64decode(other["posed_joints_b64"]), dtype="<f4")
    assert a.shape == b.shape
    gap = float(np.abs(a - b).max())
    golden = np.load(GOLDEN_2S_NPZ, allow_pickle=False)["posed_joints"].reshape(-1)
    residue = float(np.abs(a - golden).max())
    assert residue < 1e-4, "the forward-prompt run should reproduce the reference"
    assert gap > 1e-3, f"prompt perturbation barely moved the clip ({gap:g}); the control is vacuous"
    # The reported fingerprints really are different, so the payload says which
    # embedding produced which clip.
    assert same["embedding"]["embedding_sha256"] != other["embedding"]["embedding_sha256"]


@pytest.mark.skipif(
    not REAL_CHECKPOINT.is_file(),
    reason="converted checkpoint not present (weight-derived, never committed)",
)
def test_negative_changing_the_seed_changes_a_random_clip(tmp_path):
    """The fault: the seed input not reaching the RNG."""
    app, _ = _real_app(tmp_path)
    one = app.plan(prompt="A person walks forward.", duration=2.0, noise="random", seed=1)
    two = app.plan(prompt="A person walks forward.", duration=2.0, noise="random", seed=2)

    import base64

    a = np.frombuffer(base64.b64decode(one["posed_joints_b64"]), dtype="<f4")
    b = np.frombuffer(base64.b64decode(two["posed_joints_b64"]), dtype="<f4")
    assert float(np.abs(a - b).max()) > 1e-4


@pytest.mark.skipif(
    not REAL_CHECKPOINT.is_file() or not GOLDEN_2S_NPZ.is_file(),
    reason="converted checkpoint or golden npz not present (weight-derived, never committed)",
)
def test_negative_golden_noise_is_what_makes_the_clip_match_the_reference(tmp_path):
    """The fault: using MLX's RNG where the reference x_T is required.

    A random draw produces a valid clip; it just is not the reference clip. The
    measured distance is the reason the page labels which noise it used, and the
    reason a "reproduces the reference" claim is only made for golden noise.
    """
    app, _ = _real_app(tmp_path)
    golden = app.plan(prompt="A person walks forward.", duration=2.0, noise="golden")
    random = app.plan(prompt="A person walks forward.", duration=2.0, noise="random", seed=0)

    import base64

    reference = np.load(GOLDEN_2S_NPZ, allow_pickle=False)["posed_joints"].reshape(-1)
    golden_joints = np.frombuffer(base64.b64decode(golden["posed_joints_b64"]), dtype="<f4")
    random_joints = np.frombuffer(base64.b64decode(random["posed_joints_b64"]), dtype="<f4")

    golden_residue = float(np.abs(golden_joints - reference).max())
    random_gap = float(np.abs(random_joints - golden_joints).max())
    assert golden_residue < 1e-4, f"golden noise should reproduce the reference, got {golden_residue:g}"
    assert random_gap > 1e-2, f"random noise produced the reference clip ({random_gap:g}) -- the label would be a lie"
    assert golden["noise"]["mode"] == "golden" and random["noise"]["mode"] == "random"
