"""Viewer service tests: the real checkpoint, and the one rule threading has to obey.

Two different things are checked here.

1. **MLX is used from one thread.** Its GPU stream is thread-local; loading the
   weights in one thread and evaluating in another fails with ``RuntimeError: There
   is no Stream(gpu, 1) in current thread``. The viewer binds its socket, then loads
   *and* serves on a single worker thread, and this file drives
   :func:`ardy_viewer.main` to prove that arrangement produces a clip.
2. **The viewer and the CLI load the checkpoint through the same function**
   (:func:`ardy_mlx.sampling.pipeline.build_mlx_runner`), so ``--backend mlx`` and
   ``/api/plan`` cannot drift into two definitions of "the standard path". The check
   is a direct frame comparison, not a code inspection.

Everything that needs the converted weights or the golden dumps skips with an
explicit reason when they are absent (they are weight-derived and never committed;
see ``NOTICE``).
"""

from __future__ import annotations

import base64
import json
import socket
import threading
import time
import urllib.error
import urllib.request

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

import mlx.core as mx  # noqa: E402

from ardy_mlx.sampling import pipeline  # noqa: E402

#: The shipped Core layout, a 40x cheaper layout with the same arithmetic, and the
#: prompt the one authorised reference run encoded.
CORE = ardy_viewer.PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
TINY = CORE.__class__(nbjoints=27, fps=5, horizon=5, skeleton="CoreSkeleton27")
PROMPT = "A person walks forward."
PROMPT_HASH = pipeline.prompt_hash(PROMPT)
CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"
TOKENIZER = CKPT / "mlx" / "tokenizer.mlx.safetensors"
DENOISER = CKPT / "mlx" / "denoiser.mlx.safetensors"
GOLDEN_2S_NPZ = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"
GOLDEN_4S_NPZ = ROOT / "artifacts" / "golden-run-4s" / "out" / "golden_core_h40_4s_seed0.npz"
EMBEDDINGS = ROOT / "artifacts" / "embeddings"

WEIGHTS_PRESENT = TOKENIZER.is_file() and DENOISER.is_file()
WEIGHTS_REASON = (
    "converted MLX weights are not present (run scripts/ardy_convert.py; NOTICE forbids "
    "committing them)"
)
GOLDEN_REASON = "golden run npz is not present (weight-derived, never committed)"


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


def _free_port() -> int:
    """An ephemeral loopback port, so the test never collides with a running viewer."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def load_script(name: str, relative: str):
    """Import another ``scripts/*.py`` by path (they are not a package either)."""
    import importlib.util

    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader, f"cannot load {relative}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


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


def test_the_shipped_command_serves_a_real_clip_over_http():
    """Run ``main()`` in-process, then drive it over a real socket.

    ``main`` is started on a daemon thread of this process rather than in a
    subprocess: the point is the *arrangement* (bind, then load and serve from one
    and the same thread), and the payload is checked against the golden reference
    so "it answered 200" is not mistaken for "it produced the right clip".

    Two things this catches that a direct ``ViewerApp.plan`` call cannot: that the
    HTTP layer really is single-threaded end to end, and that
    ``RuntimeError: There is no Stream(gpu, 1) in current thread`` has not come
    back (which is what happens if the weights are loaded on one thread and used
    from another).
    """
    port = _free_port()
    argv = [
        "--port", str(port), "--device", "gpu",
        "--embeddings-dir", str(EMBEDDINGS),
        # Self-terminate, so the server thread main() owns cannot outlive this test
        # and hold the collection open.
        "--exit-after", "150",
    ]
    exit_codes: list[int] = []
    thread = threading.Thread(
        target=lambda: exit_codes.append(ardy_viewer.main(argv)), daemon=True, name="main-under-test"
    )
    thread.start()

    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 120
    health = None
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{base}/api/health", timeout=5) as response:
                health = json.loads(response.read())
            break
        except Exception:  # noqa: BLE001 - the socket is not listening yet
            time.sleep(0.25)
    assert health is not None, "the viewer never started listening"
    assert health["status"] == "ok", health["error"]
    assert health["parameters"] > 100_000_000

    body = json.dumps({"prompt": PROMPT_HASH, "seed": 0, "duration": 2.0, "noise": "golden"}).encode()
    request = urllib.request.Request(
        f"{base}/api/plan", data=body, method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=600) as response:
        payload = json.loads(response.read())
    assert payload["frames"] == 40
    assert payload["noise"]["mode"] == "golden"
    joints = np.frombuffer(
        __import__("base64").b64decode(payload["posed_joints_b64"]), dtype="<f4"
    ).reshape(40, 27, 3)
    reference = np.load(GOLDEN_2S_NPZ, allow_pickle=False)["posed_joints"]
    assert float(np.abs(joints - reference).max()) < 1e-3

    # Errors must survive the same path, with their names intact.
    bad = json.dumps({"prompt": PROMPT_HASH, "duration": 1.0, "noise": "random"}).encode()
    bad_request = urllib.request.Request(
        f"{base}/api/plan", data=bad, method="POST", headers={"Content-Type": "application/json"}
    )
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(bad_request, timeout=60)
    assert excinfo.value.code == 400
    assert json.loads(excinfo.value.read())["error"] == "duration_not_horizon_multiple"

    assert exit_codes == []          # main() is still serving, as it should be
    assert thread.is_alive()


# --------------------------------------------------------------------------- #
# the viewer and the CLI share one loader
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not WEIGHTS_PRESENT, reason=WEIGHTS_REASON)
@pytest.mark.skipif(
    not (EMBEDDINGS / f"{PROMPT_HASH}.npy").is_file(),
    reason="the cached prompt embedding is not present (see embeddings/README.md)",
)
def test_the_viewer_and_the_cli_generate_the_same_frames(tmp_path):
    """``build_mlx_runner(...).generate`` and the CLI's ``_run_mlx_backend`` agree.

    The CLI is driven the way ``tests/test_generate_cli.py`` drives it -- through
    ``main()``, with the reference ``x_T`` injected -- and its sidecar's frame count
    and the viewer's are compared after decoding both npz/plan outputs. A shared
    loader that had quietly diverged (a different history crop, a different CFG
    weight, a different noise path) would show up here.
    """
    embedding = pipeline.load_text_embedding(EMBEDDINGS / f"{PROMPT_HASH}.npy")
    runner = pipeline.build_mlx_runner("ARDY-Core-RP-20FPS-Horizon40", embedding, repo_root=ROOT)
    noise = np.load(ROOT / "artifacts" / "golden-run" / "raw" / "denoise_step0_x_in.npy", allow_pickle=False)

    def injected(step, shape):
        assert tuple(shape) == runner.noise_window_shape, (shape, runner.noise_window_shape)
        return mx.array(noise)

    direct = runner.generate(
        num_frames=40, num_steps=10, cfg_weight=(2.0, 2.0), history_frames=None,
        noise_fn=injected, seed=0, windows=1,
    )
    mx.eval(direct["frames"])
    direct_frames = np.asarray(direct["frames"], dtype=np.float32)

    # The same run through the shipped CLI, which must land on the same numbers.
    cli = load_script("ardy_generate_cli_for_viewer", "scripts/ardy_generate.py")
    out_a = tmp_path / "cli"
    ncalls = {"n": 0}

    def fake_normal(shape, key=None):
        ncalls["n"] += 1
        return mx.array(noise)

    original = mx.random.normal
    mx.random.normal = fake_normal
    try:
        code = cli.main([
            "--embedding", str(EMBEDDINGS / f"{PROMPT_HASH}.npy"),
            "--model", "ARDY-Core-RP-20FPS-Horizon40",
            "--backend", "mlx", "--duration", "2.0", "--seed", "0",
            "--no-postprocess", "--out", str(out_a), "--checkpoint-dir", str(CKPT),
        ])
    finally:
        mx.random.normal = original
    assert code == 0, f"the CLI exited {code}"
    npz = next((out_a / "ARDY-Core-RP-20FPS-Horizon40").glob("*.npz"))
    with np.load(npz, allow_pickle=False) as data:
        cli_frames = np.asarray(data["frames"], dtype=np.float32)
    assert ncalls["n"] > 0, "the CLI never drew an x_T; the injection was not exercised"
    assert cli_frames.shape == direct_frames.shape
    assert float(np.abs(cli_frames - direct_frames).max()) == 0.0, (
        "the viewer's runner and the CLI no longer produce the same clip"
    )


# --------------------------------------------------------------------------- #
# the real thing, against the golden reference
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not WEIGHTS_PRESENT, reason=WEIGHTS_REASON)
@pytest.mark.skipif(not GOLDEN_2S_NPZ.is_file(), reason=GOLDEN_REASON)
@pytest.mark.skipif(
    not (EMBEDDINGS / f"{PROMPT_HASH}.npy").is_file(),
    reason="the cached prompt embedding is not present (see embeddings/README.md)",
)
def test_the_viewer_reproduces_the_two_second_golden_clip():
    """End to end through ``ViewerApp.plan``: 2.0 s, golden noise, one window.

    5.178e-06 is the residue the repository's alignment report records for
    ``posed_joints`` at this duration; the bound below is the frozen 1e-3
    tolerance, loose enough to be about correctness rather than about float32
    ordering.
    """
    app = ardy_viewer.ViewerApp(embeddings_dir=EMBEDDINGS, device="gpu")
    app.start()
    assert app.runner is not None, app.error

    payload = app.plan(prompt=PROMPT, duration=2.0, noise="golden", seed=0)
    assert payload["windows"] == 1
    joints = np.frombuffer(__import__("base64").b64decode(payload["posed_joints_b64"]), dtype="<f4")
    joints = joints.reshape(payload["frames"], payload["joints"], 3)
    reference = np.load(GOLDEN_2S_NPZ, allow_pickle=False)["posed_joints"]
    assert joints.shape == reference.shape
    assert float(np.abs(joints - reference).max()) < 1e-3


@pytest.mark.skipif(not WEIGHTS_PRESENT, reason=WEIGHTS_REASON)
@pytest.mark.skipif(not GOLDEN_4S_NPZ.is_file(), reason=GOLDEN_REASON)
@pytest.mark.skipif(
    not (EMBEDDINGS / f"{PROMPT_HASH}.npy").is_file(),
    reason="the cached prompt embedding is not present (see embeddings/README.md)",
)
def test_the_viewer_reproduces_the_four_second_golden_clip():
    """The multi-window path: two windows, one recorded x_T each."""
    app = ardy_viewer.ViewerApp(embeddings_dir=EMBEDDINGS, device="gpu")
    app.start()
    assert app.runner is not None, app.error

    payload = app.plan(prompt=PROMPT, duration=4.0, noise="golden", seed=0)
    assert payload["windows"] == 2
    assert payload["offsets"] == [0, 40, 80]
    assert len(payload["noise"]["source_files"]) == 2
    joints = np.frombuffer(__import__("base64").b64decode(payload["posed_joints_b64"]), dtype="<f4")
    joints = joints.reshape(payload["frames"], payload["joints"], 3)
    reference = np.load(GOLDEN_4S_NPZ, allow_pickle=False)["posed_joints"]
    assert joints.shape == reference.shape
    assert float(np.abs(joints - reference).max()) < 1e-3


# --------------------------------------------------------------------------- #
# the reference endpoint reads the real files
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not GOLDEN_2S_NPZ.is_file(), reason=GOLDEN_REASON)
def test_the_reference_endpoint_returns_the_real_golden_clip():
    app = ardy_viewer.ViewerApp(embeddings_dir=EMBEDDINGS)
    payload = app.reference(2.0)
    assert payload["frames"] == 40 and payload["joints"] == 27
    assert payload["fps"] == 20
    assert payload["prompt"] == PROMPT
    import base64

    joints = np.frombuffer(base64.b64decode(payload["posed_joints_b64"]), dtype="<f4").reshape(40, 27, 3)
    reference = np.load(GOLDEN_2S_NPZ, allow_pickle=False)["posed_joints"]
    assert float(np.abs(joints - reference).max()) == 0.0
    contacts = np.frombuffer(base64.b64decode(payload["foot_contacts_b64"]), dtype=np.uint8).reshape(40, 4)
    golden_contacts = np.load(GOLDEN_2S_NPZ, allow_pickle=False)["foot_contacts"].astype(np.uint8)
    assert (contacts == golden_contacts).all()
