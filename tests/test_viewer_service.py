"""Viewer service tests: the real checkpoint, and the one rule threading has to obey.

Two different things are checked here.

1. **The MLX call happens in the process's main thread.** The MLX GPU stream is
   thread-local; a worker thread that evaluates on a stream another thread opened
   is what aborts the process. The viewer's answer is a single-threaded
   ``HTTPServer`` that preloads the model *before* ``serve_forever``. :func:`main`
   is therefore driven here in-process, with the model started first and requests
   served from another thread -- which is exactly the shipped arrangement -- and
   the payload is checked for a real clip.

2. **The viewer and the CLI load the checkpoint through the same function**
   (:func:`ardy_mlx.sampling.pipeline.build_mlx_runner`), so ``--backend mlx`` and
   ``/api/plan`` cannot drift into two different definitions of "the standard
   path". The check is a direct frame comparison, not a code inspection.

Everything that needs the converted weights or the golden dumps skips with an
explicit reason when they are absent (they are weight-derived and never committed;
see ``NOTICE``).
"""

from __future__ import annotations

import importlib.util
import json
import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.test_viewer_api import CORE  # noqa: E402

CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"
TOKENIZER = CKPT / "mlx" / "tokenizer.mlx.safetensors"
DENOISER = CKPT / "mlx" / "denoiser.mlx.safetensors"
GOLDEN_2S_NPZ = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"
GOLDEN_4S_NPZ = ROOT / "artifacts" / "golden-run-4s" / "out" / "golden_core_h40_4s_seed0.npz"
EMBEDDINGS = ROOT / "artifacts" / "embeddings"

WEIGHTS_PRESENT = TOKENIZER.is_file() and DENOISER.is_file()
WEIGHTS_REASON = "converted MLX weights are not present (run scripts/ardy_convert.py; NOTICE forbids committing them)"
GOLDEN_REASON = "golden run npz is not present (weight-derived, never committed)"


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader, f"cannot load {relative}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


viewer = _load("ardy_viewer_service_under_test", "scripts/ardy_viewer.py")

import mlx.core as mx  # noqa: E402

from ardy_mlx.sampling import pipeline  # noqa: E402

PROMPT = "A person walks forward."
PROMPT_HASH = pipeline.prompt_hash(PROMPT)


# --------------------------------------------------------------------------- #
# the threading contract
# --------------------------------------------------------------------------- #
def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.mark.skipif(not WEIGHTS_PRESENT, reason=WEIGHTS_REASON)
@pytest.mark.skipif(not GOLDEN_2S_NPZ.is_file(), reason=GOLDEN_REASON)
@pytest.mark.skipif(
    not (EMBEDDINGS / f"{PROMPT_HASH}.npy").is_file(),
    reason="the cached prompt embedding is not present (see embeddings/README.md)",
)
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
        target=lambda: exit_codes.append(viewer.main(argv)), daemon=True, name="main-under-test"
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
    cli = _load("ardy_generate_cli_for_viewer", "scripts/ardy_generate.py")
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
    app = viewer.ViewerApp(embeddings_dir=EMBEDDINGS, device="gpu")
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
    app = viewer.ViewerApp(embeddings_dir=EMBEDDINGS, device="gpu")
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
    app = viewer.ViewerApp(embeddings_dir=EMBEDDINGS)
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
