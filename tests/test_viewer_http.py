"""Viewer HTTP tests: routes, status codes, static-file containment, error names.

The whole point of the viewer's server design is that **one thread** owns MLX (the
GPU stream is thread-local). These tests exercise the real
:class:`ardy_viewer.ViewerServer` and :class:`ardy_viewer.ViewerHandler` over a
loopback socket, on an ephemeral port, with a fake runner standing in for the
checkpoint -- so the HTTP layer is real while nothing here loads 765 MB of weights
or evaluates on the GPU. The FK is stubbed in numpy as well, so no test ever enters
MLX from the server's thread.
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
from urllib.parse import quote

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

from ardy_mlx.router import jev  # noqa: E402

#: The shipped Core layout, a 40x cheaper layout with the same arithmetic, and the
#: prompt the one authorised reference run encoded.
CORE = ardy_viewer.PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
TINY = CORE.__class__(nbjoints=27, fps=5, horizon=5, skeleton="CoreSkeleton27")
PROMPT = "A person walks forward."
PROMPT_HASH = pipeline.prompt_hash(PROMPT)
WRONG_PROMPT = "A person runs backwards."


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


def _posed_joints_without_mlx(frames):
    """FK in numpy, so no test ever enters MLX from a server thread.

    Mirrors ``posed_joints_from_features`` for this one call shape; the real function
    is exercised on the main thread by ``tests/test_model_skeleton.py`` and end to end
    by ``tests/test_viewer_service.py``. Its importance here is that the HTTP layer is
    tested without touching a thread-local GPU stream.
    """
    feats = np.asarray(frames, dtype=np.float32)
    root_positions = feats[..., 0:3]
    centres = np.asarray(feats[..., 5 : 5 + 26 * 3], dtype=np.float32).reshape(
        feats.shape[:-1] + (26, 3)
    )
    joints = np.zeros(feats.shape[:-1] + (27, 3), dtype=np.float32)
    joints[..., 0, :] = root_positions
    joints[..., 1:, :] = centres + root_positions[..., None, :]
    return joints


@pytest.fixture()
def live_server(embeddings_dir, monkeypatch, tmp_path):
    """A real HTTP server on 127.0.0.1:0, in a daemon thread, with a fake runner."""
    monkeypatch.setattr(ardy_viewer, "posed_joints_from_features", lambda frames, skeleton=None: _posed_joints_without_mlx(frames))
    # Point the static handler at a temporary viewer directory with one file per
    # whitelisted kind, so the containment tests cannot read the real checkout.
    static = tmp_path / "viewer"
    (static / "vendor").mkdir(parents=True)
    (static / "index.html").write_text("<!doctype html><title>t</title>", encoding="utf-8")
    (static / "app.js").write_text("export const ok = 1;\n", encoding="utf-8")
    (static / "vendor" / "three.module.min.js").write_text("// vendored\n", encoding="utf-8")
    # Real files in the viewer directory that are NOT on the whitelist. They must
    # be unreachable, which is what makes "the whitelist is exact" testable: a
    # suffix rule would serve both of these.
    (static / "notes.js").write_text("export const leak = 1;\n", encoding="utf-8")
    (static / "secret.html").write_text("<!doctype html><p>leak", encoding="utf-8")
    monkeypatch.setattr(ardy_viewer, "VIEWER_DIR", static)

    app = ardy_viewer.ViewerApp(embeddings_dir=embeddings_dir)
    app.embeddings.refresh()
    app.runner = FakeRunner(layout=CORE)
    app.error = None
    app.boot_seconds = 0.005

    server = ardy_viewer.ViewerServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", app
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


class _FakeTypeSafeClient:
    """A ``TypeSafeClient`` whose answers come from a dict.

    The real :class:`ardy_mlx.router.jev.JevRouter` is used over the socket -- real
    threads, real deadline, real cache, real mapping -- and only the HTTP call to
    TypeSafe is replaced. That keeps the HTTP test about the wire shape and the
    degradation behaviour without a network or an API key.
    """

    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": list(questions)})
        return type("_Response", (), {"answers": self.answers, "model": "jev-fake"})()


def _answers(route_name, *, probabilities=None, is_motion=0.97, library_pick=jev.LIBRARY_NONE):
    probabilities = probabilities or {
        jev.ROUTE_LIBRARY: 0.1, jev.ROUTE_COMPOSE: 0.8,
        jev.ROUTE_GENERATE: 0.05, jev.ROUTE_NON_MOTION: 0.05,
    }
    return {
        jev.Q_IS_MOTION: {"type": "noul", "noul": is_motion},
        jev.Q_KNOWN: {"type": "noul", "noul": 0.85},
        jev.Q_MOTION_TYPE: {"type": "noul", "noul": 0.9},
        jev.Q_ROUTE: {"type": "choice", "choice": route_name, "confidence": max(probabilities.values()),
                      "probabilities": probabilities},
        jev.Q_TEMPO: {"type": "score", "score": 0.0, "confidence": 1.0,
                      "probabilities": {"0": 0.98, "1": 0.02, "2": 0.0}, "legend": {}},
        jev.Q_INTENSITY: {"type": "score", "score": 1.0, "confidence": 0.9,
                          "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1}, "legend": {}},
        jev.Q_EMOTION: {"type": "choice", "choice": "平静", "confidence": 0.9, "probabilities": {"平静": 0.9}},
        jev.Q_INTERRUPT: {"type": "noul", "noul": 0.9},
        jev.Q_LIBRARY: {"type": "choice", "choice": library_pick, "confidence": 0.6,
                        "probabilities": {library_pick: 0.6}},
    }


@pytest.fixture()
def router_server(embeddings_dir, monkeypatch, tmp_path):
    """``live_server``'s twin, with a **real** router whose SDK call is faked."""
    monkeypatch.setattr(ardy_viewer, "posed_joints_from_features",
                        lambda frames, skeleton=None: _posed_joints_without_mlx(frames))
    monkeypatch.setattr(ardy_viewer, "VIEWER_DIR", tmp_path / "viewer")

    client = _FakeTypeSafeClient(_answers(jev.ROUTE_COMPOSE))
    router = jev.JevRouter(
        api_key="ts_test_key_do_not_use_0123456789",
        client_factory=lambda key, model, deadline: client,
    )
    app = ardy_viewer.ViewerApp(embeddings_dir=embeddings_dir, router=router)
    app.embeddings.refresh()
    app.runner = FakeRunner(layout=CORE)
    app.error = None
    app.boot_seconds = 0.005

    server = ardy_viewer.ViewerServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", app, client
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def request(url, method="GET", payload=None, raw_body=None, headers=None):
    """Return ``(status, body_bytes, headers)``, never raising on 4xx/5xx."""
    data = None
    if raw_body is not None:
        data = raw_body
    elif payload is not None:
        data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    for key, value in (headers or {}).items():
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return response.status, response.read(), dict(response.headers)
    except urllib.error.HTTPError as error:
        return error.code, error.read(), dict(error.headers)


def jrequest(url, method="GET", payload=None, raw_body=None):
    status, body, headers = request(url, method=method, payload=payload, raw_body=raw_body)
    try:
        parsed = json.loads(body)
    except json.JSONDecodeError:  # pragma: no cover - a failure here is the test failing
        parsed = None
    return status, parsed, headers


# --------------------------------------------------------------------------- #
# static files
# --------------------------------------------------------------------------- #
def test_root_serves_the_viewer_page(live_server):
    base, _ = live_server
    status, body, headers = request(f"{base}/")
    assert status == 200
    assert b"<!doctype html" in body.lower()
    assert headers["Content-Type"].startswith("text/html")
    assert "charset=utf-8" in headers["Content-Type"]
    assert headers["Cache-Control"] == "no-store"


def test_javascript_is_served_with_a_javascript_content_type(live_server):
    base, _ = live_server
    status, body, headers = request(f"{base}/app.js")
    assert status == 200
    assert b"export const ok" in body
    assert "javascript" in headers["Content-Type"]


def test_vendored_files_are_reachable(live_server):
    base, _ = live_server
    status, _, _ = request(f"{base}/vendor/three.module.min.js")
    assert status == 200


def test_static_whitelist_rejects_a_sibling_file(live_server):
    """An exact whitelist: real files beside the page are still not served."""
    base, _ = live_server
    for name in ("notes.js", "secret.html"):
        status, body, _ = jrequest(f"{base}/{name}")
        assert status == 404, name
        assert body["error"] == "not_found", name


@pytest.mark.parametrize(
    "path",
    [
        "/../pyproject.toml",
        "/..%2fpyproject.toml",
        "/%2e%2e/%2e%2e/etc/hosts",
        "/....//pyproject.toml",
        "/vendor/../../pyproject.toml",
        "/app.js/../../pyproject.toml",
    ],
)
def test_path_traversal_never_leaves_the_viewer_directory(live_server, path):
    base, _ = live_server
    status, body, _ = request(f"{base}{path}")
    assert status == 404
    assert b"ardy-mlx" not in body or b"name =" not in body


def test_unknown_api_route_is_not_a_file_lookup(live_server):
    base, _ = live_server
    status, body, _ = jrequest(f"{base}/api/nope")
    assert status == 404
    assert body["error"] == "not_found"


# --------------------------------------------------------------------------- #
# /api/health
# --------------------------------------------------------------------------- #
def test_health_is_json_and_names_everything_the_page_shows(live_server):
    base, _ = live_server
    status, body, headers = jrequest(f"{base}/api/health")
    assert status == 200
    assert headers["Content-Type"].startswith("application/json")
    assert body["status"] == "ok"
    assert body["error"] is None
    assert body["model"] == "ARDY-Core-RP-20FPS-Horizon40"
    assert body["device"]["backend"] == "fake" or "name" in body["device"]
    assert body["parameters"] == 1234
    assert body["cold_start_s"] == pytest.approx(0.005)
    assert body["horizon"] == CORE.horizon and body["fps"] == CORE.fps
    assert body["duration_step_s"] == pytest.approx(CORE.horizon / CORE.fps)
    assert body["skeleton"]["name"] == "cskel27"
    assert len(body["skeleton"]["jointNames"]) == 27
    assert len(body["skeleton"]["jointParents"]) == 27
    assert [r["duration_s"] for r in body["golden_references"]] == [2.0, 4.0]
    assert [e["prompt_hash"] for e in body["embeddings"]] == [pipeline.prompt_hash("A person walks forward.")]
    assert body["limits"]["text_encoder"].startswith("a local MLX text encoder is available")
    # The encoder block is part of the contract the page reads, even here where no
    # encoder was wired in: the state has to be sayable, not absent.
    assert body["encoder"]["state"] == "disabled"
    assert body["embeddings_detail"]["cache_keys"] == [
        [pipeline.prompt_hash("A person walks forward."), "reference-llm2vec-bf16"]
    ]


def test_health_reports_an_unloaded_model_instead_of_pretending(live_server):
    base, app = live_server
    app.runner = None
    app.error = "FileNotFoundError: checkpoint directory not found: /nope"
    status, body, _ = jrequest(f"{base}/api/health")
    assert status == 200                       # the page must still load
    assert body["status"] == "unavailable"
    assert "checkpoint directory not found" in body["error"]


def test_health_rejects_a_post(live_server):
    base, _ = live_server
    status, body, _ = jrequest(f"{base}/api/health", method="POST", payload={})
    assert status == 405
    assert body["error"] == "method_not_allowed"


# --------------------------------------------------------------------------- #
# /api/plan
# --------------------------------------------------------------------------- #
def test_plan_by_hash_over_post(live_server):
    base, _ = live_server
    digest = pipeline.prompt_hash("A person walks forward.")
    status, body, _ = jrequest(
        f"{base}/api/plan", method="POST",
        payload={"prompt": digest, "seed": 4, "duration": 2.0, "noise": "random"},
    )
    assert status == 200
    assert body["kind"] == "plan"
    assert body["frames"] == int(2.0 * CORE.fps)
    assert body["joints"] == 27
    assert body["seed"] == 4
    joints = np.frombuffer(base64.b64decode(body["posed_joints_b64"]), dtype="<f4")
    assert joints.size == body["frames"] * 27 * 3
    assert np.isfinite(joints).all()


def test_plan_by_prompt_text_over_get(live_server):
    """The GET form takes query parameters; the page uses POST, the CLI can use GET."""
    base, _ = live_server
    from urllib.parse import quote

    status, body, _ = jrequest(
        f"{base}/api/plan?prompt={quote('A person walks forward.')}&duration=2&noise=random&seed=1"
    )
    assert status == 200
    assert body["kind"] == "plan"
    assert body["noise"]["mode"] == "random"


def test_plan_returns_named_errors_not_stack_traces(live_server):
    base, _ = live_server
    cases = [
        (400, "duration_not_horizon_multiple", {"prompt": "A person walks forward.", "duration": 1.0, "noise": "random"}),
        (400, "no_cached_embedding_for_prompt", {"prompt": "A person runs.", "duration": 2.0, "noise": "random"}),
        (400, "no_prompt_given", {"duration": 2.0, "noise": "random"}),
        (400, "unknown_noise_mode", {"prompt": "A person walks forward.", "duration": 2.0, "noise": "maybe"}),
        (400, "invalid_duration", {"prompt": "A person walks forward.", "noise": "random"}),
        (400, "invalid_parameter", {"prompt": "A person walks forward.", "duration": "soon", "noise": "random"}),
    ]
    for expected_status, expected_error, payload in cases:
        status, body, _ = jrequest(f"{base}/api/plan", method="POST", payload=payload)
        assert status == expected_status, (payload, status, body)
        assert body["error"] == expected_error, (payload, body)
        assert body["message"]


def test_plan_rejects_invalid_json(live_server):
    base, _ = live_server
    status, body, _ = jrequest(f"{base}/api/plan", method="POST", raw_body=b"{not json")
    assert status == 400
    assert body["error"] == "invalid_json"


def test_plan_rejects_a_non_object_body(live_server):
    base, _ = live_server
    status, body, _ = jrequest(f"{base}/api/plan", method="POST", raw_body=b"[1, 2, 3]")
    assert status == 400
    assert body["error"] == "invalid_json"


def test_plan_rejects_an_oversized_body(live_server):
    """The body limit is refused before it is read, so a broken pipe is expected."""
    base, _ = live_server
    try:
        status, body, _ = jrequest(f"{base}/api/plan", method="POST", raw_body=b"x" * (1 << 20 | 1))
    except urllib.error.URLError:
        # The server answers 413 and closes while the client is still uploading a
        # megabyte it was never going to read. Either observation is the refusal;
        # what must not happen is the request being served.
        return
    assert status == 413
    assert body["error"] == "body_too_large"


def test_plan_without_a_model_is_503(live_server):
    base, app = live_server
    app.runner = None
    app.error = "FileNotFoundError: no converted weights"
    status, body, _ = jrequest(
        f"{base}/api/plan", method="POST",
        payload={"prompt": "A person walks forward.", "duration": 2.0, "noise": "random"},
    )
    assert status == 503
    assert body["error"] == "model_unavailable"


def test_plan_only_accepts_get_and_post(live_server):
    base, _ = live_server
    status, body, _ = jrequest(f"{base}/api/plan", method="DELETE")
    assert status == 405
    assert body["error"] == "method_not_allowed"


# --------------------------------------------------------------------------- #
# /api/reference
# --------------------------------------------------------------------------- #
def test_reference_without_a_duration_is_named(live_server):
    base, _ = live_server
    status, body, _ = jrequest(f"{base}/api/reference")
    assert status == 400
    assert body["error"] == "no_duration_given"


def test_reference_for_an_unknown_duration_is_404(live_server):
    base, _ = live_server
    status, body, _ = jrequest(f"{base}/api/reference?duration=8.0")
    assert status == 404
    assert body["error"] == "no_golden_reference_for_duration"
    assert "2.0s" in body["message"] or "2.0" in body["message"]


# --------------------------------------------------------------------------- #
# the single-threaded promise
# --------------------------------------------------------------------------- #
def test_the_server_is_not_threaded():
    """``ThreadingHTTPServer`` would hand MLX to a second thread; this must not."""
    server = ardy_viewer.ViewerServer(("127.0.0.1", 0), object())
    try:
        mro = [klass.__name__ for klass in type(server).__mro__]
        assert "ThreadingHTTPServer" not in mro
        assert "ThreadingMixIn" not in [klass.__name__ for klass in type(server).__mro__]
        assert "HTTPServer" in mro
        # The handler is the same class for every request: no per-request worker.
        assert server.RequestHandlerClass is ardy_viewer.ViewerHandler
    finally:
        server.server_close()


def test_binding_off_loopback_is_refused(tmp_path, monkeypatch, capsys):
    """The viewer has no authentication; it must not be reachable off-host."""
    code = ardy_viewer.main(["--host", "0.0.0.0", "--port", "0", "--no-preload",
                        "--embeddings-dir", str(tmp_path)])
    assert code == 2
    assert "refusing to bind" in capsys.readouterr().err


def test_cpu_device_needs_an_explicit_opt_in(embeddings_dir):
    app = ardy_viewer.ViewerApp(embeddings_dir=embeddings_dir, device="cpu")
    app.start()
    assert app.runner is None
    assert "--allow-cpu" in (app.error or "")


# --------------------------------------------------------------------------- #
# the page's own guard logic (node, no browser)
# --------------------------------------------------------------------------- #
def test_the_pages_prompt_guard_runs_under_node():
    """The one part of the page that is testable without a browser, actually run.

    The viewer's front end has no browser test harness and the project does not
    synthesise input, so the decision *whether to send a request at all* lives in a
    marker-delimited region of pure functions inside ``viewer/app.js``. That region
    is loaded out of the **shipped file** by ``tests/viewer_js_guard.test.mjs`` and
    executed under node, so a page that loses the guard -- or a region that stops
    being pure -- fails here as well as in the browser.

    ``node`` is a build-time tool, not a Python dependency: this test skips (with a
    reason) rather than failing on a host that does not have it.
    """
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is not on PATH; the page's guard logic cannot be run here")

    script = ROOT / "tests" / "viewer_js_guard.test.mjs"
    assert script.is_file(), f"missing {script}"
    completed = subprocess.run(
        [node, str(script)],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert completed.returncode == 0, (
        f"node {script.name} failed with {completed.returncode}\n"
        f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
    )
    # A green node run that discovered nothing would be a green that means nothing:
    # read the runner's own summary instead of trusting the exit status alone.
    summary = {}
    for line in completed.stdout.splitlines():
        parts = line.strip().removeprefix("\u2139 ").split()
        if len(parts) == 2 and parts[0] in ("pass", "fail") and parts[1].isdigit():
            summary[parts[0]] = int(parts[1])
    assert summary.get("fail") == 0, completed.stdout
    # Twenty cases: the three control states, the not-ready states, whitespace, the
    # body/guard negative controls, the reference decisions, purity, the bubble
    # (pending, routed, hold, degraded, cached, conservative, plain, purity), and two
    # wiring checks -- one of which proves the others can fail.
    assert summary.get("pass", 0) >= 20, completed.stdout


# --------------------------------------------------------------------------- #
# the router over the wire
# --------------------------------------------------------------------------- #
def test_plan_over_http_carries_the_router_block_and_the_decision(router_server):
    """The six named fields, on a real socket, from a real ``JevRouter``."""
    base, app, client = router_server
    status, body, _ = request(f"{base}/api/plan", "POST",
                              {"prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random"})
    assert status == 200, body
    payload = json.loads(body)
    block = payload["router"]
    for field in ("state", "router_id", "route", "probabilities", "latency_ms", "cache_hit"):
        assert field in block, field
    assert block["state"] == jev.STATE_OK
    assert block["route"] == jev.ROUTE_COMPOSE
    assert block["cache_hit"] is False
    assert block["degraded"] is False
    assert block["known_intents_rev"]
    assert payload["decision"]["action"] == jev.ACTION_COMPOSE
    assert payload["decision"]["source"] == "jev"
    assert payload["decision"]["message"], "the bubble sentence travels with the clip"


def test_health_over_http_reports_the_router_and_the_privacy_note(router_server):
    base, app, client = router_server
    status, body, _ = request(f"{base}/api/health")
    assert status == 200
    block = json.loads(body)["router"]
    assert block["state"] == jev.STATE_OK
    assert block["router_id"].startswith("jev-latest+typesafe-sdk==")
    assert block["deadline_ms"] == jev.DEFAULT_DEADLINE_MS
    assert block["available"] is True
    assert "sent to" in block["privacy"]
    # Neither the key nor its value is anywhere in the response.
    assert b"ts_test_key_do_not_use" not in body


def test_the_same_sentence_over_http_is_judged_once(router_server):
    base, app, client = router_server
    for _ in range(2):
        status, body, _ = request(f"{base}/api/plan", "POST",
                                  {"prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random"})
        assert status == 200
    assert len(client.calls) == 1, "the second identical sentence must be answered from the judgment cache"
    assert json.loads(body)["router"]["cache_hit"] is True


def test_a_non_motion_sentence_over_http_returns_a_hold_and_no_clip(router_server):
    base, app, client = router_server
    client.answers = _answers(
        jev.ROUTE_NON_MOTION, is_motion=0.05,
        probabilities={jev.ROUTE_LIBRARY: 0.02, jev.ROUTE_COMPOSE: 0.05,
                       jev.ROUTE_GENERATE: 0.03, jev.ROUTE_NON_MOTION: 0.9},
    )
    app.router.clear_cache()
    status, body, _ = request(f"{base}/api/plan", "POST",
                              {"prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random"})
    assert status == 200, body
    payload = json.loads(body)
    assert payload["kind"] == "hold"
    assert payload["decision"]["starts_new_action"] is False
    assert "frames" not in payload
    assert app.runner.calls == [], "nothing may be drawn for a sentence that did not ask for motion"


def test_a_router_that_times_out_over_http_still_draws_and_says_it_did_not_parse(embeddings_dir, monkeypatch, tmp_path):
    """The degradation path, end to end: a deadline of 1 ms cannot be met."""
    monkeypatch.setattr(ardy_viewer, "posed_joints_from_features",
                        lambda frames, skeleton=None: _posed_joints_without_mlx(frames))
    monkeypatch.setattr(ardy_viewer, "VIEWER_DIR", tmp_path / "viewer")
    client = _FakeTypeSafeClient(_answers(jev.ROUTE_COMPOSE))
    real_system_one = client.system_one

    def slow(state, questions):
        time.sleep(0.3)
        return real_system_one(state, questions)

    client.system_one = slow
    router = jev.JevRouter(api_key="ts_test_key_do_not_use_0123456789", deadline_ms=5,
                           client_factory=lambda key, model, deadline: client)
    app = ardy_viewer.ViewerApp(embeddings_dir=embeddings_dir, router=router)
    app.embeddings.refresh()
    app.runner = FakeRunner(layout=CORE)
    app.error = None
    app.boot_seconds = 0.005
    server = ardy_viewer.ViewerServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        started = time.perf_counter()
        status, body, _ = request(f"http://{host}:{port}/api/plan", "POST",
                                  {"prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random"})
        elapsed = time.perf_counter() - started
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert status == 200, body
    payload = json.loads(body)
    assert elapsed < 1.5, "the deadline bounds the wait; the slow call must not hold the request"
    assert payload["kind"] == "plan", "the plain vector route still produces a clip"
    assert payload["frames"] == 40
    assert payload["router"]["state"] == jev.STATE_TIMEOUT
    assert payload["router"]["route"] is None
    assert payload["router"]["degraded"] is True
    assert payload["decision"]["source"] == "vector"
    assert "没做语义解析" in payload["decision"]["message"]


# --------------------------------------------------------------------------- #
# continuation over the wire
# --------------------------------------------------------------------------- #
def test_health_reports_where_golden_noise_exists_and_what_continuation_is(live_server):
    """The two facts the page needs *before* it sends anything.

    `golden_references` says which durations have a golden *clip*; this is the
    different fact of which have a recorded reference ``x_T``, and it is what stops
    the page asking for `noise=golden` on a duration that cannot supply it.
    """
    base, _app = live_server
    status, body, _ = request(f"{base}/api/health")
    assert status == 200
    health = json.loads(body)
    noise = {round(entry["duration_s"], 1): entry for entry in health["golden_noise"]}
    assert set(noise) == {2.0, 4.0}, "the two authorised reference runs, and nothing invented"
    assert noise[2.0]["available"] is True and noise[2.0]["windows"] == 1
    assert noise[4.0]["available"] is True and noise[4.0]["windows"] == 2
    assert all(entry["source_files"] for entry in noise.values())
    # The fake runner has no batch path, and the page is told so by name.
    assert health["continuation"]["supported"] is False
    assert health["continuation"]["active"] is False
    assert health["continuation"]["history_frames"] == CORE.default_history_frames == 160
    assert "continues the clip" in health["continuation"]["note"]


def test_plan_over_http_says_which_continuation_it_was(live_server):
    """Every plan carries the block, and the first one is honestly not a continuation."""
    base, _app = live_server
    status, body, _ = request(f"{base}/api/plan", "POST",
                              {"prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random"})
    assert status == 200, body
    block = json.loads(body)["continuation"]
    assert block["requested"] is True, "the page sends no reset"
    assert block["applied"] is False
    assert block["reason"] == "no_previous_clip"
    assert block["seam"] is None


def test_reset_over_http_is_a_real_switch_not_a_comment(live_server):
    """`"reset": true` must reach ``plan()``, and mean something different.

    On the same server, the same request without the flag asks to continue and is
    refused only because there is nothing on screen yet; with the flag it is a
    requested reset. Two different `reason`s from one control is what makes the
    negative control (and the switch) real rather than a claim about the source.
    """
    base, app = live_server
    status, body, _ = request(f"{base}/api/plan", "POST",
                              {"prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random"})
    assert status == 200
    assert json.loads(body)["continuation"]["reason"] == "no_previous_clip"

    # Give the server something on screen without needing a model, so the flag is
    # the only difference between the two requests below.
    app.continuation = ardy_viewer.ContinuationState(
        frames=np.zeros((1, 40, 330), dtype=np.float32),
        joints=np.zeros((27, 3), dtype=np.float32),
        roots=np.zeros(3, dtype=np.float32),
        contacts=np.zeros(4, dtype=np.uint8),
        prompt_hash=PROMPT_HASH,
        prompt_text=PROMPT,
        model=app.model,
    )

    status, body, _ = request(f"{base}/api/plan", "POST",
                              {"prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random"})
    without = json.loads(body)["continuation"]
    assert status == 200
    assert without["requested"] is True
    # The fake runner cannot continue, and the payload says *that* -- not "there was
    # nothing to continue from".
    assert without["reason"] == "runner_has_no_continuation_path"

    status, body, _ = request(f"{base}/api/plan", "POST",
                              {"prompt": PROMPT_HASH, "seed": 0, "duration": 2,
                               "noise": "random", "reset": True})
    with_reset = json.loads(body)["continuation"]
    assert status == 200
    assert with_reset["requested"] is False
    assert with_reset["reason"] == "reset_requested"
    assert with_reset["applied"] is False


# --------------------------------------------------------------------------- #
# the page's own log: POST/GET /api/client-log
# --------------------------------------------------------------------------- #
# The page has no terminal. Before this endpoint, a renderer exception and a healthy
# run looked identical from the server -- 200s all the way -- which is why diagnosing
# "似乎不太对" meant guessing. These tests exercise the real wire path, the named
# refusals, and the one thing this endpoint could itself get wrong: becoming the leak.

CLIENT_LOG_SENTINEL = "ts_test_key_do_not_use_0123456789"


def _client_event(text="boom", kind="error", **extra):
    event = {"kind": kind, "text": text}
    event.update(extra)
    return event


def _all_text(value):
    """Every string anywhere in a decoded JSON payload."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_text(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_text(item)


def test_client_log_round_trip_records_and_is_queryable_by_kind(live_server):
    base, app = live_server
    payload = {
        "events": [
            _client_event("TypeError: x is not a function", at=1200.0, source="app.js", line=3, col=9),
            _client_event("three.js: context lost", kind="console.warn", at=1300.0),
        ],
        "snapshot": {"playhead": 411, "segments": 6, "buffered": 12, "held": False,
                     "route": "compose", "route_probability": 0.81,
                     # The two fields that say *where the character was*: without them
                     # "the character walked out of frame" is a guess, and they are the
                     # reason this snapshot is worth sending at all.
                     "screen": [0.12, -0.303, 5.1], "root": [0.0, 0.9, 20.4],
                     "camera": {"follow_frames": 400, "clamped": 0, "max_step_m": 0.168},
                     "url": "/"},
    }
    status, body, _ = jrequest(f"{base}/api/client-log", "POST", payload)
    assert status == 200, body
    assert body["accepted"] == 2
    assert body["kinds"] == ["console.warn", "error"]
    assert body["snapshot"]["playhead"] == 411
    assert body["snapshot"]["screen"] == [0.12, -0.303, 5.1]
    assert body["snapshot"]["root"] == [0.0, 0.9, 20.4]
    assert body["snapshot"]["camera"]["clamped"] == 0
    assert body["received_at"] is not None

    # ... and it is readable back, by name.
    status, body, _ = jrequest(f"{base}/api/client-log")
    assert status == 200
    assert body["received"] == 2
    assert body["returned"] == 2
    assert [event["kind"] for event in body["events"]] == ["error", "console.warn"]
    assert body["last_kind"] == "console.warn"
    assert body["last_at"] is not None
    assert body["snapshot"]["route"] == "compose"

    status, body, _ = jrequest(f"{base}/api/client-log?kind=error")
    assert status == 200
    assert body["matching"] == 1
    assert [event["kind"] for event in body["events"]] == ["error"]
    assert body["events"][0]["line"] == 3

    # A kind with nothing in it returns nothing -- it does not fall back to "all".
    status, body, _ = jrequest(f"{base}/api/client-log?kind=unhandledrejection")
    assert status == 200
    assert body["matching"] == 0
    assert body["events"] == []

    # `limit` trims from the newest end and says so, rather than pretending.
    status, body, _ = jrequest(f"{base}/api/client-log?limit=1")
    assert status == 200
    assert body["returned"] == 1
    assert body["truncated"] is True
    assert body["events"][0]["kind"] == "console.warn"


def test_client_log_appears_in_health_with_a_count_and_a_timestamp(live_server):
    base, app = live_server
    before = app.health()["client_log"]
    assert before["received"] == 0
    assert before["last_at"] is None
    assert before["last_kind"] is None
    assert before["kept"] == 0

    jrequest(f"{base}/api/client-log", "POST", {"events": [_client_event("nope")]})
    status, body, _ = jrequest(f"{base}/api/health")
    assert status == 200
    block = body["client_log"]
    assert block["received"] == 1
    assert block["last_kind"] == "error"
    assert isinstance(block["last_at"], float) and block["last_at"] > 0
    assert block["kept"] == 1
    assert block["keep_limit"] == ardy_viewer.CLIENT_LOG_KEEP


def test_client_log_is_named_in_the_server_log_not_silently_stored(live_server, capsys):
    # The endpoint's whole reason for existing is that the server's *log* is where a
    # reader looks. An event kept only in a JSON document nobody polls is as invisible
    # as no endpoint at all.
    base, _ = live_server
    jrequest(f"{base}/api/client-log", "POST", {
        "events": [_client_event("Uncaught ReferenceError: fitCameraToClip is not defined")],
    })
    captured = capsys.readouterr()
    assert "client-log error" in captured.err
    assert "fitCameraToClip is not defined" in captured.err

    # The snapshot is named too, because it is what says *what the page was doing*.
    jrequest(f"{base}/api/client-log", "POST", {
        "events": [], "snapshot": {"playhead": 411, "segments": 6, "held": False},
    })
    captured = capsys.readouterr()
    assert "client-log snapshot" in captured.err
    assert '"playhead": 411' in captured.err


def test_client_log_negative_control_a_key_shaped_string_never_reaches_the_log(
    live_server, capsys, monkeypatch,
):
    """This endpoint is the one that could *create* a leak, so it must not.

    Both halves run. The positive half shows the leak check can go red (the key is
    really in the text that was sent); the negative half shows that nothing which
    reaches a reader -- the POST response, `GET /api/client-log`, `/api/health`, or
    the server's own stderr -- carries it.
    """
    monkeypatch.setenv("TYPESAFE_API_KEY", CLIENT_LOG_SENTINEL)
    base, _ = live_server
    shape_only = "sk-live_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345"
    text = f"failed to call https://api.example with key {CLIENT_LOG_SENTINEL}"
    payload = {
        "events": [_client_event(text, source=f"https://x/?k={CLIENT_LOG_SENTINEL}")],
        "snapshot": {"url": f"/?token={CLIENT_LOG_SENTINEL}", "playhead": 1},
    }
    # The check can fail: the sentinel really is in what was posted.
    assert CLIENT_LOG_SENTINEL in json.dumps(payload)

    status, body, _ = jrequest(f"{base}/api/client-log", "POST", payload)
    assert status == 200, body
    stderr_after_post = capsys.readouterr().err

    status, listing, _ = jrequest(f"{base}/api/client-log")
    assert status == 200
    status_h, health, _ = jrequest(f"{base}/api/health")
    assert status_h == 200

    surfaces = {
        "POST /api/client-log": json.dumps(body),
        "GET /api/client-log": json.dumps(listing),
        "GET /api/health": json.dumps(health),
        "stderr": stderr_after_post,
    }
    for name, blob in surfaces.items():
        for text_field in _all_text(json.loads(blob)) if name != "stderr" else [blob]:
            assert CLIENT_LOG_SENTINEL not in text_field, f"the key leaked into {name}"

    # Masked, not dropped: the marker names the rule, so the field stays diagnosable.
    stored = listing["events"][0]["text"]
    assert "[redacted:" in stored
    assert "failed to call" in stored
    stderr_after_post = stderr_after_post  # already checked above

    # A key-shaped token that is *not* this process's environment still gets masked:
    # a pasted key is the case the env sweep cannot see.
    jrequest(f"{base}/api/client-log", "POST", {"events": [_client_event(f"paste {shape_only}")]})
    status, listing, _ = jrequest(f"{base}/api/client-log?kind=error")
    assert status == 200
    assert shape_only not in json.dumps(listing)
    assert "[redacted:openai-key]" in listing["events"][-1]["text"]


def test_client_log_refuses_too_many_events_by_name(live_server):
    base, app = live_server
    over = ardy_viewer.CLIENT_LOG_MAX_EVENTS + 1
    status, body, _ = jrequest(f"{base}/api/client-log", "POST",
                               {"events": [_client_event(str(i)) for i in range(over)]})
    assert status == 413
    assert body["error"] == "client_log_too_many_events"
    assert str(over) in body["message"] and str(ardy_viewer.CLIENT_LOG_MAX_EVENTS) in body["message"]
    # Refused, not trimmed: the request left nothing behind.
    assert app.client_log_report()["received"] == 0
    # ... and the refusal itself is counted, because it is also a finding.
    assert app.health()["client_log"]["rejected"] == 1


def test_client_log_refuses_an_over_long_text_by_name(live_server):
    base, app = live_server
    status, body, _ = jrequest(f"{base}/api/client-log", "POST", {
        "events": [_client_event("x" * (ardy_viewer.CLIENT_LOG_MAX_TEXT + 1))],
    })
    assert status == 413
    assert body["error"] == "client_log_text_too_long"
    assert app.client_log_report()["received"] == 0

    # Exactly at the limit is accepted: the boundary is inclusive, and a page that
    # truncates to the documented number must not be refused for it.
    status, body, _ = jrequest(f"{base}/api/client-log", "POST", {
        "events": [_client_event("y" * ardy_viewer.CLIENT_LOG_MAX_TEXT)],
    })
    assert status == 200, body
    assert app.client_log_report()["received"] == 1


def test_client_log_refuses_its_own_body_limit_by_name(live_server):
    """A body over this route's 64 KiB ceiling is refused as *this route's* error.

    The shared 1 MiB refusal keeps its own name and its own test
    (`test_plan_rejects_an_oversized_body`); what matters here is that the tighter
    ceiling is named after the route rather than reported as the shared one, so a page
    reading ``{"error": ...}`` knows which limit it hit.
    """
    base, app = live_server
    filler = "z" * (ardy_viewer.CLIENT_LOG_MAX_BODY + 1024)
    status, body, _ = jrequest(f"{base}/api/client-log", "POST",
                               {"events": [], "snapshot": {"url": filler}})
    assert status == 413
    assert body["error"] == "client_log_body_too_large"
    assert str(ardy_viewer.CLIENT_LOG_MAX_BODY) in body["message"]
    assert app.client_log_report()["received"] == 0
    # The tighter ceiling is genuinely tighter, i.e. it is the one that fires first.
    assert ardy_viewer.CLIENT_LOG_MAX_BODY < (1 << 20)


def test_client_log_refuses_an_unknown_kind_by_name(live_server):
    base, app = live_server
    status, body, _ = jrequest(f"{base}/api/client-log", "POST",
                               {"events": [_client_event("hi", kind="console.log")]})
    assert status == 400
    assert body["error"] == "client_log_unknown_kind"
    assert "console.log" in body["message"]
    # The filter is the same closed vocabulary: an unknown name is a refusal, not an
    # empty list that would read as "there were none".
    status, body, _ = jrequest(f"{base}/api/client-log?kind=console.log")
    assert status == 400
    assert body["error"] == "client_log_unknown_kind"
    assert app.client_log_report()["received"] == 0


@pytest.mark.parametrize("snapshot, expected", [
    ("not an object", "client_log_invalid_snapshot"),
    ({"url": "u" * (ardy_viewer.CLIENT_LOG_MAX_SNAPSHOT + 1)}, "client_log_snapshot_too_large"),
])
def test_client_log_refuses_a_bad_snapshot_by_name(live_server, snapshot, expected):
    base, app = live_server
    status, body, _ = jrequest(f"{base}/api/client-log", "POST",
                               {"events": [], "snapshot": snapshot})
    assert status == 413 if expected.endswith("too_large") else status == 400
    assert body["error"] == expected
    assert app.client_log_report()["received"] == 0


def test_client_log_keeps_a_bounded_ring_and_counts_what_fell_out(live_server):
    base, app = live_server
    total = ardy_viewer.CLIENT_LOG_KEEP + 7
    sent = 0
    while sent < total:
        batch = min(ardy_viewer.CLIENT_LOG_MAX_EVENTS, total - sent)
        status, body, _ = jrequest(f"{base}/api/client-log", "POST", {
            "events": [_client_event(f"event {sent + i}") for i in range(batch)],
        })
        assert status == 200, body
        sent += batch
    report = app.client_log_report()
    assert report["received"] == total
    assert report["kept"] == ardy_viewer.CLIENT_LOG_KEEP
    assert report["dropped"] == total - ardy_viewer.CLIENT_LOG_KEEP
    # The ring drops the *oldest*, so the newest event is still there.
    assert report["events"][-1]["text"] == f"event {total - 1}"
    assert report["events"][0]["text"] == f"event {total - ardy_viewer.CLIENT_LOG_KEEP}"


def test_client_log_rejects_the_verbs_it_does_not_have(live_server):
    base, _ = live_server
    for method in ("PUT", "PATCH", "DELETE"):
        status, body, _ = jrequest(f"{base}/api/client-log", method, {"events": []})
        assert status == 405, method
        assert body["error"] == "method_not_allowed"
        assert "client-log" in body["message"]


def test_client_log_never_writes_a_secret_into_the_health_snapshot(live_server, monkeypatch, capsys):
    # The snapshot is echoed back by `/api/health` *and* written to stderr, so it is
    # the other place a key could land. It goes through the same redaction.
    monkeypatch.setenv("JEV_API_KEY", CLIENT_LOG_SENTINEL)
    base, _ = live_server
    jrequest(f"{base}/api/client-log", "POST", {
        "events": [], "snapshot": {"url": f"/?key={CLIENT_LOG_SENTINEL}", "playhead": 7},
    })
    err = capsys.readouterr().err
    status, health, _ = jrequest(f"{base}/api/health")
    assert status == 200
    assert CLIENT_LOG_SENTINEL not in err
    assert CLIENT_LOG_SENTINEL not in json.dumps(health)
    assert health["client_log"]["snapshot"]["playhead"] == 7
    assert "[redacted:env-secret]" in health["client_log"]["snapshot"]["url"]
