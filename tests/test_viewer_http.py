"""Viewer HTTP tests: routes, status codes, static-file containment, error names.

The whole point of the viewer's server design is that **one thread** owns MLX (the
GPU stream is thread-local). These tests exercise the real
:class:`ardy_viewer.ViewerServer` and :class:`ardy_viewer.ViewerHandler` over a
loopback socket, on an ephemeral port, with a fake runner standing in for the
checkpoint -- so the HTTP layer is real while nothing here loads 765 MB of weights
or evaluates on the GPU.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
import threading
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tests.test_viewer_api import CORE, FakeRunner, synthetic_frames  # noqa: E402


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader, f"cannot load {relative}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


viewer = _load("ardy_viewer_http_under_test", "scripts/ardy_viewer.py")

from ardy_mlx.sampling import pipeline  # noqa: E402

#: The HTTP tests run against the **real** Core layout (20 fps, 40-frame horizon),
#: so the duration grid the page sees is the one it sees in production. Only the
#: weights are faked.
CORE = viewer.PRESETS["ARDY-Core-RP-20FPS-Horizon40"]


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
@pytest.fixture()
def embeddings_dir(tmp_path):
    directory = tmp_path / "embeddings"
    directory.mkdir()
    text = "A person walks forward."
    digest = pipeline.prompt_hash(text)
    rng = np.random.default_rng(3)
    np.save(directory / f"{digest}.npy", rng.standard_normal((1, 1, 4096)).astype(np.float32))
    (directory / f"{digest}.json").write_text(json.dumps({"prompt": text}), encoding="utf-8")
    return directory


def _posed_joints_without_mlx(frames):
    """FK in numpy, so no test ever enters MLX from a server thread.

    Mirrors ``posed_joints_from_features`` for this one call shape; the real
    function is exercised on the main thread by ``tests/test_model_skeleton.py``
    and end-to-end by ``tests/test_viewer_service.py``. Its importance here is that
    the HTTP layer is tested without touching a thread-local GPU stream.
    """
    skeleton = viewer.core_skeleton()
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
    monkeypatch.setattr(viewer, "posed_joints_from_features", lambda frames, skeleton=None: _posed_joints_without_mlx(frames))
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
    monkeypatch.setattr(viewer, "VIEWER_DIR", static)

    app = viewer.ViewerApp(embeddings_dir=embeddings_dir)
    app.embeddings.refresh()
    app.runner = FakeRunner(layout=CORE)
    app.error = None
    app.boot_seconds = 0.005

    server = viewer.ViewerServer(("127.0.0.1", 0), app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    host, port = server.server_address[:2]
    try:
        yield f"http://{host}:{port}", app
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
    assert body["limits"]["text_encoder"].startswith("not ported")


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
    server = viewer.ViewerServer(("127.0.0.1", 0), object())
    try:
        mro = [klass.__name__ for klass in type(server).__mro__]
        assert "ThreadingHTTPServer" not in mro
        assert "ThreadingMixIn" not in [klass.__name__ for klass in type(server).__mro__]
        assert "HTTPServer" in mro
        # The handler is the same class for every request: no per-request worker.
        assert server.RequestHandlerClass is viewer.ViewerHandler
    finally:
        server.server_close()


def test_binding_off_loopback_is_refused(tmp_path, monkeypatch, capsys):
    """The viewer has no authentication; it must not be reachable off-host."""
    code = viewer.main(["--host", "0.0.0.0", "--port", "0", "--no-preload",
                        "--embeddings-dir", str(tmp_path)])
    assert code == 2
    assert "refusing to bind" in capsys.readouterr().err


def test_cpu_device_needs_an_explicit_opt_in(embeddings_dir):
    app = viewer.ViewerApp(embeddings_dir=embeddings_dir, device="cpu")
    app.start()
    assert app.runner is None
    assert "--allow-cpu" in (app.error or "")
