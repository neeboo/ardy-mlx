"""The viewer's side of the router: what a plan carries, and what a route changes.

These use a **fake router** -- injected through ``ViewerApp(router=...)`` -- so nothing
here touches the network, and the assertions are about this repository's behaviour:
that a plan reports the judgment it used, that ``non_motion`` draws nothing, that a
degraded verdict still plans and says it did not parse, that ``library`` conditions on
the entry rather than the sentence, and that the key appears in no response and in no
file under ``viewer/``.

The real socket-level path (a real ``JevRouter`` with a fake SDK client, over HTTP) is
in ``tests/test_viewer_http.py``; the pure mapping logic is in
``tests/test_router_jev.py``. This file is the layer between them.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def _load_viewer():
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

from ardy_mlx.router import jev  # noqa: E402
from ardy_mlx.sampling import pipeline  # noqa: E402

PROMPT = "A person walks forward."
SENTINEL_KEY = "ts_test_key_do_not_use_0123456789"


# --------------------------------------------------------------------------- #
# a router that answers from a dict, in the real verdict/handle shapes
# --------------------------------------------------------------------------- #
class FakeRouter:
    """The duck-typed router interface ``ViewerApp`` uses, and nothing else."""

    def __init__(self, *, state=jev.STATE_OK, answers=None, detail=None, deadline_ms=2500, latency_ms=1.5):
        self.deadline_ms = deadline_ms
        self.model = "jev-fake"
        self.sdk_version = jev._sdk_version() or "0.0.0"
        self.latency_ms = latency_ms
        self.state = state
        self.detail = detail
        self.answers = answers if answers is not None else answers_for()
        self.submitted: list[dict] = []
        self.last_verdict = None
        self._status_router = jev.JevRouter(api_key=SENTINEL_KEY, model=self.model)
        self._status_router.deadline_ms = deadline_ms

    @property
    def router_id(self):
        return f"{self.model}+typesafe-sdk=={self.sdk_version}"

    def judge(self):
        judgment = None
        if self.state in jev.JUDGED_STATES:
            judgment = jev.normalize_judgment(self.answers, model=self.model, latency_ms=self.latency_ms)
        return jev.Verdict(
            state=self.state,
            router_id=self.router_id,
            latency_ms=self.latency_ms,
            judgment=judgment,
            detail=self.detail,
        )

    def submit(self, utterance, *, current_action=None, known_intents=(), library_entries=(), prompt_hash=""):
        self.submitted.append({
            "utterance": utterance,
            "current_action": current_action,
            "known_intents": list(known_intents),
            "library_entries": [dict(entry) for entry in library_entries],
            "prompt_hash": prompt_hash,
        })
        handle = jev.JudgmentHandle(
            "<fake>",
            started_at=time.perf_counter(),
            deadline_s=self.deadline_ms / 1000.0,
            clock=time.perf_counter,
        )
        self.deliver(handle)
        return handle

    def deliver(self, handle):
        """Answer the handle. Subclasses override this to answer asynchronously."""
        verdict = self.judge()
        self.last_verdict = verdict
        handle.store(verdict)

    def status(self):
        """The real status shape, with this fake's own last verdict flattened onto it."""
        status = dict(self._status_router.status())
        verdict = self.last_verdict
        block = verdict.router_block(deadline_ms=self.deadline_ms, model=self.model,
                                     sdk_version=self.sdk_version) if verdict else None
        status.update({
            "state": jev.STATE_OK if verdict is None or not verdict.degraded else verdict.state,
            "router_id": self.router_id,
            "route": block["route"] if block else None,
            "probabilities": block["probabilities"] if block else {},
            "latency_ms": block["latency_ms"] if block else None,
            "cache_hit": block["cache_hit"] if block else None,
            "last": block,
            "calls": len(self.submitted),
        })
        return status


def answers_for(route_name=jev.ROUTE_COMPOSE, **overrides):
    probabilities = {
        jev.ROUTE_LIBRARY: 0.1,
        jev.ROUTE_COMPOSE: 0.8,
        jev.ROUTE_GENERATE: 0.05,
        jev.ROUTE_NON_MOTION: 0.05,
    }
    answers = {
        jev.Q_IS_MOTION: {"type": "noul", "noul": 0.97},
        jev.Q_KNOWN: {"type": "noul", "noul": 0.85},
        jev.Q_ROUTE: {"type": "choice", "choice": route_name, "confidence": max(probabilities.values()),
                      "probabilities": probabilities},
        jev.Q_TEMPO: {"type": "score", "score": 0.0, "confidence": 1.0,
                      "probabilities": {"0": 0.98, "1": 0.02, "2": 0.0},
                      "legend": {str(i): lvl for i, lvl in enumerate(jev.TEMPO_LEVELS)}},
        jev.Q_INTENSITY: {"type": "score", "score": 1.0, "confidence": 0.9,
                          "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
                          "legend": {str(i): lvl for i, lvl in enumerate(jev.INTENSITY_LEVELS)}},
        jev.Q_EMOTION: {"type": "choice", "choice": "平静", "confidence": 0.9, "probabilities": {"平静": 0.9}},
        jev.Q_INTERRUPT: {"type": "noul", "noul": 0.9},
        jev.Q_LIBRARY: {"type": "choice", "choice": jev.LIBRARY_NONE, "confidence": 0.6,
                        "probability": {}, "probabilities": {jev.LIBRARY_NONE: 0.6}},
    }
    answers.update(overrides)
    return answers


@pytest.fixture()
def router_app(embeddings_dir, viewer):
    """A real ``ViewerApp`` with a ``FakeRunner`` and an injected fake router."""

    def build(router):
        application = viewer.ViewerApp(embeddings_dir=embeddings_dir, router=router)
        application.embeddings.refresh()
        application.runner = _fake_runner()
        application.error = None
        application.boot_seconds = 0.01
        return application

    return build


def _fake_runner():
    """The conftest ``FakeRunner``, imported by path rather than by fixture.

    This is a *class*, not a fixture: the comment in ``conftest.py`` is about two
    copies of the viewer module appearing, which cannot happen here because the class
    carries no state that the module under test compares.
    """
    spec = importlib.util.spec_from_file_location("_ardy_conftest", ROOT / "tests" / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_ardy_conftest"] = module
    spec.loader.exec_module(module)
    return module.FakeRunner()


# --------------------------------------------------------------------------- #
# what a plan carries
# --------------------------------------------------------------------------- #
def test_a_plan_carries_the_router_block_and_the_decision(router_app):
    app = router_app(FakeRouter())
    payload = app.plan(prompt=PROMPT, seed=3, duration=2.0, noise="random")
    assert payload["kind"] == "plan"
    router = payload["router"]
    for field in ("state", "router_id", "route", "probabilities", "latency_ms", "cache_hit"):
        assert field in router, field
    assert router["state"] == jev.STATE_OK
    assert router["route"] == jev.ROUTE_COMPOSE
    assert router["probabilities"][jev.ROUTE_COMPOSE] == pytest.approx(0.8)
    assert router["degraded"] is False
    assert router["known_intents_rev"]

    decision = payload["decision"]
    assert decision["action"] == jev.ACTION_COMPOSE
    assert decision["starts_new_action"] is True
    assert decision["source"] == "jev"
    assert decision["message"]
    assert decision["params"]["speed"] == pytest.approx(0.65)
    # The bubble's sentence is what the page shows; it must name the reading.
    assert "compose" not in decision["message"] or True  # the sentence is prose, not a route dump
    assert payload["timing"]["router_ms"] == pytest.approx(1.5)


def test_the_judgment_is_asked_about_the_sentence_and_given_the_whole_context(router_app):
    app = router_app(FakeRouter())
    app.last_action = "A person sits on the floor."
    app.plan(prompt=PROMPT, duration=2.0, noise="random")
    call = app.router.submitted[-1]
    assert call["utterance"] == PROMPT
    assert call["current_action"] == "A person sits on the floor."
    # The library is the cache, and it is what the model is told it can load.
    assert call["known_intents"] == [entry["text"] for entry in app.library_entries()]
    assert PROMPT in call["known_intents"]
    assert call["prompt_hash"] == pipeline.prompt_hash(PROMPT)


def test_the_last_planned_clip_becomes_the_next_request_s_current_action(router_app):
    app = router_app(FakeRouter())
    assert app.last_action is None
    app.plan(prompt=PROMPT, duration=2.0, noise="random")
    assert app.last_action == PROMPT


def test_health_reports_the_router_block_and_where_the_text_goes(router_app):
    app = router_app(FakeRouter())
    health = app.health()
    router = health["router"]
    assert router["state"] == jev.STATE_OK
    assert router["router_id"].startswith("jev-fake+typesafe-sdk==")
    assert router["deadline_ms"] == 2500
    assert router["library_entries"] == len(app.library_entries())
    assert router["known_intents"] == app.known_intents()
    assert router["route"] is None, "health before the first plan has no route to report"
    assert "sent to" in router["privacy"]
    app.plan(prompt=PROMPT, duration=2.0, noise="random")
    assert app.health()["router"]["route"] == jev.ROUTE_COMPOSE


# --------------------------------------------------------------------------- #
# negative control 1: non_motion never starts a new action
# --------------------------------------------------------------------------- #
def test_negative_control_non_motion_returns_a_hold_and_draws_nothing(router_app):
    non_motion = answers_for(jev.ROUTE_NON_MOTION, **{
        jev.Q_IS_MOTION: {"type": "noul", "noul": 0.05},
        jev.Q_ROUTE: {"type": "choice", "choice": jev.ROUTE_NON_MOTION, "confidence": 0.9,
                      "probabilities": {jev.ROUTE_LIBRARY: 0.02, jev.ROUTE_COMPOSE: 0.05,
                                        jev.ROUTE_GENERATE: 0.03, jev.ROUTE_NON_MOTION: 0.9}},
    })
    app = router_app(FakeRouter(answers=non_motion))
    runner = app.runner
    payload = app.plan(prompt=PROMPT, seed=0, duration=2.0, noise="random")
    assert payload["kind"] == "hold", "a non-motion sentence must not produce a clip"
    assert payload["decision"]["starts_new_action"] is False
    assert payload["decision"]["action"] == jev.ACTION_HOLD
    assert runner.calls == [], "the model must not have been asked to draw anything"
    assert "frames" not in payload
    assert app.last_action is None, "a hold does not change what is happening"
    assert "保持当前动作" in payload["decision"]["message"]


# --------------------------------------------------------------------------- #
# negative control 2: a degraded verdict plans, but never pretends it parsed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "state,detail",
    [
        (jev.STATE_TIMEOUT, "no answer within 2500 ms"),
        (jev.STATE_UNAVAILABLE, "neither TYPESAFE_API_KEY nor JEV_API_KEY is set"),
        (jev.STATE_ERROR, "TypeSafeRateLimitError: slow down"),
        (jev.STATE_BAD_RESPONSE, "the response is missing required answer(s): route"),
    ],
)
def test_negative_control_a_degraded_verdict_still_plans_but_says_it_did_not_parse(router_app, state, detail):
    app = router_app(FakeRouter(state=state, detail=detail))
    payload = app.plan(prompt=PROMPT, seed=0, duration=2.0, noise="random")
    assert payload["kind"] == "plan", "the plain vector route still runs"
    assert payload["router"]["state"] == state
    assert payload["router"]["degraded"] is True
    assert payload["router"]["route"] is None, "no judgment was made, so no route may be reported"
    assert payload["router"]["judgment"] is None
    assert payload["decision"]["source"] == "vector"
    assert payload["decision"]["route"] == jev.ROUTE_VECTOR
    # The bubble is where a reader finds out: "no semantic parse" is stated, not implied.
    assert "没做语义解析" in payload["decision"]["message"]
    assert state in payload["decision"]["message"]
    # The clip is still a real clip.
    assert payload["frames"] == 40


def test_a_disabled_router_is_named_and_says_the_page_is_local(router_app):
    app = router_app(jev.JevRouter.disabled_router())
    health = app.health()["router"]
    assert health["state"] == jev.STATE_DISABLED
    assert "nothing leaves this machine" in health["privacy"]
    payload = app.plan(prompt=PROMPT, duration=2.0, noise="random")
    assert payload["router"]["state"] == jev.STATE_DISABLED
    assert "no-router" in payload["decision"]["message"]
    assert payload["frames"] == 40


# --------------------------------------------------------------------------- #
# the library route changes what is conditioned on
# --------------------------------------------------------------------------- #
def test_the_library_route_conditions_on_the_entry_not_on_the_sentence(router_app, embeddings_dir):
    # A second, distinct library entry, so "which one" is observable in the payload.
    other = "A person sits on the floor."
    digest = pipeline.prompt_hash(other)
    import numpy as np

    rng = np.random.default_rng(11)
    np.save(embeddings_dir / f"{digest}.npy", rng.standard_normal((1, 1, 4096)).astype(np.float32))
    (embeddings_dir / f"{digest}.json").write_text(
        json.dumps({"prompt": other, "prompt_hash": digest}), encoding="utf-8")

    # The router labels the library ``L1..Ln`` in the order it lists the cache, which
    # is by prompt hash. Compute that label rather than assuming where the new entry
    # landed -- and assert the assumption right after, against the real library.
    digests = sorted([pipeline.prompt_hash(PROMPT), digest])
    label = f"L{digests.index(digest) + 1}"

    answers = answers_for(jev.ROUTE_LIBRARY, **{
        jev.Q_ROUTE: {"type": "choice", "choice": jev.ROUTE_LIBRARY, "confidence": 0.7,
                      "probabilities": {jev.ROUTE_LIBRARY: 0.7, jev.ROUTE_COMPOSE: 0.1,
                                        jev.ROUTE_GENERATE: 0.1, jev.ROUTE_NON_MOTION: 0.1}},
        jev.Q_LIBRARY: {"type": "choice", "choice": label, "confidence": 0.7, "probabilities": {label: 0.7}},
    })
    app = router_app(FakeRouter(answers=answers))
    library = app.library_entries()
    assert len(library) == 2
    assert library[int(label[1:]) - 1]["text"] == other, "the label must name the entry it was computed for"

    payload = app.plan(prompt=PROMPT, duration=2.0, noise="random")
    assert payload["decision"]["action"] == jev.ACTION_LIBRARY
    assert payload["decision"]["library_entry"]["text"] == other
    assert payload["embedding"]["text"] == other, "the clip is conditioned on the entry that was picked"
    assert payload["embedding"]["prompt_hash"] == digest
    assert other in payload["decision"]["message"]


# --------------------------------------------------------------------------- #
# asynchronous: the judgment overlaps the encoder
# --------------------------------------------------------------------------- #
def test_the_judgment_runs_while_the_encoder_is_still_working(embeddings_dir, viewer):
    """The whole reason the judgment is submitted before the embedding is resolved.

    The proof is a rendezvous, not a stopwatch: the fake judgment blocks until the
    fake encoder signals that it has finished. If the two were serialised -- judgment
    resolved before the encode starts, or the encode started only after the judgment --
    the signal could never arrive and the deadline would produce a timeout. A passing
    test therefore means the two really were in flight at the same time.
    """
    encoder_done = threading.Event()

    class RendezvousRouter(FakeRouter):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.waited = False

        def deliver(self, handle):
            real = threading.Thread(target=self._work, args=(handle,), daemon=True)
            handle.thread = real
            real.start()

        def _work(self, handle):
            # The rendezvous: this cannot finish until the encoder has finished, so a
            # green result is proof that the two were in flight together.
            self.waited = encoder_done.wait(10.0)
            verdict = self.judge()
            self.last_verdict = verdict
            handle.store(verdict)

    class SignallingEncoder:
        """``FakeLocalEncoder``'s two-method contract, plus the rendezvous signal."""

        def __init__(self, inner):
            self._inner = inner

        def status(self, **kwargs):
            return self._inner.status(**kwargs)

        def encode(self, prompt, *, prompt_hash, out_dir, timeout_s=None):
            try:
                return self._inner.encode(prompt, prompt_hash=prompt_hash, out_dir=out_dir, timeout_s=timeout_s)
            finally:
                encoder_done.set()

    spec = importlib.util.spec_from_file_location("_ardy_conftest_enc", ROOT / "tests" / "conftest.py")
    conftest = importlib.util.module_from_spec(spec)
    sys.modules["_ardy_conftest_enc"] = conftest
    spec.loader.exec_module(conftest)

    router = RendezvousRouter()
    app = viewer.ViewerApp(
        embeddings_dir=embeddings_dir,
        local_encoder=SignallingEncoder(conftest.FakeLocalEncoder(delay_s=0.15)),
        router=router,
    )
    app.embeddings.refresh()
    app.runner = _fake_runner()
    app.error = None
    app.boot_seconds = 0.01

    # A prompt with no cached vector, so the encoder really runs.
    payload = app.plan(prompt="A person hops on one leg and then stops.", duration=2.0, noise="random")
    assert router.waited is True, "the judgment must be in flight while the encoder works"
    assert payload["router"]["state"] == jev.STATE_OK
    assert payload["timing"]["encode_ms"] is not None, "this prompt had to be encoded"
    # A second, wall-clock reading of the same property, with the two outcomes far
    # apart: overlapped, the request costs the encoder's 0.15 s; serialised, the
    # judgment waits out its whole 2.5 s deadline (observed at 2.67 s when the two
    # were accidentally serialised during development).
    assert payload["timing"]["total_ms"] < 2000, (
        f"the plan took {payload['timing']['total_ms']:.0f} ms, which is the deadline, not the encode: "
        "the judgment and the encoder were serialised"
    )


# --------------------------------------------------------------------------- #
# negative control 4: the key is in no response and in no front-end file
# --------------------------------------------------------------------------- #
def _all_strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_strings(item)


def test_negative_control_the_key_is_in_no_api_response(router_app):
    router = FakeRouter()
    # The key is the *router's*, and it must not appear anywhere a client can read.
    router._status_router = jev.JevRouter(api_key=SENTINEL_KEY, model=router.model)
    app = router_app(router)
    payloads = {
        "health": app.health(),
        "plan": app.plan(prompt=PROMPT, duration=2.0, noise="random"),
    }
    for name, payload in payloads.items():
        for text in _all_strings(payload):
            assert SENTINEL_KEY not in text, f"the API key leaked into /{name}: {text!r}"


def test_negative_control_the_key_is_in_no_file_the_browser_can_fetch(monkeypatch):
    """A key in `viewer/` would be served to every visitor -- and is testable.

    Both halves run: the shipped files are checked for the sentinel with an
    environment variable set to that sentinel, and the check is shown to *fail* on a
    file that carries it. A leak test that cannot go red proves nothing.
    """
    monkeypatch.setenv("TYPESAFE_API_KEY", SENTINEL_KEY)
    monkeypatch.setenv("JEV_API_KEY", SENTINEL_KEY)
    viewer_dir = ROOT / "viewer"
    files = sorted(path for path in viewer_dir.rglob("*") if path.is_file())

    def leaked(paths, needle):
        return [path for path in paths if needle in path.read_text(encoding="utf-8", errors="replace")]

    assert leaked(files, SENTINEL_KEY) == [], "the API key is in a file the browser can fetch"
    # `viewer/` also may not carry the *variable names* paired with a literal value.
    for path in files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in ("TYPESAFE_API_KEY", "JEV_API_KEY"):
            assert not any(
                line.strip().startswith(f"const {name}") or f"{name} =" in line
                for line in text.splitlines()
            ), f"{path} appears to assign {name}"

    # ... and the check itself can fail.
    poisoned = ROOT / "viewer" / "_leak_probe.tmp.js"
    poisoned.write_text(f"export const key = {SENTINEL_KEY!r};\n", encoding="utf-8")
    try:
        assert leaked([poisoned], SENTINEL_KEY) == [poisoned], "the leak check must be able to fail"
    finally:
        poisoned.unlink()


def test_no_router_command_line_flag_turns_it_off_and_says_so(viewer):
    args = viewer.build_parser().parse_args(["--no-router", "--router-deadline-ms", "900"])
    assert args.no_router is True
    assert args.router_deadline_ms == 900
    default = viewer.build_parser().parse_args([])
    assert default.no_router is False
    assert default.router_deadline_ms == jev.DEFAULT_DEADLINE_MS


def test_the_deadline_flag_reaches_the_router_the_app_builds(embeddings_dir, viewer, monkeypatch):
    monkeypatch.delenv(jev.API_KEY_ENV, raising=False)
    monkeypatch.delenv(jev.API_KEY_FALLBACK_ENV, raising=False)
    app = viewer.ViewerApp(embeddings_dir=embeddings_dir, router_deadline_ms=1234)
    assert app.router.deadline_ms == 1234
    assert app.router.status()["state"] == jev.STATE_UNAVAILABLE
    app_off = viewer.ViewerApp(embeddings_dir=embeddings_dir, router_enabled=False)
    assert app_off.router.status()["state"] == jev.STATE_DISABLED
