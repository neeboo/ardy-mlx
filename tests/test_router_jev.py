"""The router: question set, mapping table, thresholds, degradation, cache, key hygiene.

Everything here runs **without a network**. The SDK is replaced by a fake client that
returns the same shapes, so the parts under test are this repository's own decisions:
which questions are asked, how an answer becomes an action, what happens when there is
no answer, and what a key may never appear in. The handful of assertions that need the
real SDK skip, with a reason, when ``typesafe-sdk`` is not installed.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from ardy_mlx.router import jev  # noqa: E402

#: A key that is obviously not a real one, used to prove it never escapes the router.
SENTINEL_KEY = "ts_test_key_do_not_use_0123456789"

HAVE_SDK = importlib.util.find_spec("typesafe_sdk") is not None
needs_sdk = pytest.mark.skipif(not HAVE_SDK, reason="typesafe-sdk is not installed; the real answer objects cannot be built")


# --------------------------------------------------------------------------- #
# fakes: the SDK's shape, none of its network
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, answers, model="jev-fake"):
        self.answers = answers
        self.model = model


class FakeClient:
    """A ``TypeSafeClient`` that answers from a dict and records what it was asked."""

    def __init__(self, answers=None, *, delay_s=0.0, error=None, model="jev-fake"):
        self.answers = answers if answers is not None else {}
        self.delay_s = delay_s
        self.error = error
        self.model = model
        self.calls = []

    def system_one(self, state, questions):
        self.calls.append({"state": state, "questions": questions})
        if self.delay_s:
            time.sleep(self.delay_s)
        if self.error is not None:
            raise self.error
        return FakeResponse(self.answers, model=self.model)


def answers_for(
    route="compose",
    *,
    route_probabilities=None,
    is_motion=0.97,
    known=0.85,
    tempo=1.0,
    intensity=1.0,
    emotion="平静",
    interrupt=0.9,
    library_pick=jev.LIBRARY_NONE,
):
    """A complete, well-formed set of answers, as plain dicts."""
    probabilities = route_probabilities or {
        jev.ROUTE_LIBRARY: 0.1,
        jev.ROUTE_COMPOSE: 0.8,
        jev.ROUTE_GENERATE: 0.05,
        jev.ROUTE_NON_MOTION: 0.05,
    }
    return {
        jev.Q_IS_MOTION: {"type": "noul", "noul": is_motion},
        jev.Q_KNOWN: {"type": "noul", "noul": known},
        jev.Q_ROUTE: {
            "type": "choice",
            "choice": route,
            "confidence": max(probabilities.values()),
            "probabilities": probabilities,
        },
        jev.Q_TEMPO: {
            "type": "score",
            "score": tempo,
            "confidence": 0.9,
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
            "legend": {str(i): level for i, level in enumerate(jev.TEMPO_LEVELS)},
        },
        jev.Q_INTENSITY: {
            "type": "score",
            "score": intensity,
            "confidence": 0.9,
            "probabilities": {"0": 0.1, "1": 0.8, "2": 0.1},
            "legend": {str(i): level for i, level in enumerate(jev.INTENSITY_LEVELS)},
        },
        jev.Q_EMOTION: {"type": "choice", "choice": emotion, "confidence": 0.7, "probabilities": {emotion: 0.7}},
        jev.Q_INTERRUPT: {"type": "noul", "noul": interrupt},
        jev.Q_LIBRARY: {
            "type": "choice",
            "choice": library_pick,
            "confidence": 0.6,
            "probabilities": {library_pick: 0.6},
        },
    }


def judgment_for(*args, **kwargs):
    return jev.normalize_judgment(answers_for(*args, **kwargs), model="jev-fake", latency_ms=1.0)


LIBRARY = [
    {"prompt_hash": "e9cf8141cac79f35", "text": "A person walks forward."},
    {"prompt_hash": "939c209a6e5979af", "text": "A person walks forward and waves."},
]


def make_router(answers=None, *, library=LIBRARY, **client_kwargs):
    client = FakeClient(answers if answers is not None else answers_for(), **client_kwargs)
    router = jev.JevRouter(
        api_key=SENTINEL_KEY,
        client_factory=lambda key, model, deadline: client,
        deadline_ms=client_kwargs.pop("deadline_ms", jev.DEFAULT_DEADLINE_MS),
    )
    return router, client


# --------------------------------------------------------------------------- #
# the state and the question set
# --------------------------------------------------------------------------- #
def test_the_state_uses_named_fields_and_passes_the_library_verbatim():
    state = jev.build_state(
        "往前走",
        current_action="A person sits on the floor.",
        known_intents=["A person walks forward."],
        library_entries=LIBRARY,
    )
    assert state["utterance"] == "往前走"
    assert state["context"]["current_action"] == "A person sits on the floor."
    assert state["context"]["known_intents"] == ["A person walks forward."]
    assert state["context"]["library_entries"] == LIBRARY


def test_an_empty_library_is_passed_as_an_empty_list_not_omitted():
    # An omitted field reads as "no information"; an empty list says "the library is
    # empty", which is the fact that is true and the one a route decision needs.
    state = jev.build_state("往前走", library_entries=[])
    assert state["context"]["library_entries"] == []
    assert "library_entries" in state["context"]


def test_each_question_asks_one_thing_and_every_closed_set_keeps_a_way_out():
    questions = jev.build_questions(LIBRARY)
    assert list(questions) == [
        jev.Q_IS_MOTION, jev.Q_KNOWN, jev.Q_ROUTE, jev.Q_TEMPO,
        jev.Q_INTENSITY, jev.Q_EMOTION, jev.Q_INTERRUPT, jev.Q_LIBRARY,
    ]
    # The primitive each question was built with, read off the SDK objects themselves.
    kinds = {name: question.type for name, question in questions.items()}
    assert kinds == {
        jev.Q_IS_MOTION: "noul",
        jev.Q_KNOWN: "noul",
        jev.Q_ROUTE: "choice",
        jev.Q_TEMPO: "score",
        jev.Q_INTENSITY: "score",
        jev.Q_EMOTION: "choice",
        jev.Q_INTERRUPT: "noul",
        jev.Q_LIBRARY: "choice",
    }
    # The route is the four known behaviours, and "not motion at all" is one of them
    # rather than an absence: a question with no exit forces a wrong answer.
    assert list(questions[jev.Q_ROUTE].criteria) == list(jev.ROUTES)
    assert jev.ROUTE_NON_MOTION in questions[jev.Q_ROUTE].criteria
    # The library lookup can say "none of these" too.
    assert list(questions[jev.Q_LIBRARY].criteria)[-1] == jev.LIBRARY_NONE
    assert len(questions[jev.Q_LIBRARY].criteria) == len(LIBRARY) + 1
    # Every question states its own judgment; nothing is asked by implication.
    for name, question in questions.items():
        assert question.instructions, f"{name} has no instructions"
    assert len(questions[jev.Q_TEMPO].criteria) == len(jev.TEMPO_LEVELS)
    assert len(questions[jev.Q_INTENSITY].criteria) == len(jev.INTENSITY_LEVELS)
    assert list(questions[jev.Q_EMOTION].criteria) == list(jev.EMOTIONS)


def test_the_library_choice_describes_each_entry_without_leaking_a_hash_as_a_label():
    questions = jev.build_questions(LIBRARY)
    criteria = questions[jev.Q_LIBRARY].criteria
    assert criteria["L1"]["text"] == LIBRARY[0]["text"]
    assert criteria["L1"]["prompt_hash"] == LIBRARY[0]["prompt_hash"]
    assert "e9cf8141cac79f35" not in criteria, "a hash is not a label the model can reason about"


def test_a_library_with_no_entries_still_produces_a_usable_choice():
    questions = jev.build_questions([])
    assert list(questions[jev.Q_LIBRARY].criteria) == [jev.LIBRARY_NONE]


# --------------------------------------------------------------------------- #
# the mapping table
# --------------------------------------------------------------------------- #
def test_every_route_maps_to_exactly_one_action():
    assert set(jev.ROUTE_ACTIONS) == set(jev.ROUTES)
    assert jev.ROUTE_ACTIONS[jev.ROUTE_NON_MOTION] == jev.ACTION_HOLD
    assert jev.ROUTE_ACTIONS[jev.ROUTE_LIBRARY] == jev.ACTION_LIBRARY


def test_tempo_and_intensity_are_monotone_and_neutral_at_the_middle_level():
    assert jev.tempo_to_speed(0.0) < jev.tempo_to_speed(1.0) < jev.tempo_to_speed(2.0)
    assert jev.tempo_to_speed(1.0) == pytest.approx(1.0)
    assert jev.intensity_to_amplitude(0.0) < jev.intensity_to_amplitude(1.0) < jev.intensity_to_amplitude(2.0)
    assert jev.intensity_to_amplitude(1.0) == pytest.approx(1.0)
    # A missing or impossible score is neutral, never a silent extreme.
    assert jev.tempo_to_speed(None) == 1.0
    assert jev.intensity_to_amplitude(None) == 1.0
    assert jev.tempo_to_speed(99.0) == jev.tempo_to_speed(2.0)


def test_the_conservative_clamp_only_ever_slows_down_or_shrinks():
    assert jev.tempo_to_speed(2.0, conservative=True) <= 1.0
    assert jev.intensity_to_amplitude(2.0, conservative=True) <= 1.0
    assert jev.tempo_to_speed(0.0, conservative=True) == jev.tempo_to_speed(0.0)


def test_the_library_route_loads_the_entry_it_picked():
    decision = jev.decide(judgment_for(jev.ROUTE_LIBRARY, library_pick="L1"), state=jev.STATE_OK, library_entries=LIBRARY)
    assert decision.action == jev.ACTION_LIBRARY
    assert decision.starts_new_action is True
    assert decision.library_entry == LIBRARY[0]
    assert LIBRARY[0]["text"] in decision.message


def test_a_library_route_with_no_usable_pick_degrades_to_composing_and_says_so():
    decision = jev.decide(judgment_for(jev.ROUTE_LIBRARY, library_pick=jev.LIBRARY_NONE), state=jev.STATE_OK, library_entries=LIBRARY)
    assert decision.action == jev.ACTION_COMPOSE
    assert decision.conservative is True
    assert "没指出是哪一条" in decision.message
    assert decision.library_entry is None


def test_a_pick_by_text_or_hash_is_still_understood():
    by_text = jev.decide(judgment_for(jev.ROUTE_LIBRARY, library_pick=LIBRARY[1]["text"]), state=jev.STATE_OK, library_entries=LIBRARY)
    assert by_text.library_entry == LIBRARY[1]
    by_hash = jev.decide(judgment_for(jev.ROUTE_LIBRARY, library_pick=LIBRARY[0]["prompt_hash"]), state=jev.STATE_OK, library_entries=LIBRARY)
    assert by_hash.library_entry == LIBRARY[0]


def test_the_compose_route_reports_the_knobs_it_derived():
    decision = jev.decide(
        judgment_for(jev.ROUTE_COMPOSE, tempo=0.0, intensity=2.0), state=jev.STATE_OK, library_entries=LIBRARY
    )
    assert decision.action == jev.ACTION_COMPOSE
    assert decision.params["speed"] == pytest.approx(0.65)
    assert decision.params["amplitude"] == pytest.approx(1.4)
    assert decision.params["tempo"]["level"] == jev.TEMPO_LEVELS[0]
    assert decision.params["intensity"]["level"] == jev.INTENSITY_LEVELS[2]
    assert decision.params["expression_placeholder"] is True, "the rig has no face; say so in the payload"


def test_the_emotion_choice_becomes_a_placeholder_expression_and_gaze():
    decision = jev.decide(judgment_for(jev.ROUTE_COMPOSE, emotion="低落"), state=jev.STATE_OK, library_entries=LIBRARY)
    assert decision.params["emotion"] == "低落"
    assert (decision.params["expression"], decision.params["gaze"]) == jev.EMOTION_PLACEHOLDERS["低落"]
    unknown = jev.decide(judgment_for(jev.ROUTE_COMPOSE, emotion="not-a-known-emotion"), state=jev.STATE_OK)
    assert unknown.params["emotion"] == "not-a-known-emotion"
    assert unknown.params["expression"] == jev.EMOTION_PLACEHOLDERS[jev.DEFAULT_EMOTION][0]


def test_interrupt_decides_whether_the_new_action_replaces_the_current_one():
    yes = jev.decide(judgment_for(jev.ROUTE_COMPOSE, interrupt=0.9), state=jev.STATE_OK)
    no = jev.decide(judgment_for(jev.ROUTE_COMPOSE, interrupt=0.1), state=jev.STATE_OK)
    assert yes.params["replaces_current"] is True
    assert no.params["replaces_current"] is False


# --------------------------------------------------------------------------- #
# negative controls
# --------------------------------------------------------------------------- #
def test_negative_control_non_motion_never_starts_a_new_action():
    """The one outcome this system must never produce is unasked-for motion.

    A `non_motion` reading is a **refusal**, at any probability, with any other answer
    saying otherwise -- so the assertion is on the whole decision, not on the message.
    """
    for probability in (0.5, 0.6, 0.9, 0.99):
        judgment = judgment_for(
            jev.ROUTE_NON_MOTION,
            route_probabilities={jev.ROUTE_LIBRARY: 0.02, jev.ROUTE_COMPOSE: 0.03,
                                 jev.ROUTE_GENERATE: 0.05, jev.ROUTE_NON_MOTION: probability},
            is_motion=0.99,  # even with `is_motion` screaming yes
            interrupt=1.0,
        )
        decision = jev.decide(judgment, state=jev.STATE_OK, library_entries=LIBRARY)
        assert decision.action == jev.ACTION_HOLD, probability
        assert decision.starts_new_action is False, probability
        assert decision.library_entry is None


def test_negative_control_a_sentence_that_is_not_a_motion_request_never_starts_one():
    judgment = judgment_for(jev.ROUTE_COMPOSE, is_motion=0.05)
    decision = jev.decide(judgment, state=jev.STATE_OK, library_entries=LIBRARY)
    assert decision.action == jev.ACTION_HOLD
    assert decision.starts_new_action is False
    assert "is_motion" in decision.reason


def test_negative_control_a_low_confidence_reading_takes_the_conservative_branch_and_says_so():
    # Probability split between the top routes, and a confidence below the floor.
    spread = {jev.ROUTE_LIBRARY: 0.34, jev.ROUTE_COMPOSE: 0.33, jev.ROUTE_GENERATE: 0.2, jev.ROUTE_NON_MOTION: 0.13}
    judgment = judgment_for(jev.ROUTE_COMPOSE, route_probabilities=spread, tempo=2.0, intensity=2.0)
    decision = jev.decide(judgment, state=jev.STATE_OK, library_entries=LIBRARY)
    assert decision.conservative is True
    assert decision.message.startswith(jev.CONSERVATIVE_PREFIX), decision.message
    # ... and the parameters are clamped to the gentlest reading rather than the raw one.
    assert decision.params["speed"] <= 1.0
    assert decision.params["amplitude"] <= 1.0
    assert "conservative" in decision.reason


def test_a_low_known_answer_keeps_the_route_but_admits_the_engine_is_improvising():
    decision = jev.decide(judgment_for(jev.ROUTE_COMPOSE, known=0.2), state=jev.STATE_OK, library_entries=LIBRARY)
    assert decision.conservative is True
    assert decision.message.endswith(jev.IMPROVISING_SUFFIX)
    assert decision.starts_new_action is True


def test_a_confident_reading_is_not_flagged_conservative():
    decision = jev.decide(
        judgment_for(jev.ROUTE_COMPOSE, route_probabilities={jev.ROUTE_LIBRARY: 0.05, jev.ROUTE_COMPOSE: 0.9,
                                                             jev.ROUTE_GENERATE: 0.03, jev.ROUTE_NON_MOTION: 0.02}),
        state=jev.STATE_OK,
        library_entries=LIBRARY,
    )
    assert decision.conservative is False
    assert not decision.message.startswith(jev.CONSERVATIVE_PREFIX)


@pytest.mark.parametrize("state", jev.DEGRADED_STATES)
def test_negative_control_a_degraded_verdict_never_claims_a_semantic_parse(state):
    """A timeout, a missing key or a malformed answer must not look like a judgment.

    The verdict here carries a *perfect* judgment alongside a degraded state, which is
    exactly the accident that would let a fallback be rendered as a reading.
    """
    decision = jev.decide(judgment_for(jev.ROUTE_LIBRARY, library_pick="L1"), state=state, detail="caused by the test")
    assert decision.action == jev.ACTION_VECTOR
    assert decision.route == jev.ROUTE_VECTOR
    assert decision.source == "vector"
    assert decision.library_entry is None
    assert decision.starts_new_action is True, "the plain vector route still runs; it just is not a judgment"
    assert state in decision.message or state in decision.reason


def test_a_timeout_names_itself_and_the_deadline_it_missed():
    decision = jev.decide(None, state=jev.STATE_TIMEOUT, deadline_ms=2500, detail="no answer within 2500 ms")
    assert jev.STATE_TIMEOUT in decision.message
    assert "2500" in decision.message
    assert "没做语义解析" in decision.message
    assert decision.params["speed"] == 1.0 and decision.params["amplitude"] == 1.0


def test_an_unknown_state_is_a_programming_error_not_a_route():
    with pytest.raises(ValueError):
        jev.decide(None, state="ok_ish")
    with pytest.raises(ValueError):
        jev.decide(None, state=jev.STATE_OK)  # a judged state must carry a judgment


# --------------------------------------------------------------------------- #
# reading a real response
# --------------------------------------------------------------------------- #
def test_normalize_judgment_rejects_a_response_that_is_missing_a_required_answer():
    answers = answers_for()
    del answers[jev.Q_ROUTE]
    with pytest.raises(jev.BadJudgment):
        jev.normalize_judgment(answers)
    del answers[jev.Q_IS_MOTION]
    with pytest.raises(jev.BadJudgment):
        jev.normalize_judgment(answers)


def test_normalize_judgment_rejects_a_route_that_is_not_one_of_the_four():
    answers = answers_for()
    answers[jev.Q_ROUTE]["choice"] = "fly"
    with pytest.raises(jev.BadJudgment):
        jev.normalize_judgment(answers)


def test_normalize_judgment_rejects_an_answer_kind_it_does_not_model():
    with pytest.raises(jev.BadJudgment):
        jev.normalize_judgment({jev.Q_IS_MOTION: {"type": "vibe", "value": 1},
                                jev.Q_KNOWN: {"type": "noul", "noul": 1.0},
                                jev.Q_ROUTE: answers_for()[jev.Q_ROUTE]})


def test_score_probabilities_and_legend_are_keyed_by_string_for_json():
    judgment = judgment_for(jev.ROUTE_COMPOSE, tempo=1.0)
    tempo = judgment[jev.Q_TEMPO]
    assert set(tempo["probabilities"]) == {"0", "1", "2"}
    assert tempo["legend"]["1"] == jev.TEMPO_LEVELS[1]
    assert json.loads(json.dumps(judgment))  # JSON-safe, round trip included


@needs_sdk
def test_normalize_judgment_reads_the_real_sdk_answer_objects():
    """The one test that uses the shipped SDK types: the field names must still line up."""
    from typesafe_sdk import ChoiceAnswer, NoulAnswer, ScoreAnswer

    answers = {
        jev.Q_IS_MOTION: NoulAnswer(type="noul", noul=0.97),
        jev.Q_KNOWN: NoulAnswer(type="noul", noul=0.85),
        jev.Q_ROUTE: ChoiceAnswer(type="choice", choice="compose", confidence=0.83,
                                  probabilities={"library": 0.11, "compose": 0.83, "generate": 0.06, "non_motion": 0.0}),
        jev.Q_TEMPO: ScoreAnswer(type="score", score=0.0, confidence=1.0,
                                 legend={0: "slow and relaxed", 1: "normal pace", 2: "hurried"},
                                 probabilities={0: 0.98, 1: 0.02, 2: 0.0}),
    }
    judgment = jev.normalize_judgment(answers, model="jev-latest", latency_ms=12.5)
    assert judgment[jev.Q_ROUTE]["choice"] == "compose"
    assert judgment[jev.Q_TEMPO]["score"] == 0.0
    # The SDK keys these by integer; the payload must be JSON-keyed by string.
    assert judgment[jev.Q_TEMPO]["probabilities"] == {"0": 0.98, "1": 0.02, "2": 0.0}
    decision = jev.decide(judgment, state=jev.STATE_OK)
    assert decision.action == jev.ACTION_COMPOSE


# --------------------------------------------------------------------------- #
# the cache
# --------------------------------------------------------------------------- #
def test_the_cache_key_is_the_prompt_the_router_and_the_intents_revision():
    rev = jev.known_intents_revision(["a"], LIBRARY)
    key = jev.cache_key("abc123", "jev-latest+typesafe-sdk==0.7.4", rev)
    assert key == f"abc123|jev-latest+typesafe-sdk==0.7.4|{rev}"
    # A different router is a different key, even for the same sentence.
    assert jev.cache_key("abc123", "other", rev) != key
    # ... and so is a different library.
    assert jev.known_intents_revision(["a"], LIBRARY) != jev.known_intents_revision(["b"], LIBRARY)
    assert jev.known_intents_revision(["a"], LIBRARY) != jev.known_intents_revision(["a"], LIBRARY[:1])
    # The same inputs give the same revision, regardless of dict ordering.
    reordered = [dict(reversed(list(entry.items()))) for entry in LIBRARY]
    assert jev.known_intents_revision(["a"], reordered) == rev
    # An unavailable router has no id, and that is still a stable key.
    assert jev.cache_key("abc123", None, rev).startswith("abc123|-|")


def test_the_same_sentence_is_judged_once_and_then_answered_from_the_cache():
    router, client = make_router()
    first = router.submit("往前走", library_entries=LIBRARY, known_intents=["a"], prompt_hash="abc123").resolve()
    second = router.submit("往前走", library_entries=LIBRARY, known_intents=["a"], prompt_hash="abc123").resolve()
    assert first.state == jev.STATE_OK and first.cache_hit is False
    assert second.state == jev.STATE_CACHED and second.cache_hit is True
    assert len(client.calls) == 1, "a cached sentence must not cost a second call"
    assert second.latency_ms == 0.0
    assert second.judgment == first.judgment
    assert second.route == first.route


def test_a_different_library_invalidates_the_cached_judgment():
    router, client = make_router()
    router.submit("往前走", library_entries=LIBRARY, known_intents=["a"], prompt_hash="abc").resolve()
    router.submit("往前走", library_entries=LIBRARY[:1], known_intents=["a"], prompt_hash="abc").resolve()
    assert len(client.calls) == 2, "route and library_pick were chosen against the library, so it is part of the key"


def test_a_call_that_outlives_its_deadline_still_fills_the_cache():
    """A timeout is a fact about *this* request, not a reason to pay twice."""
    client = FakeClient(answers_for(), delay_s=0.35)
    router = jev.JevRouter(api_key=SENTINEL_KEY, deadline_ms=40,
                           client_factory=lambda key, model, deadline: client)
    timed_out = router.submit("慢慢走过去", library_entries=LIBRARY, prompt_hash="slow").resolve()
    assert timed_out.state == jev.STATE_TIMEOUT
    assert timed_out.latency_ms is not None and timed_out.latency_ms < 350
    # The worker is still running; wait for it, then ask again.
    time.sleep(0.6)
    assert router.status()["timeouts"] == 1
    cached = router.submit("慢慢走过去", library_entries=LIBRARY, prompt_hash="slow").resolve()
    assert cached.state == jev.STATE_CACHED
    assert len(client.calls) == 1


def test_the_cache_is_bounded_and_evicts_the_oldest_sentence():
    router, client = make_router()
    router.cache_size = 2
    for index in range(3):
        router.submit(f"sentence {index}", library_entries=LIBRARY, prompt_hash=f"p{index}").resolve()
    assert router.status()["cache_size"] == 2
    assert len(client.calls) == 3
    # The first one was evicted, so it is asked again.
    router.submit("sentence 0", library_entries=LIBRARY, prompt_hash="p0").resolve()
    assert len(client.calls) == 4


# --------------------------------------------------------------------------- #
# asynchronous, and honest about its deadline
# --------------------------------------------------------------------------- #
def test_the_call_runs_off_the_calling_thread_and_the_caller_only_waits_the_deadline():
    client = FakeClient(answers_for(), delay_s=0.3)
    router = jev.JevRouter(api_key=SENTINEL_KEY, deadline_ms=2000,
                           client_factory=lambda key, model, deadline: client)
    started = time.perf_counter()
    handle = router.submit("往前走", library_entries=LIBRARY, prompt_hash="x")
    submitted = time.perf_counter() - started
    assert submitted < 0.1, "submit() must return immediately; the judgment is not on this thread"
    assert handle.pending is True
    assert handle.resolve().state == jev.STATE_OK
    assert time.perf_counter() - started >= 0.3


def test_a_slow_call_is_named_a_timeout_rather_than_waited_out():
    client = FakeClient(answers_for(), delay_s=1.0)
    router = jev.JevRouter(api_key=SENTINEL_KEY, deadline_ms=80,
                           client_factory=lambda key, model, deadline: client)
    started = time.perf_counter()
    verdict = router.submit("往前走", library_entries=LIBRARY, prompt_hash="x").resolve()
    elapsed = time.perf_counter() - started
    assert verdict.state == jev.STATE_TIMEOUT
    assert elapsed < 0.5, "the deadline is the contract; the wait is bounded by it"
    assert "80 ms" in verdict.detail
    assert verdict.degraded is True


# --------------------------------------------------------------------------- #
# named degradations
# --------------------------------------------------------------------------- #
def test_a_missing_key_is_named_and_no_client_is_ever_built(monkeypatch):
    for name in (jev.API_KEY_ENV, jev.API_KEY_FALLBACK_ENV):
        monkeypatch.delenv(name, raising=False)
    built = []

    def factory(*args):
        built.append(args)
        return FakeClient(answers_for())

    router = jev.JevRouter(client_factory=factory)
    assert router.available is False
    assert router.router_id is None
    verdict = router.submit("往前走", library_entries=LIBRARY).resolve()
    assert verdict.state == jev.STATE_UNAVAILABLE
    assert verdict.degraded is True
    assert jev.API_KEY_ENV in verdict.detail and jev.API_KEY_FALLBACK_ENV in verdict.detail
    assert built == [], "a router with no key must not construct a client"
    assert router.status()["state"] == jev.STATE_UNAVAILABLE


def test_a_disabled_router_says_disabled_rather_than_unavailable():
    router = jev.JevRouter.disabled_router()
    verdict = router.submit("往前走", library_entries=LIBRARY).resolve()
    assert verdict.state == jev.STATE_DISABLED
    assert router.status()["enabled"] is False
    assert router.status()["state"] == jev.STATE_DISABLED
    decision = jev.decide(None, state=verdict.state)
    assert decision.source == "vector"
    assert "no-router" in decision.message


def test_a_request_with_no_text_to_judge_is_named_rather_than_judged():
    router, client = make_router()
    verdict = router.submit(None, library_entries=LIBRARY).resolve()
    assert verdict.state == jev.STATE_NO_UTTERANCE
    verdict = router.submit("   ", library_entries=LIBRARY).resolve()
    assert verdict.state == jev.STATE_NO_UTTERANCE
    assert client.calls == [], "a hex hash is not a sentence; nothing is sent for one"


@pytest.mark.parametrize(
    "error_name,expected",
    [
        ("TypeSafeAPITimeoutError", jev.STATE_TIMEOUT),
        ("TypeSafeAuthenticationError", jev.STATE_UNAVAILABLE),
        ("TypeSafeAPIConnectionError", jev.STATE_UNAVAILABLE),
        ("TypeSafeRateLimitError", jev.STATE_ERROR),
        ("TypeSafeInternalServerError", jev.STATE_ERROR),
    ],
)
def test_each_sdk_failure_is_classified_by_its_own_name(error_name, expected):
    error = type(error_name, (Exception,), {})("the SDK refused")
    router, _ = make_router(error=error)
    verdict = router.submit("往前走", library_entries=LIBRARY, prompt_hash="x").resolve()
    assert verdict.state == expected
    assert verdict.judgment is None
    assert verdict.degraded is True


def test_a_failure_whose_message_merely_says_timeout_is_not_a_timeout():
    """Classification is by exception name, not by a word that happens to appear."""
    error = type("TypeSafeBadRequestError", (Exception,), {})("the timeout field is malformed")
    router, _ = make_router(error=error)
    verdict = router.submit("往前走", library_entries=LIBRARY, prompt_hash="x").resolve()
    assert verdict.state == jev.STATE_ERROR


def test_a_malformed_response_is_named_rather_than_defaulted_into_a_route():
    router, _ = make_router(answers={jev.Q_IS_MOTION: {"type": "noul", "noul": 1.0}})
    verdict = router.submit("往前走", library_entries=LIBRARY, prompt_hash="x").resolve()
    assert verdict.state == jev.STATE_BAD_RESPONSE
    assert "route" in verdict.detail
    decision = jev.decide(None, state=verdict.state, detail=verdict.detail)
    assert decision.source == "vector"


# --------------------------------------------------------------------------- #
# the key
# --------------------------------------------------------------------------- #
def test_typesafe_api_key_wins_and_jev_api_key_is_mapped_onto_the_same_slot():
    assert jev.resolve_api_key({jev.API_KEY_ENV: "primary", jev.API_KEY_FALLBACK_ENV: "fallback"}) == (
        "primary", jev.API_KEY_ENV)
    assert jev.resolve_api_key({jev.API_KEY_FALLBACK_ENV: "fallback"}) == ("fallback", jev.API_KEY_FALLBACK_ENV)
    assert jev.resolve_api_key({}) == (None, None)
    # Blank and whitespace-only values are not keys.
    assert jev.resolve_api_key({jev.API_KEY_ENV: "   "}) == (None, None)
    assert jev.resolve_api_key({jev.API_KEY_ENV: "  spaced  "}) == ("spaced", jev.API_KEY_ENV)


def _all_strings(value):
    """Every string anywhere inside a JSON-ish structure."""
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _all_strings(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _all_strings(item)


def test_negative_control_the_key_is_in_no_output_the_router_can_produce():
    """The key enters here and must not leave in any payload, status or verdict."""
    router, _ = make_router()
    verdict = router.submit("往前走", library_entries=LIBRARY, prompt_hash="x").resolve()
    block = verdict.router_block(deadline_ms=router.deadline_ms, model=router.model, sdk_version=router.sdk_version)
    payloads = {
        "verdict": verdict.as_dict(),
        "router_block": block,
        "status": router.status(),
        "decision": jev.decide(verdict.judgment, state=verdict.state).as_dict(),
    }
    for name, payload in payloads.items():
        serialized = json.dumps(payload, ensure_ascii=False, default=str)
        assert SENTINEL_KEY not in serialized, f"the API key leaked into {name}"
        for text in _all_strings(payload):
            assert SENTINEL_KEY not in text, f"the API key leaked into {name}: {text!r}"
    # The status names the *slot* the key came from, which is what an operator needs,
    # and nothing else about the value. The word "api_key" is allowed to appear only
    # as that label -- never as a field carrying a secret.
    assert router.status()["key_env"] == "explicit api_key"
    for text in _all_strings(router.status()):
        if "api_key" in text:
            # Either the bare label or a sentence that ends by naming the slot.
            assert text == "explicit api_key" or text.endswith("(explicit api_key)"), \
                f"unexpected api_key mention: {text!r}"


def test_the_router_id_records_the_model_and_the_sdk_version():
    router = jev.JevRouter(api_key=SENTINEL_KEY, model="jev-latest")
    version = jev._sdk_version()
    if version is None:
        assert router.router_id == "jev-latest+typesafe-sdk==unknown"
    else:
        assert router.router_id == f"jev-latest+typesafe-sdk=={version}"
    assert jev.JevRouter(api_key=SENTINEL_KEY, model="jev-1.13").router_id.startswith("jev-1.13+")


def test_the_privacy_note_distinguishes_on_off_and_disabled(monkeypatch):
    on = jev.JevRouter(api_key=SENTINEL_KEY).status()["privacy"]
    assert "sent to" in on and "api.typesafe.ai" in on
    # ... and it names the environment variable a reader can check.
    monkeypatch.setenv(jev.API_KEY_ENV, SENTINEL_KEY)
    from_env = jev.JevRouter.from_environment()
    assert jev.API_KEY_ENV in from_env.status()["privacy"]
    disabled = jev.JevRouter.disabled_router().status()["privacy"]
    assert "nothing leaves this machine" in disabled
    unavailable = jev.JevRouter(
        api_key=None, unavailable_reason="no key"
    ).status()["privacy"]
    assert "nothing is sent" in unavailable


# --------------------------------------------------------------------------- #
# what the API carries
# --------------------------------------------------------------------------- #
def test_the_router_block_carries_the_six_named_fields():
    router, _ = make_router()
    verdict = router.submit("往前走", library_entries=LIBRARY, prompt_hash="x").resolve()
    block = verdict.router_block(deadline_ms=router.deadline_ms, model=router.model, sdk_version=router.sdk_version)
    for field in ("state", "router_id", "route", "probabilities", "latency_ms", "cache_hit"):
        assert field in block, field
    assert block["state"] == jev.STATE_OK
    assert block["route"] == jev.ROUTE_COMPOSE
    assert block["probabilities"][jev.ROUTE_COMPOSE] == pytest.approx(0.8)
    assert block["cache_hit"] is False
    assert block["deadline_ms"] == router.deadline_ms
    assert block["judgment"][jev.Q_TEMPO]["score"] == pytest.approx(1.0)


def test_a_degraded_router_block_reports_no_route_at_all():
    router = jev.JevRouter.disabled_router()
    verdict = router.submit("往前走", library_entries=LIBRARY).resolve()
    block = verdict.router_block(deadline_ms=router.deadline_ms, model=router.model, sdk_version=router.sdk_version)
    assert block["route"] is None
    assert block["probabilities"] == {}
    assert block["degraded"] is True


def test_health_status_flattens_the_last_verdict_onto_the_same_six_names():
    router, _ = make_router()
    before = router.status()
    assert before["route"] is None and before["probabilities"] == {} and before["cache_hit"] is None
    router.submit("往前走", library_entries=LIBRARY, prompt_hash="x").resolve()
    after = router.status()
    assert after["route"] == jev.ROUTE_COMPOSE
    assert after["latency_ms"] is not None
    assert after["cache_hit"] is False
    assert after["calls"] == 1


# --------------------------------------------------------------------------- #
# the router still runs without the optional extra
# --------------------------------------------------------------------------- #
def test_building_a_client_without_the_sdk_is_named_unavailable_rather_than_raising(monkeypatch):
    monkeypatch.setitem(sys.modules, "typesafe_sdk", None)
    error = jev._classify(RuntimeError("typesafe-sdk is not installed: ImportError"))
    assert error[0] == jev.STATE_UNAVAILABLE
