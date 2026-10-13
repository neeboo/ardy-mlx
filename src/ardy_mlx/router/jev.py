"""Jev (TypeSafe System One) as a *judge* for the viewer.

    A user says one sentence. Code decides what to do; Jev only supplies judgments.

Why the judgment is not the decision
------------------------------------
``route`` is a :class:`Choice` over four behaviours and the mapping from those four
names to what the viewer actually does lives in :func:`decide` -- a pure function of
plain data, in this file, next to the thresholds it reads. Nothing about the mapping
is inferred from model text, and no route name reaches the engine untranslated. That
is the whole design: the model answers a closed question, and *this* module is the
only place that turns an answer into an action.

The four questions that matter most, and why they are asked separately:

* ``is_motion`` -- "should a body move at all". A system that moves when it was not
  asked is the worst outcome available here, so this is its own Noul with its own
  floor, checked before the route is even consulted.
* ``route`` -- *which* of the four behaviours. One Choice, one answer, four options,
  and ``non_motion`` is one of them rather than an absence: a question with no "none
  of these" exit forces a wrong answer out of an off-topic sentence.
* ``known`` -- whether the sentence names something this engine can already do. It
  does not select a behaviour; it decides whether the bubble admits the engine is
  improvising.
* ``interrupt`` -- whether a new action replaces the one in flight. Independent of
  *what* the action is.

``library_pick`` is asked **speculatively**, in the same request, for the same reason
the TypeSafe docs give: all questions see the same state and run in parallel, so a
question whose answer is only consumed on one branch costs tokens but no wall clock.
Without it ``route == "library"`` would name a behaviour ("load the library entry")
with no way to say *which* entry.

What leaves the device
----------------------
Everything in ``state`` -- the user's sentence, the current action, the known intents
and the library texts -- is sent to TypeSafe's hosted API when the router is enabled.
That is documented in ``docs/VIEWER.md``; ``--no-router`` keeps the whole viewer local.
The API key is read from the server's environment only. It is never put in a payload,
never logged, and never reaches ``viewer/``.

Async, and honest about it
--------------------------
:meth:`JevRouter.submit` starts the call in its own thread and returns immediately, so
the sentence is being judged while the local encoder is still working. The request
collects the answer with :meth:`JudgmentHandle.resolve`, which waits only for the
remaining deadline. If the deadline passes, the verdict is ``router_timeout``, the
plan proceeds down the plain vector path, and the page says so: a degraded answer is
named, never passed off as a judgment. A call that finishes *after* its deadline still
lands in the cache, so the next identical sentence is answered from memory.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# names and defaults
# --------------------------------------------------------------------------- #

#: The environment variable the router reads. The SDK reads the same one.
API_KEY_ENV = "TYPESAFE_API_KEY"
#: Accepted fallback, mapped onto the same slot. Only used when the one above is unset.
API_KEY_FALLBACK_ENV = "JEV_API_KEY"
MODEL_ENV = "TYPESAFE_DEFAULT_MODEL"
BASE_URL_ENV = "TYPESAFE_BASE_URL"

DEFAULT_MODEL = "jev-latest"
#: A generous default: the first call of a process pays Jev's cold start (measured
#: 1712 ms on this machine), every warm call is far below it. The wait is bounded and
#: the caller is told which side of the deadline it landed on.
DEFAULT_DEADLINE_MS = 2500
#: Judgments are cached; this bounds the cache. One entry per distinct sentence.
DEFAULT_CACHE_SIZE = 256

# -- routes: the closed set a Choice may return, plus the degraded pseudo-route ----
ROUTE_LIBRARY = "library"
ROUTE_COMPOSE = "compose"
ROUTE_GENERATE = "generate"
ROUTE_NON_MOTION = "non_motion"
ROUTES: Tuple[str, ...] = (ROUTE_LIBRARY, ROUTE_COMPOSE, ROUTE_GENERATE, ROUTE_NON_MOTION)
#: Not a Choice option. The route name used when **no judgment was made**, so a
#: degraded payload can never be mistaken for one of the four readings above.
ROUTE_VECTOR = "vector"

# -- actions: what the viewer does about a route -------------------------------
ACTION_LIBRARY = "load_library"
ACTION_COMPOSE = "compose_knobs"
ACTION_GENERATE = "generate"
ACTION_HOLD = "hold"
ACTION_VECTOR = "vector_fallback"

#: route -> action. Total over :data:`ROUTES`, asserted by the tests.
ROUTE_ACTIONS: Dict[str, str] = {
    ROUTE_LIBRARY: ACTION_LIBRARY,
    ROUTE_COMPOSE: ACTION_COMPOSE,
    ROUTE_GENERATE: ACTION_GENERATE,
    ROUTE_NON_MOTION: ACTION_HOLD,
}

# -- router states. Every one of these is *named* in the payload. ---------------
STATE_OK = "ok"
STATE_CACHED = "cached"
STATE_TIMEOUT = "router_timeout"
STATE_UNAVAILABLE = "router_unavailable"
STATE_ERROR = "router_error"
STATE_BAD_RESPONSE = "router_bad_response"
STATE_DISABLED = "router_disabled"
STATE_NO_UTTERANCE = "router_no_utterance"
#: ``ok``/``cached`` are the only states in which a judgment was actually made.
JUDGED_STATES: Tuple[str, ...] = (STATE_OK, STATE_CACHED)
DEGRADED_STATES: Tuple[str, ...] = (
    STATE_TIMEOUT,
    STATE_UNAVAILABLE,
    STATE_ERROR,
    STATE_BAD_RESPONSE,
    STATE_DISABLED,
    STATE_NO_UTTERANCE,
)

# -- question names: also the answer keys --------------------------------------
Q_IS_MOTION = "is_motion"
Q_KNOWN = "known"
Q_ROUTE = "route"
Q_TEMPO = "tempo"
Q_INTENSITY = "intensity"
Q_EMOTION = "emotion"
Q_INTERRUPT = "interrupt"
Q_LIBRARY = "library_pick"

#: The label a ``library_pick`` answer carries when no library entry is close enough.
LIBRARY_NONE = "none"

# -- question definitions ------------------------------------------------------
#: Score levels are the model's rubric. They are descriptive on their own (the docs
#: are explicit that bare numbers give it nothing to match against), and they are
#: ordered from least to most, which is what makes a fractional score interpretable.
TEMPO_LEVELS: Tuple[str, ...] = ("slow and relaxed", "normal pace", "hurried")
INTENSITY_LEVELS: Tuple[str, ...] = ("subtle", "moderate", "vigorous")
#: Choice options for emotion. The user's own words, kept verbatim: the sentence the
#: model is judging is usually Chinese, and translating the *labels* while leaving the
#: sentence alone would ask it to match across languages for no reason.
EMOTIONS: Tuple[str, ...] = ("平静", "高兴", "烦躁", "低落")

#: How far ``tempo`` moves the engine's own pace, per level away from ``normal``.
TEMPO_SPEED_SPAN = 0.35
#: ``intensity`` maps linearly onto this amplitude band.
INTENSITY_AMPLITUDE_RANGE: Tuple[float, float] = (0.6, 1.4)

#: emotion -> (expression, gaze). **Placeholders**: the rig has no face and no eyes,
#: so these are carried in the payload for the next layer and marked as unwired.
EMOTION_PLACEHOLDERS: Dict[str, Tuple[str, str]] = {
    "平静": ("neutral", "forward"),
    "高兴": ("smile", "forward"),
    "烦躁": ("frown", "averted"),
    "低落": ("downcast", "down"),
}
DEFAULT_EMOTION = "平静"


@dataclass(frozen=True)
class Thresholds:
    """Every number that turns a probability into a branch, in one readable place.

    The docs' guidance is that a confidence threshold is not one number: it depends on
    what getting the answer wrong costs. These are the three different costs in this
    system, and they are deliberately not equal.
    """

    #: Below this the route is a reading rather than a decision: the conservative
    #: branch runs, the parameters are clamped to the gentlest setting, and the bubble
    #: says "I am not sure, using the nearest one".
    route_confidence: float = 0.5
    #: Below this the probability mass is split between the top routes. Same branch.
    route_top_probability: float = 0.5
    #: Below this probability the sentence is not a request for motion, and **no new
    #: action is started**. This is the one gate that can refuse to act, because
    #: moving unasked is worse than not moving when asked.
    is_motion: float = 0.5
    #: Below this the request is unfamiliar. It does not change the action -- it makes
    #: the bubble say the engine is improvising.
    known: float = 0.5


THRESHOLDS = Thresholds()


# --------------------------------------------------------------------------- #
# the state and the questions
# --------------------------------------------------------------------------- #
def build_state(
    utterance: Optional[str],
    *,
    current_action: Optional[str] = None,
    known_intents: Sequence[str] = (),
    library_entries: Sequence[Mapping[str, Any]] = (),
) -> dict:
    """The one state every question is asked about, with **named fields**.

    ``library_entries`` is always present, as ``[]`` when the library is empty. An
    omitted field would read as "no library information"; an empty list says "the
    library is empty", which is a different fact and the one that is true.
    """
    return {
        "utterance": utterance if utterance is not None else "",
        "context": {
            "current_action": current_action,
            "known_intents": list(known_intents),
            "library_entries": [dict(entry) for entry in library_entries],
        },
    }


def _library_labels(library_entries: Sequence[Mapping[str, Any]]) -> "OrderedDict[str, dict]":
    """Stable ``L1..Ln`` labels -> the entry behind each one.

    The labels are what the model chooses between, so they have to be short and
    countable; the *descriptions* carry the meaning. A prompt hash as a label would be
    opaque to the model and is kept inside the description instead.
    """
    labels: "OrderedDict[str, dict]" = OrderedDict()
    for index, entry in enumerate(library_entries, start=1):
        labels[f"L{index}"] = dict(entry)
    return labels


def library_choices(library_entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """The ``library_pick`` criteria: one described option per entry, plus an exit."""
    criteria: Dict[str, Any] = {}
    for label, entry in _library_labels(library_entries).items():
        criteria[label] = {
            "text": entry.get("text"),
            "prompt_hash": entry.get("prompt_hash"),
        }
    criteria[LIBRARY_NONE] = (
        "None of the library entries above is close enough to the request; it needs "
        "composing from existing motions or generating."
    )
    return criteria


def build_questions(
    library_entries: Sequence[Mapping[str, Any]] = (),
    *,
    sdk: Any = None,
) -> "OrderedDict[str, Any]":
    """The question set: seven questions about the sentence, one about the library.

    Every question asks **one** thing, and every closed question keeps a way out:
    ``route`` has ``non_motion``, ``library_pick`` has ``none``. Each question is
    individually decidable -- nothing here depends on another answer, which is what
    lets all of them travel in one request and run in parallel.

    ``sdk`` is the imported ``typesafe_sdk`` module. It is a parameter so this
    function can be exercised (and its shape asserted) with a stand-in, and so the
    question set is defined even on a host where the SDK is an uninstalled extra.
    """
    if sdk is None:
        import typesafe_sdk as sdk  # noqa: PLC0415 - optional extra, imported on demand

    questions: "OrderedDict[str, Any]" = OrderedDict()
    questions[Q_IS_MOTION] = sdk.Noul(
        # The wording matters, and the first version of it was wrong: it asked whether
        # the speaker was *requesting* motion, and this viewer's whole prompt
        # vocabulary is third-person *descriptions* ("A person walks forward."). Every
        # cached library entry was therefore read as a non-request and held -- found by
        # driving the running server, not by a test. `is_motion` has to mean "does this
        # name a physical action", and the false case has to be about the speaker's own
        # state rather than about grammar.
        instructions=(
            "`utterance` asks for a physical action, or **names** one that a body could "
            "perform. A description of an action counts: `A person walks forward.` is a "
            "motion. A remark about how the speaker feels, a greeting, an opinion or "
            "small talk is not."
        ),
        criteria={
            "true": "The sentence asks for, or names, a physical action a body could perform.",
            "false": "The sentence is about how someone feels, or is a greeting, an opinion or small talk.",
        },
    )
    questions[Q_KNOWN] = sdk.Noul(
        instructions=(
            "This system can act on `utterance` if it names one "
            "of `context.known_intents`, or if it can be produced by composing the "
            "engine's existing motions (walking forward, turning, speeding up, slowing "
            "down, waving, stopping). A sentence about an ability the system does not have "
            "is not known."
        ),
        criteria={
            "true": "The system can act on this sentence.",
            "false": "It needs an ability the system does not have.",
        },
    )
    questions[Q_ROUTE] = sdk.Choice(
        instructions=(
            "Choose how this system should carry out `utterance` -- whether it is phrased "
            "as a request or as a description of an action. "
            "`context.library_entries` are motions that already exist as ready vectors: "
            "choose `library` when one of them is the motion meant. Choose "
            "`compose` when the sentence is about how an existing motion is performed -- "
            "its speed, its direction, whether it stops -- or is a combination of "
            "existing motions. Choose `generate` when it is a physical action "
            "the library does not contain and that cannot be composed from existing "
            "motions. Choose `non_motion` when the sentence is not about motion at all."
        ),
        criteria={
            ROUTE_LIBRARY: "One of `context.library_entries` is the motion meant.",
            ROUTE_COMPOSE: (
                "How an existing motion is performed (pace, direction, stopping), or a "
                "combination of the engine's existing motions."
            ),
            ROUTE_GENERATE: (
                "A physical action the library does not contain and that cannot be "
                "composed from existing motions."
            ),
            ROUTE_NON_MOTION: "The sentence is not about motion. Keep doing whatever is happening.",
        },
    )
    questions[Q_TEMPO] = sdk.Score(instructions="How fast should the action in `utterance` be performed?", criteria=list(TEMPO_LEVELS))
    questions[Q_INTENSITY] = sdk.Score(
        instructions="How large and forceful should the action in `utterance` be?", criteria=list(INTENSITY_LEVELS)
    )
    questions[Q_EMOTION] = sdk.Choice(
        instructions="What is the speaker's mood in `utterance`?",
        criteria={emotion: None for emotion in EMOTIONS},
    )
    questions[Q_INTERRUPT] = sdk.Noul(
        instructions=(
            "Should the request in `utterance` interrupt and replace `context.current_action` "
            "straight away, rather than wait for the current action to finish?"
        ),
        criteria={
            "true": "Start this now; replace what is happening.",
            "false": "It can wait for the current action to finish.",
        },
    )
    questions[Q_LIBRARY] = sdk.Choice(
        instructions=(
            "If `utterance` means one of the motions in `context.library_entries`, which "
            "one? Answer `none` when no entry is close enough. This question is a lookup "
            "only; it does not decide how the request is carried out."
        ),
        criteria=library_choices(library_entries),
    )
    return questions


# --------------------------------------------------------------------------- #
# reading an answer
# --------------------------------------------------------------------------- #
class BadJudgment(ValueError):
    """The response arrived but could not be read as the questions that were asked."""


def _field(answer: Any, name: str) -> Any:
    if isinstance(answer, Mapping):
        if name not in answer:
            raise BadJudgment(f"answer has no {name!r} field")
        return answer[name]
    if not hasattr(answer, name):
        raise BadJudgment(f"answer has no {name!r} field")
    return getattr(answer, name)


def normalize_answer(answer: Any) -> dict:
    """One SDK answer object (or an equivalent mapping) as plain, JSON-safe data.

    Duck-typed on the ``type`` tag rather than on the SDK's classes, so a fake can
    stand in for the SDK in a test -- and so a future SDK that adds an answer kind
    fails loudly here instead of being mistaken for one of these three.
    """
    kind = answer.get("type") if isinstance(answer, Mapping) else getattr(answer, "type", None)
    if kind == "noul":
        return {"type": "noul", "noul": float(_field(answer, "noul"))}
    if kind == "choice":
        probabilities = _field(answer, "probabilities") or {}
        return {
            "type": "choice",
            "choice": str(_field(answer, "choice")),
            "confidence": float(_field(answer, "confidence")),
            "probabilities": {str(key): float(value) for key, value in probabilities.items()},
        }
    if kind == "score":
        # The SDK keys `probabilities`/`legend` by *integer* level; JSON keys are
        # strings. Normalise to strings here so the payload and the tests agree.
        probabilities = _field(answer, "probabilities") or {}
        legend = _field(answer, "legend") or {}
        return {
            "type": "score",
            "score": float(_field(answer, "score")),
            "confidence": float(_field(answer, "confidence")),
            "probabilities": {str(int(key)): float(value) for key, value in probabilities.items()},
            "legend": {str(int(key)): value for key, value in legend.items()},
        }
    raise BadJudgment(f"unrecognized answer type {kind!r}")


def normalize_judgment(answers: Mapping[str, Any], *, model: Optional[str] = None, latency_ms: Optional[float] = None) -> dict:
    """Every answer of one response, keyed by question name, plus how it was made.

    The required questions are required: a response missing ``route`` is not a
    partial judgment, it is an unusable one, and it raises here rather than being
    defaulted into an action.
    """
    normalized: Dict[str, Any] = {}
    for name, answer in answers.items():
        normalized[name] = normalize_answer(answer)
    missing = [name for name in (Q_IS_MOTION, Q_KNOWN, Q_ROUTE) if name not in normalized]
    if missing:
        raise BadJudgment(f"the response is missing required answer(s): {', '.join(missing)}")
    if normalized[Q_ROUTE].get("choice") not in ROUTES:
        raise BadJudgment(f"route answer {normalized[Q_ROUTE].get('choice')!r} is not one of {ROUTES}")
    judgment: Dict[str, Any] = dict(normalized)
    judgment["model"] = model
    judgment["model_latency_ms"] = latency_ms
    return judgment


# --------------------------------------------------------------------------- #
# pure mappings: probability -> parameter
# --------------------------------------------------------------------------- #
def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def tempo_to_speed(score: Optional[float], *, conservative: bool = False) -> float:
    """A tempo Score (0..2 over :data:`TEMPO_LEVELS`) as a multiplier on the engine's pace.

    Level 1 ("normal pace") is 1.0, so an unreadable or missing score is neutral.
    ``conservative`` clamps the result to at most 1.0: when the reading is a guess, the
    guess is allowed to be slower but never faster.
    """
    if score is None:
        return 1.0
    level = _clamp(float(score), 0.0, len(TEMPO_LEVELS) - 1)
    speed = 1.0 + (level - 1.0) * TEMPO_SPEED_SPAN
    if conservative:
        speed = min(speed, 1.0)
    return round(speed, 4)


def intensity_to_amplitude(score: Optional[float], *, conservative: bool = False) -> float:
    """An intensity Score as a multiplier on the engine's range. Neutral at level 1."""
    if score is None:
        return 1.0
    level = _clamp(float(score), 0.0, len(INTENSITY_LEVELS) - 1)
    low, high = INTENSITY_AMPLITUDE_RANGE
    amplitude = low + (level / (len(INTENSITY_LEVELS) - 1)) * (high - low)
    if conservative:
        amplitude = min(amplitude, 1.0)
    return round(amplitude, 4)


def _level_name(levels: Sequence[str], score: Optional[float]) -> Optional[str]:
    if score is None:
        return None
    return levels[int(round(_clamp(float(score), 0.0, len(levels) - 1)))]


# --------------------------------------------------------------------------- #
# the decision
# --------------------------------------------------------------------------- #
#: What the bubble says. One table, in the language the *speaker* reads: the rest of
#: the page is developer-facing English, but the bubble is product copy, and the
#: sentences it explains are usually Chinese. Changing this table changes the UI.
MESSAGES: Dict[str, str] = {
    ACTION_LIBRARY: "照库里那条来：{library_text}（Jev 读出这句最接近库里已有的一条）。",
    ACTION_COMPOSE: "用现有动作组合：速度 {speed:.2f}×、幅度 {amplitude:.2f}×（Jev 读出 {tempo} / {intensity}）。",
    ACTION_GENERATE: "交给本机 ARDY 生成（Jev 读出库里没有、也不宜从现有动作组合）。",
    "hold_non_motion": "这句不是动作要求（Jev 读 non_motion 的概率 {p:.2f}），保持当前动作。",
    "hold_not_motion": "这句更像在说话而不是要动（Jev 读 is_motion 的概率 {p:.2f}），保持当前动作。",
    "library_unresolved": "Jev 说走库，但没指出是哪一条，先用最接近的组合。",
    STATE_TIMEOUT: "这次没做语义解析（router_timeout：超过 {deadline} ms），先用纯向量路由继续。",
    STATE_UNAVAILABLE: "这次没做语义解析（router_unavailable：{detail}），先用纯向量路由继续。",
    STATE_ERROR: "这次没做语义解析（router_error：{detail}），先用纯向量路由继续。",
    STATE_BAD_RESPONSE: "这次没做语义解析（router_bad_response：{detail}），先用纯向量路由继续。",
    STATE_DISABLED: "语义路由已关闭（--no-router），只用纯向量路由。",
    STATE_NO_UTTERANCE: "这次没做语义解析（router_no_utterance：prompt 是哈希且库里没有对应文本）。",
}
#: Prefixed to any message produced by the conservative branch, so "I am not sure" is
#: never inferred from a probability the reader cannot see.
CONSERVATIVE_PREFIX = "我拿不准，先用最接近的："
#: Appended when ``known`` says the engine is improvising.
IMPROVISING_SUFFIX = "（Jev 说这类请求它没见过，先试一下。）"


@dataclass(frozen=True)
class Decision:
    """What the viewer does, why, and what to put in the bubble."""

    route: str
    action: str
    starts_new_action: bool
    conservative: bool
    source: str
    reason: str
    message: str
    params: dict = field(default_factory=dict)
    library_entry: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "route": self.route,
            "action": self.action,
            "starts_new_action": self.starts_new_action,
            "conservative": self.conservative,
            "source": self.source,
            "reason": self.reason,
            "message": self.message,
            "params": dict(self.params),
            "library_entry": dict(self.library_entry) if self.library_entry else None,
        }


def _hold(*, reason: str, message: str, params: Optional[dict] = None) -> Decision:
    return Decision(
        route=ROUTE_NON_MOTION,
        action=ACTION_HOLD,
        starts_new_action=False,
        conservative=False,
        source="jev",
        reason=reason,
        message=message,
        params=params or {},
    )


def decide(
    judgment: Optional[Mapping[str, Any]],
    *,
    state: str,
    deadline_ms: int = DEFAULT_DEADLINE_MS,
    library_entries: Sequence[Mapping[str, Any]] = (),
    thresholds: Thresholds = THRESHOLDS,
    detail: Optional[str] = None,
) -> Decision:
    """Turn a verdict's judgment into an action. **Pure** -- no clock, no IO, no SDK.

    The order of the checks is the safety property, and it is deliberate:

    1. a degraded state never claims a semantic parse (:data:`DEGRADED_STATES`);
    2. ``non_motion`` never starts a new action -- not at any confidence;
    3. an ``is_motion`` below its floor never starts a new action either, whatever the
       route said. Those two gates are the only places this function refuses to act,
       and both refuse in the direction that costs least;
    4. a low-confidence reading of a *motion* route is still carried out -- but on the
       gentlest parameters, flagged ``conservative``, and announced in the bubble.
    """
    # 1 -- degraded: say so, do not pretend.
    if state in DEGRADED_STATES:
        message = MESSAGES[state].format(deadline=deadline_ms, detail=detail or state)
        return Decision(
            route=ROUTE_VECTOR,
            action=ACTION_VECTOR,
            starts_new_action=True,
            conservative=False,
            source="vector",
            reason=f"state={state}" + (f" detail={detail}" if detail else ""),
            message=message,
            params=_neutral_params(),
        )
    if state not in JUDGED_STATES:
        raise ValueError(f"unknown router state {state!r}")

    if judgment is None:
        raise ValueError(f"state={state!r} is a judged state, so it must carry a judgment")
    route_answer = judgment[Q_ROUTE]
    route = str(route_answer["choice"])
    probabilities = dict(route_answer.get("probabilities") or {})
    confidence = float(route_answer.get("confidence") or 0.0)
    top_probability = max(probabilities.values()) if probabilities else confidence

    is_motion_answer = judgment.get(Q_IS_MOTION) or {}
    is_motion = is_motion_answer.get("noul")
    known_answer = judgment.get(Q_KNOWN) or {}
    known = known_answer.get("noul")

    tempo = (judgment.get(Q_TEMPO) or {}).get("score")
    intensity = (judgment.get(Q_INTENSITY) or {}).get("score")
    emotion_answer = judgment.get(Q_EMOTION) or {}
    emotion = emotion_answer.get("choice") or DEFAULT_EMOTION
    interrupt = (judgment.get(Q_INTERRUPT) or {}).get("noul")
    library_pick = (judgment.get(Q_LIBRARY) or {}).get("choice")

    conservative = (
        confidence < thresholds.route_confidence
        or top_probability < thresholds.route_top_probability
        or (known is not None and known < thresholds.known)
    )

    params = _params(
        tempo=tempo,
        intensity=intensity,
        emotion=emotion,
        interrupt=interrupt,
        conservative=conservative,
        judgment=judgment,
    )

    # 2 -- "not a motion request" is a refusal, not a route with a low score.
    if route == ROUTE_NON_MOTION:
        p = probabilities.get(ROUTE_NON_MOTION, top_probability)
        return _hold(
            reason=f"route={ROUTE_NON_MOTION} p={p:.3f}",
            message=MESSAGES["hold_non_motion"].format(p=p),
            params=params,
        )

    # 3 -- the sentence may not be a request for motion at all.
    if is_motion is not None and float(is_motion) < thresholds.is_motion:
        return _hold(
            reason=f"is_motion={float(is_motion):.3f} < {thresholds.is_motion}",
            message=MESSAGES["hold_not_motion"].format(p=float(is_motion)),
            params=params,
        )

    # 4 -- a motion route. Read the parameters, then act.
    library_entry: Optional[dict] = None
    reason = f"route={route} p={top_probability:.3f} conf={confidence:.3f}"
    if conservative:
        reason += " conservative"

    if route == ROUTE_LIBRARY:
        library_entry = _picked_entry(library_pick, library_entries)
        if library_entry is None:
            # The route said "one of the library entries" and the speculative lookup
            # did not name one. Composing is the nearest thing that can actually run,
            # and the bubble says the lookup came up empty rather than showing an
            # unrelated clip as if it were the one that was meant.
            route = ROUTE_COMPOSE
            conservative = True
            reason += f" library_pick={library_pick!r} unresolved"
            message = MESSAGES["library_unresolved"] + MESSAGES[ACTION_COMPOSE].format(
                speed=params["speed"],
                amplitude=params["amplitude"],
                tempo=params["tempo"]["level"],
                intensity=params["intensity"]["level"],
            )
        else:
            message = MESSAGES[ACTION_LIBRARY].format(library_text=library_entry.get("text"))
    elif route == ROUTE_COMPOSE:
        message = MESSAGES[ACTION_COMPOSE].format(
            speed=params["speed"],
            amplitude=params["amplitude"],
            tempo=params["tempo"]["level"],
            intensity=params["intensity"]["level"],
        )
    else:
        message = MESSAGES[ACTION_GENERATE]

    if conservative:
        message = CONSERVATIVE_PREFIX + message
    if known is not None and float(known) < thresholds.known:
        message += IMPROVISING_SUFFIX

    return Decision(
        route=route,
        action=ROUTE_ACTIONS[route],
        starts_new_action=True,
        conservative=conservative,
        source="jev",
        reason=reason,
        message=message,
        params=params,
        library_entry=library_entry,
    )


def _picked_entry(label: Optional[str], library_entries: Sequence[Mapping[str, Any]]) -> Optional[dict]:
    """Map a ``library_pick`` label back to the entry it stood for."""
    if not label or label == LIBRARY_NONE:
        return None
    labels = _library_labels(library_entries)
    entry = labels.get(str(label))
    if entry is None:
        # Tolerate a model that answered with the entry's own text or hash instead of
        # the label: the mapping is still exact, and refusing here would throw away a
        # usable answer.
        for candidate in labels.values():
            if label in (candidate.get("text"), candidate.get("prompt_hash")):
                return candidate
        return None
    return entry


def _neutral_params() -> dict:
    return {
        "speed": 1.0,
        "amplitude": 1.0,
        "tempo": {"score": None, "level": None, "probabilities": {}},
        "intensity": {"score": None, "level": None, "probabilities": {}},
        "emotion": DEFAULT_EMOTION,
        "expression": EMOTION_PLACEHOLDERS[DEFAULT_EMOTION][0],
        "gaze": EMOTION_PLACEHOLDERS[DEFAULT_EMOTION][1],
        "replaces_current": True,
        # The rig has no face and no eyes; these two fields are carried for the next
        # layer and are explicitly marked as not wired to anything.
        "expression_placeholder": True,
    }


def _params(
    *,
    tempo: Optional[float],
    intensity: Optional[float],
    emotion: str,
    interrupt: Optional[float],
    conservative: bool,
    judgment: Mapping[str, Any],
) -> dict:
    expression, gaze = EMOTION_PLACEHOLDERS.get(emotion, EMOTION_PLACEHOLDERS[DEFAULT_EMOTION])
    tempo_answer = judgment.get(Q_TEMPO) or {}
    intensity_answer = judgment.get(Q_INTENSITY) or {}
    return {
        "speed": tempo_to_speed(tempo, conservative=conservative),
        "amplitude": intensity_to_amplitude(intensity, conservative=conservative),
        "tempo": {
            "score": tempo,
            "level": _level_name(TEMPO_LEVELS, tempo),
            "probabilities": dict(tempo_answer.get("probabilities") or {}),
        },
        "intensity": {
            "score": intensity,
            "level": _level_name(INTENSITY_LEVELS, intensity),
            "probabilities": dict(intensity_answer.get("probabilities") or {}),
        },
        "emotion": emotion,
        "expression": expression,
        "gaze": gaze,
        "replaces_current": bool(interrupt is None or float(interrupt) >= 0.5),
        "expression_placeholder": True,
    }


# --------------------------------------------------------------------------- #
# the verdict and its cache key
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Verdict:
    """One judgment, or the named reason there is none."""

    state: str
    router_id: Optional[str] = None
    latency_ms: Optional[float] = None
    cache_hit: bool = False
    judgment: Optional[dict] = None
    detail: Optional[str] = None

    @property
    def degraded(self) -> bool:
        return self.state not in JUDGED_STATES

    @property
    def route(self) -> Optional[str]:
        if self.state not in JUDGED_STATES or not self.judgment:
            return None
        route_answer = self.judgment.get(Q_ROUTE) or {}
        return route_answer.get("choice")

    @property
    def probabilities(self) -> dict:
        if self.state not in JUDGED_STATES or not self.judgment:
            return {}
        return dict((self.judgment.get(Q_ROUTE) or {}).get("probabilities") or {})

    def router_block(self, *, deadline_ms: int, model: Optional[str] = None, sdk_version: Optional[str] = None) -> dict:
        """The ``router`` object the API carries. Exactly the six named fields, plus
        the ones a reader needs to tell a guess from a reading."""
        return {
            "state": self.state,
            "router_id": self.router_id,
            "route": self.route,
            "probabilities": self.probabilities,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "degraded": self.degraded,
            "detail": self.detail,
            "deadline_ms": deadline_ms,
            "model": model,
            "sdk_version": sdk_version,
            "judgment": self.judgment,
        }

    def as_dict(self) -> dict:
        return {
            "state": self.state,
            "router_id": self.router_id,
            "latency_ms": self.latency_ms,
            "cache_hit": self.cache_hit,
            "judgment": self.judgment,
            "detail": self.detail,
        }


def known_intents_revision(known_intents: Sequence[str], library_entries: Sequence[Mapping[str, Any]] = ()) -> str:
    """A short revision over everything the judgment was conditioned on besides the sentence.

    The cache key is ``(prompt_hash, router_id, known_intents_rev)``: a cached answer
    stops being valid the moment the library changes, because ``route`` and
    ``library_pick`` were both chosen against it. Hashing the intents and the entries
    together makes "the library changed" a cache miss by construction rather than by
    remembering to clear anything.
    """
    payload = {
        "known_intents": [str(item) for item in known_intents],
        "library_entries": [
            {"prompt_hash": entry.get("prompt_hash"), "text": entry.get("text")} for entry in library_entries
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def cache_key(prompt_hash: str, router_id: Optional[str], known_intents_rev: str) -> str:
    """The judgment cache key, as one string so it can be asserted on and logged."""
    return f"{prompt_hash}|{router_id or '-'}|{known_intents_rev}"


def resolve_api_key(environ: Optional[Mapping[str, str]] = None) -> Tuple[Optional[str], Optional[str]]:
    """``(key, which environment variable it came from)`` -- or ``(None, None)``.

    ``TYPESAFE_API_KEY`` wins. ``JEV_API_KEY`` is accepted as a fallback and *mapped
    onto the same slot* rather than used in a second code path, so there is exactly
    one place a key can enter this module. The value is never returned to a caller
    that writes it anywhere: :meth:`JevRouter.status` reports the variable's *name*.
    """
    env = os.environ if environ is None else environ
    for name in (API_KEY_ENV, API_KEY_FALLBACK_ENV):
        value = (env.get(name) or "").strip()
        if value:
            return value, name
    return None, None


# --------------------------------------------------------------------------- #
# the router
# --------------------------------------------------------------------------- #
class JudgmentHandle:
    """A Jev call running in its own thread, joined once with a deadline.

    A handle can also be *pre-resolved* -- for a disabled router, an unknown sentence,
    or a cache hit there is nothing to wait for -- so the caller has exactly one shape
    to read and no second code path for "the router did not run".
    """

    def __init__(
        self,
        key: str,
        *,
        thread: Optional[threading.Thread] = None,
        started_at: Optional[float] = None,
        deadline_s: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
        verdict: Optional[Verdict] = None,
        recorder: Optional[Callable[[Verdict], Verdict]] = None,
    ):
        self.key = key
        self.thread = thread
        self.started_at = clock() if started_at is None else started_at
        self.deadline_s = deadline_s
        self._clock = clock
        self._recorder = recorder
        #: The thread appends here. A one-element list append is atomic under the GIL
        #: and needs no lock; the handle is read only after a join or a liveness check.
        self._result: List[Verdict] = [] if verdict is None else [verdict]

    def store(self, verdict: Verdict) -> None:
        self._result.append(verdict)

    @property
    def result(self) -> Optional[Verdict]:
        return self._result[0] if self._result else None

    @property
    def pending(self) -> bool:
        return not self._result and self.thread is not None

    def elapsed_ms(self) -> float:
        return (self._clock() - self.started_at) * 1000.0

    def resolve(self) -> Verdict:
        """Wait for the remaining deadline, then answer -- with a judgment or a name."""
        if self.thread is None:
            assert self._result, "a handle with no thread must already hold a verdict"
            return self._result[0]
        remaining = max(0.0, self.deadline_s - (self._clock() - self.started_at))
        self.thread.join(remaining)
        if self.thread.is_alive():
            # The thread keeps running: its verdict will still be stored and cached.
            timed_out = Verdict(
                state=STATE_TIMEOUT,
                latency_ms=round(self.elapsed_ms(), 3),
                detail=f"no answer within {self.deadline_s * 1000.0:.0f} ms",
            )
            return self._recorder(timed_out) if self._recorder else timed_out
        stored = self.result
        if stored is None:  # pragma: no cover - a thread that ends without storing
            return Verdict(state=STATE_ERROR, latency_ms=round(self.elapsed_ms(), 3), detail="the judgment thread ended without an answer")
        return stored


class JevRouter:
    """Owns the SDK client, the deadline, the thread and the judgment cache.

    One instance per viewer process. The SDK's ``system_one`` is synchronous; running
    it in a thread is what keeps it off the request/MLX thread, and the cache is what
    keeps a repeated sentence from costing a second round trip (or a second charge).
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        key_source: Optional[str] = None,
        model: str = DEFAULT_MODEL,
        deadline_ms: int = DEFAULT_DEADLINE_MS,
        base_url: Optional[str] = None,
        cache_size: int = DEFAULT_CACHE_SIZE,
        client_factory: Optional[Callable[[str, str, int], Any]] = None,
        clock: Callable[[], float] = time.perf_counter,
        disabled: bool = False,
        disabled_reason: Optional[str] = None,
        unavailable_reason: Optional[str] = None,
    ):
        self.model = model
        self.deadline_ms = int(deadline_ms)
        self.base_url = base_url
        self.cache_size = int(cache_size)
        self._clock = clock
        self._client_factory = client_factory
        self._client: Any = None
        self._client_lock = threading.Lock()
        self._cache: "OrderedDict[str, Verdict]" = OrderedDict()
        self._cache_lock = threading.Lock()
        self._client_factory_calls = 0
        #: Named degradations, decided once at construction.
        self.disabled = bool(disabled)
        self.disabled_reason = disabled_reason
        self.unavailable_reason = unavailable_reason
        #: **Which slot the key came from** -- an environment variable's *name*, or
        #: ``"explicit api_key"`` when one was passed in. It is never the key itself.
        self.key_source = key_source or ("explicit api_key" if api_key else None)
        self._api_key = api_key
        self.sdk_version = _sdk_version()
        self.calls = 0
        self.cache_hits = 0
        self.timeouts = 0
        self.last_verdict: Optional[Verdict] = None
        self._stats_lock = threading.Lock()

        if not self.disabled and self._api_key is None and self.unavailable_reason is None:
            key, source = resolve_api_key()
            if key is None:
                self.unavailable_reason = f"neither {API_KEY_ENV} nor {API_KEY_FALLBACK_ENV} is set in this server's environment"
            else:
                self._api_key = key
                self.key_source = source

    # -- construction helpers -------------------------------------------
    @classmethod
    def disabled_router(cls, reason: str = "--no-router", **kwargs: Any) -> "JevRouter":
        """The router that exists so a disabled viewer can still *say* it is disabled."""
        return cls(disabled=True, disabled_reason=reason, **kwargs)

    @classmethod
    def from_environment(cls, *, deadline_ms: int = DEFAULT_DEADLINE_MS, model: Optional[str] = None, **kwargs: Any) -> "JevRouter":
        """The default construction: on when a key is present, named-off when not."""
        env_model = (os.environ.get(MODEL_ENV) or "").strip()
        return cls(
            model=model or env_model or DEFAULT_MODEL,
            deadline_ms=deadline_ms,
            base_url=(os.environ.get(BASE_URL_ENV) or "").strip() or None,
            **kwargs,
        )

    # -- identity --------------------------------------------------------
    @property
    def router_id(self) -> Optional[str]:
        """Model **and** SDK version. A judgment is only reusable against both."""
        if self.disabled or self.unavailable_reason:
            return None
        version = self.sdk_version or "unknown"
        return f"{self.model}+typesafe-sdk=={version}"

    @property
    def available(self) -> bool:
        return not self.disabled and self.unavailable_reason is None

    def status(self) -> dict:
        """What ``/api/health`` reports. Never contains the key, only its variable.

        The same six names a plan carries are at the top level -- ``state``,
        ``router_id``, ``route``, ``probabilities``, ``latency_ms``, ``cache_hit`` --
        so one reader can consume either payload. ``state`` here is the router's own
        condition (would it run?), not the last verdict's; the last verdict is under
        ``last``, and the four per-verdict fields are null until one exists.
        """
        with self._stats_lock:
            calls, hits, timeouts = self.calls, self.cache_hits, self.timeouts
        last_block = (
            self.last_verdict.router_block(deadline_ms=self.deadline_ms, model=self.model, sdk_version=self.sdk_version)
            if self.last_verdict
            else None
        )
        return {
            "enabled": not self.disabled,
            "available": self.available,
            "state": (
                STATE_DISABLED
                if self.disabled
                else (STATE_UNAVAILABLE if self.unavailable_reason else STATE_OK)
            ),
            "router_id": self.router_id,
            "route": last_block["route"] if last_block else None,
            "probabilities": last_block["probabilities"] if last_block else {},
            "latency_ms": last_block["latency_ms"] if last_block else None,
            "cache_hit": last_block["cache_hit"] if last_block else None,
            "model": self.model,
            "sdk_version": self.sdk_version,
            "deadline_ms": self.deadline_ms,
            "base_url": self.base_url,
            "key_env": self.key_source,
            "reason": self.disabled_reason or self.unavailable_reason,
            "calls": calls,
            "cache_hits": hits,
            "timeouts": timeouts,
            "cache_size": len(self._cache),
            "last": last_block,
            # Stated in the payload as well as the docs: the sentence leaves the
            # device when this is on, and does not when it is off.
            "privacy": (
                "disabled: nothing leaves this machine; the viewer is purely local"
                if self.disabled
                else (
                    "off: no API key in this server's environment, so nothing is sent"
                    if self.unavailable_reason
                    else (
                        f"on: `utterance` and `context` in every judgment are sent to the "
                        f"hosted TypeSafe API at {self.base_url or 'https://api.typesafe.ai'} "
                        f"({self.key_source})"
                    )
                )
            ),
        }

    # -- the client ------------------------------------------------------
    def _ensure_client(self) -> Any:
        if self._client is None:
            with self._client_lock:
                if self._client is None:
                    if self._client_factory is not None:
                        self._client_factory_calls += 1
                        self._client = self._client_factory(self._api_key, self.model, self.deadline_ms)
                    else:
                        self._client = _build_sdk_client(self._api_key, self.model, self.base_url, self.deadline_ms)
        return self._client

    # -- the cache -------------------------------------------------------
    def _cache_get(self, key: str) -> Optional[Verdict]:
        with self._cache_lock:
            verdict = self._cache.get(key)
            if verdict is not None:
                self._cache.move_to_end(key)
            return verdict

    def _cache_put(self, key: str, verdict: Verdict) -> None:
        with self._cache_lock:
            self._cache[key] = verdict
            self._cache.move_to_end(key)
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)

    def clear_cache(self) -> None:
        with self._cache_lock:
            self._cache.clear()

    # -- judging ---------------------------------------------------------
    def judge_sync(self, state: Mapping[str, Any], questions: Mapping[str, Any]) -> Verdict:
        """One call, in this thread. Used by the worker thread and by the tests."""
        started = self._clock()
        try:
            response = self._ensure_client().system_one(state=dict(state), questions=dict(questions))
        except Exception as exc:  # noqa: BLE001 - every failure below is named
            state_name, detail = _classify(exc)
            return Verdict(state=state_name, latency_ms=round((self._clock() - started) * 1000.0, 3), detail=detail)
        latency_ms = round((self._clock() - started) * 1000.0, 3)
        answers = getattr(response, "answers", None)
        if answers is None and isinstance(response, Mapping):
            answers = response.get("answers")
        model = getattr(response, "model", None) or self.model
        try:
            judgment = normalize_judgment(answers or {}, model=model, latency_ms=latency_ms)
        except BadJudgment as exc:
            return Verdict(state=STATE_BAD_RESPONSE, latency_ms=latency_ms, detail=str(exc))
        return Verdict(state=STATE_OK, router_id=self.router_id, latency_ms=latency_ms, judgment=judgment)

    def submit(
        self,
        utterance: Optional[str],
        *,
        current_action: Optional[str] = None,
        known_intents: Sequence[str] = (),
        library_entries: Sequence[Mapping[str, Any]] = (),
        prompt_hash: str = "",
        questions_factory: Callable[[Sequence[Mapping[str, Any]]], Mapping[str, Any]] = build_questions,
    ) -> JudgmentHandle:
        """Start the judgment and return immediately.

        A degraded router, or one whose sentence cannot be judged, returns a handle
        that is already resolved -- the caller's ``resolve()`` is uniform either way,
        so there is no second code path for "the router is off".
        """
        prepared = self._prepare(
            utterance,
            current_action=current_action,
            known_intents=known_intents,
            library_entries=library_entries,
            prompt_hash=prompt_hash,
        )
        if isinstance(prepared, Verdict):
            return JudgmentHandle("<degraded>", clock=self._clock, verdict=self._record(prepared), recorder=self._record)

        key, state, questions = prepared
        cached = self._cache_get(key)
        if cached is not None:
            with self._stats_lock:
                self.cache_hits += 1
            hit = Verdict(
                state=STATE_CACHED,
                router_id=cached.router_id,
                # This request spent no time at the model. The original call's own
                # cost stays visible in the judgment, where it is a fact about that
                # call rather than about this one.
                latency_ms=0.0,
                cache_hit=True,
                judgment=cached.judgment,
                detail=f"cached under {key}",
            )
            return JudgmentHandle(key, clock=self._clock, verdict=self._record(hit), recorder=self._record)

        handle = JudgmentHandle(
            key, started_at=self._clock(), deadline_s=self.deadline_ms / 1000.0, clock=self._clock, recorder=self._record
        )
        self._start_worker(handle, key, state, questions)
        return handle

    def _start_worker(self, handle: JudgmentHandle, key: str, state: Mapping[str, Any], questions: Mapping[str, Any]) -> None:
        def run() -> None:
            with self._stats_lock:
                self.calls += 1
            verdict = self.judge_sync(state, questions)
            verdict = self._record(verdict)
            if verdict.state in JUDGED_STATES:
                # Even a call that outlived its deadline is worth keeping: the next
                # identical sentence is answered from here instead of from the network.
                self._cache_put(key, verdict)
            handle.store(verdict)

        thread = threading.Thread(target=run, name="jev-judgment", daemon=True)
        handle.thread = thread
        thread.start()

    def _record(self, verdict: Verdict) -> Verdict:
        with self._stats_lock:
            if verdict.state == STATE_TIMEOUT:
                self.timeouts += 1
            self.last_verdict = verdict
        return verdict

    def _prepare(
        self,
        utterance: Optional[str],
        *,
        current_action: Optional[str],
        known_intents: Sequence[str],
        library_entries: Sequence[Mapping[str, Any]],
        prompt_hash: str,
    ) -> Any:
        """Everything decided before any work starts: the state, the questions, the key.

        Returns either a :class:`Verdict` (there is nothing to ask) or the tuple the
        worker needs.
        """
        if self.disabled:
            return Verdict(state=STATE_DISABLED, detail=self.disabled_reason)
        if self.unavailable_reason:
            return Verdict(state=STATE_UNAVAILABLE, detail=self.unavailable_reason)
        text = (utterance or "").strip()
        if not text:
            return Verdict(state=STATE_NO_UTTERANCE, detail="the request names no text to judge")
        state = build_state(
            text,
            current_action=current_action,
            known_intents=known_intents,
            library_entries=library_entries,
        )
        questions = build_questions(library_entries)
        rev = known_intents_revision(known_intents, library_entries)
        key = cache_key(prompt_hash, self.router_id, rev)
        return key, state, questions


def _build_sdk_client(api_key: Optional[str], model: str, base_url: Optional[str], deadline_ms: int) -> Any:
    """Construct the real SDK client, once, lazily.

    The SDK is an optional extra: a viewer with no router dependency installed still
    runs, and says ``router_unavailable`` instead of failing to import.
    """
    try:
        from typesafe_sdk import RetryPolicy, TypeSafeClient
    except Exception as exc:  # pragma: no cover - depends on the install
        raise RuntimeError(f"typesafe-sdk is not installed: {type(exc).__name__}: {exc}") from exc

    return TypeSafeClient(
        api_key=api_key,
        model=model,
        base_url=base_url,
        # The deadline is the contract with the page. A retry that outlives it would
        # turn a named timeout into a mystery stall, so the SDK's own retries are off
        # and the wait is bounded here instead.
        retry=RetryPolicy(max_retries=0),
        timeout=max(deadline_ms / 1000.0, 0.1),
    )


def _classify(exc: BaseException) -> Tuple[str, str]:
    """Name a failure the way the payload reports it. Never re-raises.

    Classified by the SDK's own exception *names* rather than by matching words in the
    message: a sentence that happens to contain "timeout" must not turn some other
    failure into ``router_timeout``.
    """
    name = type(exc).__name__
    detail = str(exc) or name
    if name in ("TypeSafeAPITimeoutError", "TimeoutError", "ReadTimeout", "ConnectTimeout"):
        return STATE_TIMEOUT, detail
    if name in ("TypeSafeAuthenticationError", "TypeSafePermissionDeniedError"):
        return STATE_UNAVAILABLE, detail
    if name in ("TypeSafeAPIConnectionError", "ConnectionError", "OSError"):
        return STATE_UNAVAILABLE, detail
    if name in ("RuntimeError", "ImportError", "ModuleNotFoundError"):
        # `_build_sdk_client` raises RuntimeError for a missing SDK, which is the one
        # dependency failure this module can produce on its own.
        return STATE_UNAVAILABLE, detail
    return STATE_ERROR, detail


def _sdk_version() -> Optional[str]:
    try:
        from typesafe_sdk import __version__

        return str(__version__)
    except Exception:  # noqa: BLE001 - the SDK is optional
        return None
