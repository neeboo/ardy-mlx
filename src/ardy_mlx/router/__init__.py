"""Semantic routing for the viewer: **code owns the flow, Jev supplies judgments**.

This package is deliberately separate from :mod:`ardy_mlx.sampling`. Sampling decides
*how* a clip is drawn from a vector; this decides *what the user asked for* and which
of a small, closed set of behaviours that maps to. Keeping them apart is what lets the
router be tested with no model, no checkpoint and no network, and lets the sampler be
tested with no router at all.

The router is always *advisory*. Every degraded path -- no key, no SDK, a timeout, a
malformed answer -- is named, reported in the payload, and falls back to the local
vector path with the fallback stated on screen rather than silently substituted.
"""

from __future__ import annotations

from .jev import (
    ACTION_COMPOSE,
    ACTION_GENERATE,
    ACTION_HOLD,
    ACTION_LIBRARY,
    ACTION_VECTOR,
    DEFAULT_DEADLINE_MS,
    ROUTES,
    ROUTE_COMPOSE,
    ROUTE_GENERATE,
    ROUTE_LIBRARY,
    ROUTE_NON_MOTION,
    ROUTE_VECTOR,
    STATE_BAD_RESPONSE,
    STATE_CACHED,
    STATE_DISABLED,
    STATE_ERROR,
    STATE_NO_UTTERANCE,
    STATE_OK,
    STATE_TIMEOUT,
    STATE_UNAVAILABLE,
    THRESHOLDS,
    Decision,
    JevRouter,
    JudgmentHandle,
    Thresholds,
    Verdict,
    build_questions,
    build_state,
    cache_key,
    decide,
    intensity_to_amplitude,
    known_intents_revision,
    normalize_judgment,
    resolve_api_key,
    tempo_to_speed,
)

__all__ = [
    "ACTION_COMPOSE",
    "ACTION_GENERATE",
    "ACTION_HOLD",
    "ACTION_LIBRARY",
    "ACTION_VECTOR",
    "DEFAULT_DEADLINE_MS",
    "ROUTE_COMPOSE",
    "ROUTE_GENERATE",
    "ROUTE_LIBRARY",
    "ROUTE_NON_MOTION",
    "ROUTE_VECTOR",
    "ROUTES",
    "STATE_BAD_RESPONSE",
    "STATE_CACHED",
    "STATE_DISABLED",
    "STATE_ERROR",
    "STATE_NO_UTTERANCE",
    "STATE_OK",
    "STATE_TIMEOUT",
    "STATE_UNAVAILABLE",
    "THRESHOLDS",
    "Decision",
    "JevRouter",
    "JudgmentHandle",
    "Thresholds",
    "Verdict",
    "build_questions",
    "build_state",
    "cache_key",
    "decide",
    "intensity_to_amplitude",
    "known_intents_revision",
    "normalize_judgment",
    "resolve_api_key",
    "tempo_to_speed",
]
