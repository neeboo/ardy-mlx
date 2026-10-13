"""The server's own stream counters must grow for as long as plans are asked for.

This is the server half of the round's symptom. The reader watched
``/api/health``'s ``stream.segments_generated`` / ``stream.frames_generated`` **freeze**
(10 segments / 800 frames, byte-identical reads four seconds apart) while the process
stayed alive and kept answering ``/api/health`` -- and the page stopped sending
``POST /api/plan``. So the two halves of that failure are measured separately and in
the form each one can be measured in:

* **this file** -- over real HTTP, against the real handler, the counters are strictly
  monotone across a sequence of plans. If the server ever stopped counting, the growth
  the reader was watching would be a lie, and this would catch it;
* ``tests/viewer_js_guard.test.mjs`` -- the *client* keeps asking: a >= 20-segment soak
  driven by the shipped pure functions against a stub ``/api/plan``, with negative
  controls that switch prefetching off, restore the old ``intent: null`` answer, wedge
  the in-flight flag, and break the deadline.

The checkpoint is not loaded: ``FakeRunner`` stands in, so the payload is real HTTP and
real encoding but no GPU work happens (see ``tests/conftest.py``). What that costs is
continuation: the fake runner has no continuation path, so every plan is honestly a
fresh clip (``continuation.applied`` is ``False`` throughout). The counters under test
are the counters the reader watched, and they do not depend on it.
"""

from __future__ import annotations

import json

import pytest

import test_viewer_http as tvh
from test_viewer_http import PROMPT_HASH, jrequest

# `live_server` is defined in `test_viewer_http`; fixtures are module-scoped by
# default, so publishing it here is what lets this file use the identical server setup
# (real handler on an ephemeral loopback port, `FakeRunner`, temp viewer directory)
# rather than a second, subtly different one.
live_server = tvh.live_server


@pytest.mark.parametrize("plans", [1, 20])
def test_stream_counters_are_monotone_across_a_plan_sequence(live_server, plans, capsys):
    """``segments_generated`` and ``frames_generated`` never stand still or go back."""
    base, _app = live_server
    seen = []
    for attempt in range(1, plans + 1):
        status, body, _ = jrequest(f"{base}/api/plan", "POST", {
            "prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random",
        })
        assert status == 200, body
        _s, health, _ = jrequest(f"{base}/api/health")
        stream = health["stream"]
        seen.append((attempt, stream["segments_generated"], stream["frames_generated"]))

    # One plan, one segment: the counter is the number of plans that produced a clip,
    # not a number that happens to be non-zero.
    assert [row[1] for row in seen] == list(range(1, plans + 1)), seen
    assert [row[2] for row in seen] == [40 * i for i in range(1, plans + 1)], seen
    # Strictly increasing on both axes, which is exactly what "it froze" falsifies.
    for previous, current in zip(seen, seen[1:]):
        assert current[0] == previous[0] + 1
        assert current[1] > previous[1]
        assert current[2] > previous[2]

    with capsys.disabled():
        print("\n  plan | segments_generated | frames_generated")
        for attempt, segments, frames in seen:
            print("  %4d | %18d | %16d" % (attempt, segments, frames))
        print("  final stream block: %s"
              % json.dumps({"segments_generated": seen[-1][1],
                            "frames_generated": seen[-1][2]}, sort_keys=True))


def test_stream_counters_do_not_move_without_a_plan(live_server):
    """The other half: with no request, nothing grows. A frozen counter is only a bug
    when the client is still asking -- which is why the reading must be paired with
    the page's own snapshot (`client_log.snapshot.segments`) rather than read alone."""
    base, _app = live_server
    jrequest(f"{base}/api/plan", "POST",
             {"prompt": PROMPT_HASH, "seed": 0, "duration": 2, "noise": "random"})
    _s, first, _ = jrequest(f"{base}/api/health")
    _s, second, _ = jrequest(f"{base}/api/health")
    assert first["stream"]["segments_generated"] == 1
    assert second["stream"]["segments_generated"] == 1, "an idle server must not invent segments"
    assert first["stream"]["frames_generated"] == second["stream"]["frames_generated"] == 40
