// tests/viewer_js_guard.test.mjs -- the viewer page's prompt guard, run under node.
//
// Why this file exists: the page has no browser test harness, and synthesising
// clicks/keys is not an option. So the *decision* whether to send a request lives in
// `viewer/app.js` inside a marker-delimited region of pure functions
// (`pure-prompt-guard`), and this test loads that region out of the **shipped file**
// and runs it in a bare `node:vm` context as data. Nothing is copied: if the region
// stops parsing, or loses its guard, or stops being pure, this test fails.
//
//   node tests/viewer_js_guard.test.mjs
//   node --test tests/viewer_js_guard.test.mjs
//   ./.venv/bin/python -m pytest -q      (via test_viewer_http.py, which shells out)
//
// The two rules under test:
//
//   * `resolveRequest` turns the controls into a prompt, or into `null`;
//   * `shouldPlan` is the single guard every trigger goes through, so a request is
//     never built from a prompt that did not resolve. Without it the body is
//     `{"seed":0,...}` -- no `prompt` key at all, because `JSON.stringify` drops
//     `undefined` -- which the server refuses as 400 `no_prompt_given`.

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import vm from 'node:vm';

const HERE = dirname(fileURLToPath(import.meta.url));
const APP_PATH = join(HERE, '..', 'viewer', 'app.js');
const SOURCE = readFileSync(APP_PATH, 'utf8');

const START = '// --- pure-prompt-guard: start';
const END = '// --- pure-prompt-guard: end';
const TIMELINE_START = '// --- pure-timeline: start';
const TIMELINE_END = '// --- pure-timeline: end';
const BUBBLE_START = '// --- pure-bubble: start';
const BUBBLE_END = '// --- pure-bubble: end';

/** A marker-delimited region of the shipped page, verbatim. */
function markedRegion(source, start, end) {
  const from = source.indexOf(start);
  const to = source.indexOf(end);
  assert.ok(from >= 0, `viewer/app.js must contain the "${start}" marker`);
  assert.ok(to > from, `viewer/app.js must contain "${end}" after the start marker`);
  return source.slice(from, to);
}

/** The marked prompt-guard region of the shipped page, verbatim. */
function pureRegion(source) {
  return markedRegion(source, START, END);
}

/** The marked bubble region of the shipped page, verbatim. */
function pureBubble(source) {
  return markedRegion(source, BUBBLE_START, BUBBLE_END);
}

/** Evaluate the region in a fresh context and hand back its functions. */
function loadGuard(context = vm.createContext({})) {
  return vm.runInContext(
    `${pureRegion(SOURCE)}\n({ resolveRequest, shouldPlan, planBody, referenceDecision, NO_PROMPT_HINT });`,
    context,
    { filename: 'viewer/app.js#pure-prompt-guard' },
  );
}

/** Evaluate the bubble region in a fresh context and hand back its functions. */
function loadBubble(context = vm.createContext({})) {
  return vm.runInContext(
    `${pureBubble(SOURCE)}\n({ bubblePending, bubbleFromPlan, bubbleMeta, clip, fmtMsShort, ROUTE_TAGS, BUBBLE_QUOTE_CHARS });`,
    context,
    { filename: 'viewer/app.js#pure-bubble' },
  );
}

/** The marked timeline region of the shipped page, verbatim. */
function pureTimeline(source) {
  return markedRegion(source, TIMELINE_START, TIMELINE_END);
}

/**
 * Load the playback model. `generateClip`, `planBody` and `clip` are carried in
 * because the timeline region names them (the body builder and the quote helper are
 * the shipped page's own, not copies).
 */
function loadTimeline(context = vm.createContext({})) {
  return vm.runInContext(
    `${pureRegion(SOURCE)}\n${pureBubble(SOURCE)}\n${pureTimeline(SOURCE)}\n`
    + `({ RUNAHEAD_FRAMES, PREFETCH_TIMEOUT_MS, PREFETCH_STATE, BUFFER_EMPTY_HINT, `
    + `STREAM_HOLD, timelineOf, segmentBase, `
    + `segmentAtFrame, advancePlayhead, bufferDepth, nextSegmentIntent, shouldPrefetch, `
    + `holdResolution, prefetchNote, `
    + `shouldFitCamera, playbackIntentNote, bufferNote, poseBounds, trailPoints, `
    + `trailUnderNaiveAppend, concatFloat32, concatUint8, `
    + `followFocus, followFocusInstant, focusTargetAt, fitFocusFromBounds, `
    + `fitDistanceFromBounds, fitPlacement, cameraPositionFor, projectPoint, vec3Of, `
    + `FOCUS_TAU_S, CAMERA_MAX_STEP_M, FOCUS_MIN_Y, CAMERA_FOV_DEG });`,
    context,
    { filename: 'viewer/app.js#pure-timeline' },
  );
}

const stream = loadTimeline();

/** The body of one top-level function, up to the next declaration. */
function functionBody(source, name) {
  const marker = `function ${name}(`;
  const start = source.indexOf(marker);
  if (start < 0) return '';
  const after = source.slice(start + marker.length);
  const stop = after.search(/\n(?:async )?function |\n\/\/ ---/);
  return stop < 0 ? after : after.slice(0, stop);
}

const guard = loadGuard();

/** Objects made inside `node:vm` have that realm's prototypes: normalize to compare. */
const plain = (value) => JSON.parse(JSON.stringify(value));

// The three control states, as plain data. `prompt_hash` is what /api/health reports
// and what the <option> value is set to.
const CACHED = { prompt_hash: 'a'.repeat(64), text: 'A person walks forward and waves.' };
const NOT_READY = { prompt_hash: '', text: '' };

test('empty text box + empty dropdown: nothing resolves, so nothing is planned', () => {
  const resolved = guard.resolveRequest({ typed: '', entry: null });
  assert.equal(resolved, null, 'an empty pair must resolve to null, never to a blank prompt');
  assert.equal(guard.shouldPlan(resolved), false);
  assert.equal(guard.shouldPlan(undefined), false);
  assert.equal(guard.shouldPlan(null), false);
});

test('a dropdown that is not ready yet is not a prompt either', () => {
  // The `<select>` is empty before /api/health fills it, and the page reads
  // `select.value`, which is "" there -- that is the state the reload race used to
  // build a request from.
  for (const entry of [null, undefined, NOT_READY, { text: 'no hash here' }]) {
    const resolved = guard.resolveRequest({ typed: '', entry });
    assert.equal(guard.shouldPlan(resolved), false, `entry ${JSON.stringify(entry)}`);
  }
});

test('whitespace is not a prompt: it falls through to the dropdown', () => {
  assert.equal(guard.resolveRequest({ typed: '   \n\t ', entry: null }), null);
  const resolved = guard.resolveRequest({ typed: '   \n\t ', entry: CACHED });
  assert.equal(resolved.source, 'cached', 'blank text must not shadow a real selection');
  assert.equal(resolved.prompt, CACHED.prompt_hash);
});

test('text in the box is used verbatim (trimmed), and wins over the selection', () => {
  const resolved = guard.resolveRequest({ typed: '  a person waves  ', entry: CACHED });
  assert.equal(resolved.source, 'text');
  assert.equal(resolved.prompt, 'a person waves');
  assert.equal(resolved.text, 'a person waves');
  assert.equal(guard.shouldPlan(resolved), true);
});

test('a selected cached embedding is planned by its prompt_hash', () => {
  const resolved = guard.resolveRequest({ typed: '', entry: CACHED });
  assert.equal(resolved.source, 'cached');
  assert.equal(resolved.prompt, CACHED.prompt_hash);
  assert.equal(guard.shouldPlan(resolved), true);
  assert.deepEqual(
    plain(guard.planBody(resolved, { seed: 3, duration: 4, noise: 'golden' })),
    { prompt: CACHED.prompt_hash, seed: 3, duration: 4, noise: 'golden' },
  );
});

test('negative control: bypassing the guard produces a body with no prompt key', () => {
  // This is the whole failure mode, mechanically. An unresolved prompt is not an
  // empty string once it has been through JSON.stringify -- the key is gone.
  const unresolved = { source: 'text', prompt: undefined };
  assert.equal(guard.shouldPlan(unresolved), false, 'the guard must refuse this');

  const body = guard.planBody(unresolved, { seed: 0, duration: 2, noise: 'golden' });
  const sent = JSON.parse(JSON.stringify(body));
  assert.deepEqual(Object.keys(sent).sort(), ['duration', 'noise', 'seed']);
  assert.equal(sent.prompt, undefined);
  assert.equal(JSON.stringify(body), '{"seed":0,"duration":2,"noise":"golden"}',
    'this exact body is what the server answers with 400 no_prompt_given');
  // ... and the guarded path cannot produce it:
  assert.equal(guard.shouldPlan(guard.resolveRequest({ typed: '', entry: null })), false);
});

test('the reference is only requested for a duration /api/health says has a golden run', () => {
  const runs = [
    { duration_s: 2.0, available: true, npz_path: 'artifacts/golden-run/out/x.npz' },
    { duration_s: 4.0, available: true, npz_path: 'artifacts/golden-run-4s/out/x.npz' },
  ];
  assert.deepEqual(
    plain(guard.referenceDecision({ goldenReferences: runs, duration: 2 })),
    { fetch: true, reason: null, message: null },
  );
  assert.deepEqual(
    plain(guard.referenceDecision({ goldenReferences: runs, duration: 4 })),
    { fetch: true, reason: null, message: null },
  );

  // 8 s is a perfectly good /api/plan duration and has no golden clip: the page was
  // told so by /api/health, so it must not find out again with a 404.
  const eight = guard.referenceDecision({ goldenReferences: runs, duration: 8 });
  assert.equal(eight.fetch, false);
  assert.equal(eight.reason, 'no_golden_reference_for_duration');
  assert.match(eight.message, /No golden clip exists for 8\.0 s/);
  assert.match(eight.message, /2\.0 s, 4\.0 s/, 'the message names the runs that do exist');
});

test('a configured golden run whose npz is missing is also known in advance', () => {
  const decision = guard.referenceDecision({
    goldenReferences: [{ duration_s: 2.0, available: false }],
    duration: 2,
  });
  assert.equal(decision.fetch, false);
  assert.equal(decision.reason, 'golden_reference_missing');
});

test('reference decisions with no runs, no duration, or malformed runs never fetch', () => {
  assert.equal(guard.referenceDecision({}).fetch, false);
  assert.equal(guard.referenceDecision({ goldenReferences: [], duration: 2 }).reason,
    'no_golden_reference_for_duration');
  assert.equal(guard.referenceDecision({ goldenReferences: [], duration: 2 }).message,
    'No golden clip exists for 2.0 s.');
  assert.equal(guard.referenceDecision({ goldenReferences: [], duration: NaN }).reason, 'no_duration');
  // `Number(null)` is 0: a null record must not be read as "the 0 s golden run".
  assert.equal(
    guard.referenceDecision({ goldenReferences: [null], duration: 0 }).fetch,
    false,
  );
});

test('the guard region is pure: it runs with no document, no fetch, no state', () => {
  // A guard that reaches for the DOM cannot be tested; this is what keeps the
  // previous test honest. Every global the page uses is a trap here.
  const trap = new Proxy({}, {
    get(_target, key) { throw new Error(`the pure region touched ${String(key)}`); },
  });
  const isolated = loadGuard(vm.createContext({
    document: trap, window: trap, fetch: trap, state: trap, els: trap,
    THREE: trap, performance: trap, requestAnimationFrame: trap,
  }));
  assert.equal(isolated.shouldPlan(isolated.resolveRequest({ typed: 'walk', entry: null })), true);
  assert.equal(isolated.shouldPlan(isolated.resolveRequest({ typed: '', entry: null })), false);
  assert.equal(typeof isolated.NO_PROMPT_HINT, 'string');
  assert.notEqual(isolated.NO_PROMPT_HINT.trim(), '');
});

// ---------------------------------------------------------------------------
// the bubble: what the viewer says it understood, and where that came from
// ---------------------------------------------------------------------------

const bubble = loadBubble();

/** A `/api/plan` payload's two routing blocks, as the server sends them. */
function planPayload({ route = 'compose', state = 'ok', degraded = false, probability = 0.83,
  startsNewAction = true, message = '用现有动作组合：速度 0.65×、幅度 1.00×。', source = 'jev',
  latency = 412.3, cacheHit = false } = {}) {
  return {
    kind: startsNewAction ? 'plan' : 'hold',
    router: {
      state,
      router_id: 'jev-latest+typesafe-sdk==0.7.4',
      route: degraded ? null : route,
      // The chosen route's own probability last, so it wins for the default fixture.
      probabilities: degraded ? {} : Object.assign({ compose: 0.1 }, { [route]: probability }),
      latency_ms: latency,
      cache_hit: cacheHit,
      degraded,
    },
    decision: {
      route: degraded ? 'vector' : route,
      action: startsNewAction ? 'compose_knobs' : 'hold',
      starts_new_action: startsNewAction,
      conservative: false,
      source,
      reason: 'route=compose p=0.830 conf=0.840',
      message,
      params: {},
    },
  };
}

test('the bubble appears before the request and promises only what will happen', () => {
  const resolved = guard.resolveRequest({ typed: '慢慢走过去', entry: null });
  const asking = bubble.bubblePending(resolved, 'ok');
  assert.equal(asking.state, 'thinking');
  assert.match(asking.text, /慢慢走过去/, 'the reader must see their own sentence taken');
  assert.match(asking.meta, /不挡生成/);

  // A viewer that is not asking anybody must not say it is. This is the same rule as
  // the degraded bubble below, one step earlier in the interaction.
  for (const off of ['router_disabled', 'router_unavailable']) {
    const quiet = bubble.bubblePending(resolved, off);
    assert.doesNotMatch(quiet.meta, /Jev/, `router state ${off} must not claim to ask Jev`);
    assert.match(quiet.meta, /本地向量/);
  }
});

test('a routed plan shows what it chose, why, and the source', () => {
  const view = bubble.bubbleFromPlan(planPayload());
  assert.equal(view.state, 'routed');
  assert.equal(view.text, '用现有动作组合：速度 0.65×、幅度 1.00×。', 'the server sent the sentence');
  assert.match(view.meta, /state=ok/);
  assert.match(view.meta, /route=compose p=0\.83/);
  assert.match(view.meta, /jev-latest\+typesafe-sdk==0\.7\.4/);
  assert.match(view.meta, /412 ms/);
  assert.match(view.meta, /source=jev/);
});

test('a non-motion hold is drawn as a hold, never as a new action', () => {
  const view = bubble.bubbleFromPlan(planPayload({
    route: 'non_motion', startsNewAction: false, probability: 0.9,
    message: '这句不是动作要求（Jev 读 non_motion 的概率 0.90），保持当前动作。',
  }));
  assert.equal(view.state, 'hold');
  assert.match(view.text, /保持当前动作/);
});

test('negative control: a degraded judgment is never drawn as a normal one', () => {
  // The failure this exists for: a timeout or a missing key quietly rendered with the
  // same look as a real reading, so a page that did no semantic parsing looks exactly
  // like one that did.
  const view = bubble.bubbleFromPlan(planPayload({
    state: 'router_timeout', degraded: true, source: 'vector', latency: 2500,
    message: '这次没做语义解析（router_timeout：超过 2500 ms），先用纯向量路由继续。',
  }));
  assert.equal(view.state, 'degraded', 'a degraded verdict must be visually distinct');
  assert.match(view.meta, /state=router_timeout/);
  assert.doesNotMatch(view.meta, /route=/, 'there is no route to report, so none may be reported');
  assert.match(view.meta, /source=vector/);
  assert.match(view.text, /没做语义解析/);
});

test('a cached judgment says it was cached rather than showing a bare latency', () => {
  const view = bubble.bubbleFromPlan(planPayload({ state: 'cached', cacheHit: true, latency: 0 }));
  assert.equal(view.state, 'routed');
  assert.match(view.meta, /cached \(0 ms\)/);
});

test('a conservative decision is labelled conservative in the meta line', () => {
  const payload = planPayload();
  payload.decision.conservative = true;
  const view = bubble.bubbleFromPlan(payload);
  assert.match(view.meta, /保守分支/);
});

test('a server that does not route gets a plain bubble, not an invented route', () => {
  const view = bubble.bubbleFromPlan({ kind: 'plan', frames: 40 });
  assert.equal(view.state, 'plain');
  assert.equal(view.meta, '');
  assert.doesNotMatch(view.text, /route/);
});

test('the bubble region is pure: it runs with no document, no fetch, no state', () => {
  const trap = new Proxy({}, {
    get(_target, key) { throw new Error(`the pure region touched ${String(key)}`); },
  });
  const isolated = loadBubble(vm.createContext({
    document: trap, window: trap, fetch: trap, state: trap, els: trap,
    THREE: trap, performance: trap, requestAnimationFrame: trap,
  }));
  assert.equal(isolated.bubbleFromPlan(planPayload()).state, 'routed');
  assert.equal(isolated.clip('x'.repeat(200)).length, isolated.BUBBLE_QUOTE_CHARS);
  assert.equal(isolated.fmtMsShort(2500), '2.50 s');
});

// ---------------------------------------------------------------------------
// wiring: the guard above is only worth anything if the page actually calls it
// ---------------------------------------------------------------------------

/** The properties of the shipped page that make the guard load-bearing. */
function wiring(source) {
  const reference = functionBody(source, 'loadReference');
  const generateCalls = (source.match(/generateClip\(/g) || []).length;
  const generateDecls = (source.match(/function generateClip\(/g) || []).length;
  return {
    // `/api/plan` is only reached through a checked `resolved`.
    guardInPlan: /if \(!shouldPlan\(resolved\)\)/.test(source),
    bodyFromPlanBody: /planBody\(\{ prompt: resolved\.prompt \}, \{ seed, duration, noise \}\)/
      .test(source),
    // ... and `generateClip` has exactly one caller left: the guarded entry point.
    singleGenerateCallSite: generateDecls === 1 && generateCalls === 2
      && source.includes('return generateClip(resolved, { reset: trigger === \'reload\' });'),
    // Every control that can plan goes through that entry point.
    allTriggersGuarded: [
      "planIfReady('generate')", "planIfReady('enter')", "planIfReady('go')",
      "planIfReady('reload')", "planIfReady('boot')",
    ].every((call) => source.includes(call)),
    // The reload path waits for /api/health, which is what fills the dropdown.
    reloadAwaitsHealth: /els\.reload\.addEventListener\('click', async \(\) => \{[\s\S]{0,600}?await loadHealth\(\)[\s\S]{0,300}?await planIfReady\('reload'\)/.test(source),
    // `/api/reference` is decided before it is requested, not after a 404.
    referenceGuardedFirst: reference.indexOf('referenceDecision(') >= 0
      && reference.indexOf('referenceDecision(') < reference.indexOf('/api/reference'),
    // The three controls that invalidate the loaded reference share one function.
    oneReferenceRefreshPath: (source.match(/refreshGoldenForControls\(\)/g) || []).length >= 4,
    // The bubble is drawn before the request leaves, so it cannot be a summary that
    // only exists once the round trip is over.
    bubbleBeforeRequest: (() => {
      const body = functionBody(source, 'planIfReady');
      const pending = body.indexOf('showBubble(bubblePending(');
      return pending >= 0 && pending < body.indexOf('generateClip(resolved');
    })(),
    // ... and redrawn from the *server's* verdict, not from the page's guess at one.
    bubbleFromVerdict: /showBubble\(bubbleFromPlan\(payload\)\)/.test(source),
    // A hold keeps the queue exactly as it is: nothing is appended and the branch
    // returns.
    holdInstallsNothing: (() => {
      const body = functionBody(source, 'pullSegment');
      const hold = body.indexOf("payload.kind === 'hold'");
      const install = body.indexOf('appendSegment(payload)');
      return hold >= 0 && install > hold && /return null;/.test(body.slice(hold, install));
    })(),
    // The one entry point reaches the queue by *appending*; it has no reset of its own.
    streamAppend: (() => {
      const body = functionBody(source, 'appendSegment');
      return body.includes('state.segments.push(segment)')
        && !body.includes('resetStream');
    })(),
  };
}

test('wiring: every trigger reaches the guard, and the body is built by planBody()', () => {
  const checks = wiring(SOURCE);
  for (const [name, ok] of Object.entries(checks)) {
    assert.ok(ok, `viewer/app.js wiring check failed: ${name}`);
  }
});

test('negative control: the wiring checks fail on a guard-free copy of the page', () => {
  // If someone bypasses the guard -- the exact regression this test exists for --
  // these checks must notice. A check that cannot fail is not a check.
  const bypassed = SOURCE
    .replaceAll('if (!shouldPlan(resolved)) {', 'if (false) {')
    .replace('planBody({ prompt: resolved.prompt }, { seed, duration, noise })',
      '{ prompt: resolved.prompt, seed, duration, noise }')
    .replace('await loadHealth().catch(() => {});', 'loadHealth().catch(() => {});')
    .replace("await planIfReady('reload');", 'generateClip();')
    // ... and the bubble checks, so they cannot be checks that never fail either.
    .replace("  showBubble(bubblePending(resolved, state.health && state.health.router && state.health.router.state));\n", '')
    .replace('showBubble(bubbleFromPlan(payload));', '')
    .replace("payload.kind === 'hold'", 'false');

  const checks = wiring(bypassed);
  assert.equal(checks.guardInPlan, false);
  assert.equal(checks.bodyFromPlanBody, false);
  assert.equal(checks.singleGenerateCallSite, false);
  assert.equal(checks.reloadAwaitsHealth, false);
  assert.equal(checks.allTriggersGuarded, false);
  assert.equal(checks.bubbleBeforeRequest, false);
  assert.equal(checks.bubbleFromVerdict, false);
  assert.equal(checks.holdInstallsNothing, false);
  assert.ok(Object.values(wiring(SOURCE)).every(Boolean), 'the real page passes them all');
});

// ---------------------------------------------------------------------------
// the chat surface: the drawer, the noise fallback, the status line, continuity
// ---------------------------------------------------------------------------
//
// `pure-chat` is loaded together with `pure-bubble` on purpose: the status line
// names a route, and the route's vocabulary (`ROUTE_TAGS`) is the bubble's. Two
// copies of that table would be two places to keep in step.

const CHAT_START = '// --- pure-chat: start';
const CHAT_END = '// --- pure-chat: end';

/** The marked chat region of the shipped page, verbatim, with the bubble region it reads. */
function pureChat(source) {
  return `${pureBubble(source)}\n${markedRegion(source, CHAT_START, CHAT_END)}`;
}

/** Evaluate the chat region in a fresh context and hand back its functions. */
function loadChat(context = vm.createContext({})) {
  return vm.runInContext(
    `${pureChat(SOURCE)}\n({ advancedOpen, advancedValue, noiseDecision, routerLabel, statusLine, `
    + `continuationNote, ADVANCED_STORAGE_KEY, ROUTER_LABELS });`,
    context,
    { filename: 'viewer/app.js#pure-chat' },
  );
}

const chat = loadChat();
const INDEX_SOURCE = readFileSync(join(HERE, '..', 'viewer', 'index.html'), 'utf8');

test('the advanced drawer is closed until something opens it', () => {
  // Nothing stored is the state a first visit is in, and it is closed.
  for (const stored of [undefined, null, '', 'closed', '0', '1', 'true', 'yes', 'OPEN']) {
    assert.equal(chat.advancedOpen(stored), false, `stored=${JSON.stringify(stored)}`);
  }
  // ... and it can open, which is what makes the check above a rule rather than a
  // constant: an `advancedOpen` that always said false would fail here.
  assert.equal(chat.advancedOpen('open'), true);
  assert.equal(chat.advancedValue(true), 'open');
  assert.equal(chat.advancedValue(false), 'closed');
  assert.equal(chat.ADVANCED_STORAGE_KEY, 'ardy-viewer-advanced');
});

test('the drawer ships collapsed in the markup, and the page is what opens it', () => {
  const details = INDEX_SOURCE.match(/<details[^>]*id="advanced"[^>]*>/);
  assert.ok(details, 'index.html must carry the advanced drawer as a <details>');
  assert.doesNotMatch(details[0], /\bopen\b/, 'the advanced drawer must ship collapsed');
  // The default screen keeps the picture, the conversation, the box and the line.
  for (const id of ['viewport', 'chat-log', 'composer', 'prompt-text', 'status']) {
    assert.ok(INDEX_SOURCE.includes(`id="${id}"`), `index.html must still carry #${id}`);
  }
  // The page restores and remembers the state -- a drawer that forgets is the
  // regression this test exists for.
  assert.match(SOURCE, /els\.advanced\.open = advancedOpen\(storedAdvanced\(\)\)/);
  assert.match(SOURCE, /rememberAdvanced\(els\.advanced\.open\)/);
});

test('negative control: a drawer with the open attribute fails the collapsed check', () => {
  const opened = INDEX_SOURCE.replace('<details id="advanced" class="advanced">',
    '<details id="advanced" class="advanced" open>');
  assert.notEqual(opened, INDEX_SOURCE, 'the replacement must have matched');
  const details = opened.match(/<details[^>]*id="advanced"[^>]*>/);
  assert.match(details[0], /\bopen\b/, 'the check must be able to see an open drawer');
  assert.doesNotMatch(INDEX_SOURCE.match(/<details[^>]*id="advanced"[^>]*>/)[0], /\bopen\b/);
});

test('golden noise is only used for a duration /api/health says has a recorded x_T', () => {
  const runs = [
    { duration_s: 2.0, available: true },
    { duration_s: 4.0, available: true },
  ];
  assert.deepEqual(
    plain(chat.noiseDecision({ requested: 'golden', durationS: 2, goldenNoise: runs })),
    { mode: 'golden', requested: 'golden', fellBack: false, reason: null, message: null },
  );
  // 6 s is a perfectly good plan duration with no recorded reference x_T: the page
  // was told so, so it must not find out again with a 400.
  const six = chat.noiseDecision({ requested: 'golden', durationS: 6, goldenNoise: runs });
  assert.equal(six.mode, 'random');
  assert.equal(six.fellBack, true);
  assert.equal(six.reason, 'no_golden_noise_for_duration');
  assert.match(six.message, /没有金标准噪声/);
  assert.match(six.message, /2\.0 s、4\.0 s/, 'the message names the durations that do have one');
  // Choosing random is never second-guessed, whatever is on disk.
  const random = chat.noiseDecision({ requested: 'random', durationS: 6, goldenNoise: runs });
  assert.deepEqual(plain(random),
    { mode: 'random', requested: 'random', fellBack: false, reason: null, message: null });
});

test('negative control: the fallback is not unconditional in either direction', () => {
  const runs = [{ duration_s: 2.0, available: true }];
  // An implementation that always downgraded to random would fail this ...
  assert.equal(chat.noiseDecision({ requested: 'golden', durationS: 2, goldenNoise: runs }).fellBack, false);
  // ... and one that always upgraded to golden would fail this.
  assert.equal(chat.noiseDecision({ requested: 'random', durationS: 2, goldenNoise: runs }).mode, 'random');
  // A run whose npz is not on disk is not a duration that has an x_T either.
  const missing = chat.noiseDecision({
    requested: 'golden', durationS: 2, goldenNoise: [{ duration_s: 2.0, available: false }],
  });
  assert.equal(missing.fellBack, true);
  // No health at all (an older server) is "no recorded x_T", not a crash.
  assert.equal(chat.noiseDecision({ requested: 'golden', durationS: 2 }).fellBack, true);
  assert.match(chat.noiseDecision({ requested: 'golden', durationS: 2 }).message, /一个时长都没录过/);
});

test('the status line says what is connected, what the router is, and what is happening', () => {
  const off = chat.statusLine({
    connected: true, router: { state: 'router_unavailable' }, text: '生成中…',
  });
  assert.match(off, /已连接/);
  assert.match(off, /路由：未启用/);
  assert.match(off, /生成中/);
  assert.doesNotMatch(off, /Jev/, 'a router that is not asking Jev must not say Jev');

  // A router that *is* reading the sentence says so, with the route it chose.
  const on = chat.statusLine({
    connected: true, router: { state: 'ok', route: 'compose' }, text: '就绪',
  });
  assert.match(on, /Jev 语义路由（compose）/);

  // A degraded state keeps its own name; it never becomes "ok".
  const degraded = chat.statusLine({
    connected: true, router: { state: 'router_timeout', route: null }, text: '就绪',
  });
  assert.match(degraded, /路由超时/);
  assert.doesNotMatch(degraded, /\(/, 'no route to name, so no route is named');
});

test('negative control: a status line without the router is not the status line', () => {
  // The failure this exists for: a drop in connection or router reported as a bare
  // message, so a page that is not asking anyone looks like one that is.
  const line = chat.statusLine({ connected: false, router: null, text: '连不上' });
  assert.match(line, /未连接/);
  assert.match(line, /路由：未知/);
  assert.doesNotMatch(line, /已连接/);
  // ... and the router segment is not optional.
  const withoutRouter = chat.statusLine({ connected: true, text: '就绪' });
  assert.match(withoutRouter, /路由：未知/);
});

test('continuity is reported from the server’s block, never inferred from a 200', () => {
  const payload = {
    kind: 'plan',
    continuation: {
      applied: true, reason: null, history_frames: 40,
      seam: {
        carried_frame_joint_mm: 0.0, carried_frame_root_mm: 0.0,
        model_first_joint_mm: 117.96, model_first_root_mm: 55.14,
      },
    },
  };
  const note = chat.continuationNote(payload);
  assert.equal(note.state, 'continue');
  assert.match(note.text, /接续/);
  assert.match(note.meta, /历史窗口 40 帧/);
  assert.match(note.meta, /位移 0\.000 mm/);
  assert.match(note.meta, /118\.0 mm/, 'the model’s own next step is quoted as measured');

  assert.equal(chat.continuationNote({
    continuation: { applied: false, reason: 'reset_requested' },
  }).state, 'reset');
  assert.equal(chat.continuationNote({
    continuation: { applied: false, reason: 'no_previous_clip' },
  }).state, 'fresh');
  assert.equal(chat.continuationNote({
    continuation: { applied: false, reason: 'runner_has_no_continuation_path' },
  }).state, 'unsupported');
  // A server that says nothing about continuity gets no claim about it.
  assert.equal(chat.continuationNote({ kind: 'plan' }).state, 'unknown');
});

test('negative control: an uncontinued clip is never reported as continued', () => {
  const reset = chat.continuationNote({ continuation: { applied: false, reason: 'reset_requested' } });
  assert.doesNotMatch(reset.text, /接续/);
  // ... and a continuation with no measured seam must not invent numbers.
  const noSeam = chat.continuationNote({
    continuation: { applied: true, reason: null, history_frames: 160 },
  });
  assert.equal(noSeam.state, 'continue');
  assert.doesNotMatch(noSeam.meta, /mm/, 'no seam measured, so no seam claimed');
});

test('the chat region is pure: it runs with no document, no fetch, no state', () => {
  const trap = new Proxy({}, {
    get(_target, key) { throw new Error(`the pure region touched ${String(key)}`); },
  });
  const isolated = loadChat(vm.createContext({
    document: trap, window: trap, fetch: trap, state: trap, els: trap,
    THREE: trap, performance: trap, requestAnimationFrame: trap, localStorage: trap,
  }));
  assert.equal(isolated.advancedOpen('open'), true);
  assert.equal(isolated.noiseDecision({ requested: 'random' }).mode, 'random');
  assert.match(isolated.statusLine({ connected: true, text: 'x' }), /已连接/);
  assert.equal(isolated.continuationNote({ continuation: { applied: false } }).state, 'other');
});

// ---------------------------------------------------------------------------
// the wiring of the three behaviours the user named
// ---------------------------------------------------------------------------

/** The properties of the shipped page that make the three behaviours load-bearing. */
function chatWiring(source) {
  const generate = functionBody(source, 'generateClip');
  const pull = functionBody(source, 'pullSegment');
  const intent = functionBody(source, 'setStreamIntent');
  const planBodyCalls = (source.match(/planBody\(/g) || []).length;
  const planBodyDecls = (source.match(/function planBody\(/g) || []).length;
  return {
    // 1. a prompt change is not a reset: no `reset` key is ever put in the body...
    noResetInBody: !/\breset\s*:/.test(generate) && !/\breset\s*:/.test(pull)
      && !/\breset\s*:/.test(intent),
    // ... and the body is still built by the pure builder, once, when the conditions
    //    are recorded -- not re-derived from the live controls when it is finally sent.
    bodyFromPlanBody: planBodyDecls === 1 && planBodyCalls === 2
      && intent.includes('const body = planBody({ prompt: resolved.prompt }'),
    // 2. everything that can plan goes through the one guarded entry point, and the
    //    dropdown is one of them: picking an entry sends it.
    selectSends: /els\.promptSelect\.addEventListener\('change', \(\) => \{[\s\S]{0,400}?planIfReady\('select'\)/
      .test(source),
    enterSends: source.includes("planIfReady('enter')"),
    // 3. the noise that is sent is the decision, not the raw control -- and it is
    //    decided before the body is built.
    noiseFromDecision: /const noise = applyNoiseDecision\(\)\.mode;/.test(intent)
      && intent.indexOf('applyNoiseDecision().mode') < intent.indexOf('const body = planBody('),
    // The fallback is applied on every control that can move the duration.
    noiseFallbackOnControls: (source.match(/applyNoiseDecision\(\)/g) || []).length >= 4,
    // The response's own continuation block is what the page draws.
    announceFromPayload: /announceContinuation\(payload\)/.test(pull),
    // 4. a new sentence lands on the *next* segment, never inside the one on screen:
    //    the entry point records conditions and returns the clip already playing, and
    //    the fetch is what commits them.
    intentFrozenAtBoundary: generate.includes('setStreamIntent(resolved, { force: wasEmpty })')
      && generate.includes('return state.segments[state.segments.length - 1];')
      && generate.includes('return pullSegment({ quiet: false });')
      && pull.includes('const body = intent.body;'),
    // 5. exactly one call site of `planBody(`, so there is no second request path that
    //    could carry different conditions.
    singlePlanBodyCallSite: planBodyDecls === 1 && planBodyCalls === 2,
  };
}

test('wiring: no reset, select sends, and the noise is the decision', () => {
  const checks = chatWiring(SOURCE);
  for (const [name, ok] of Object.entries(checks)) {
    assert.ok(ok, `viewer/app.js chat wiring check failed: ${name}`);
  }
  // The pure builder really does leave `reset` out of the body, which is what "not
  // passing reset" means on the wire.
  const sent = JSON.parse(JSON.stringify(
    guard.planBody({ prompt: 'walk' }, { seed: 0, duration: 2, noise: 'random' }),
  ));
  assert.deepEqual(Object.keys(sent).sort(), ['duration', 'noise', 'prompt', 'seed']);
});

test('negative control: each of those wiring checks can go red', () => {
  const broken = SOURCE
    .replace('const noise = applyNoiseDecision().mode;', 'const noise = els.noiseSelect.value;')
    .replace("planIfReady('select');", 'updatePromptHelp();')
    .replace('announceContinuation(payload);', '')
    .replace('const body = planBody({ prompt: resolved.prompt }, { seed, duration, noise });',
      'const body = { prompt: resolved.prompt, seed, duration, noise, reset: true };');
  const checks = chatWiring(broken);
  assert.equal(checks.noResetInBody, false);
  assert.equal(checks.selectSends, false);
  assert.equal(checks.noiseFromDecision, false);
  assert.equal(checks.announceFromPayload, false);
  assert.equal(checks.bodyFromPlanBody, false);
  assert.equal(checks.singlePlanBodyCallSite, false);
  assert.ok(Object.values(chatWiring(SOURCE)).every(Boolean), 'the real page passes them all');
});

test('negative control: the docs’ own claim is checked against the markup it describes', () => {
  // The drawer summarises what continuity means, including the one thing that is
  // *not* continuous. A page that claims otherwise is the failure this guards.
  assert.match(INDEX_SOURCE, /<b>acceleration is not<\/b>/);
  assert.match(INDEX_SOURCE, /foot skating/);
  assert.match(INDEX_SOURCE, /NOT<\/b> aligned with the golden clip|not<\/b> aligned with the golden clip/);
});

// ---------------------------------------------------------------------------
// the applied-controls line: carrying a parameter is not applying it
// ---------------------------------------------------------------------------

/** A `/api/plan` payload with the server's `controls` block attached. */
function controlledPayload({ applied = ['speed', 'amplitude'], identity = false,
  placeholder = true } = {}) {
  const payload = planPayload({ message: '用现有动作组合：速度 1.35×、幅度 1.40×。' });
  // The router's own numbers, as `/api/plan` sends them under `decision.params`.
  payload.decision.params = { speed: 1.35, amplitude: 1.4, expression: 'smile', gaze: 'forward' };
  payload.controls = {
    speed: 1.35,
    amplitude: 1.4,
    applied,
    expression_placeholder: placeholder,
    output_frames: 40,
    source_frames: 80,
    saturated_fraction: 0,
    preserve_first_frame: false,
    identity,
  };
  return payload;
}

test('a plan that applied its parameters says so, and names the placeholders', () => {
  const view = bubble.bubbleFromPlan(controlledPayload());
  assert.match(view.meta, /applied=speed\+amplitude/);
  // The rig has no face or eyes; the page must not imply the expression reached it.
  assert.match(view.meta, /expression\/gaze=占位（未生效）/);
  // The parameters are still shown as what was read, next to what was applied.
  assert.match(view.meta, /params=/);
});

test('a neutral decision is reported as applying nothing, not as applying 1.0x', () => {
  const view = bubble.bubbleFromPlan(controlledPayload({ applied: [], identity: true }));
  assert.match(view.meta, /applied=none/);
  assert.doesNotMatch(view.meta, /applied=speed/);
});

test('negative control: a payload with no controls block claims no application', () => {
  // Without the server's block the page cannot know, so it must say nothing about it
  // rather than infer "applied" from the presence of `params`.
  const view = bubble.bubbleFromPlan(planPayload());
  assert.doesNotMatch(view.meta, /applied=/);
  assert.doesNotMatch(view.meta, /占位/);
});

// ---------------------------------------------------------------------------
// the playback model: a queue that never loops, and the negative controls for it
// ---------------------------------------------------------------------------
//
// `docs/decisions/0009` §二 froze this as the thing the reader was actually seeing
// ("不停地 playback / loop") and named four falsifiable criteria. Each one below is
// paired with a negative control built from a real alternative implementation --
// the loop checkbox, a naive `setTrail`-style replacement, a per-segment camera
// refit -- rather than from a mock, so a check that cannot fail would be obvious.

const JOINT_COUNT = 2;
const CONTACT_COUNT = 4;
const STREAM_FPS = 20;

/**
 * A clip shaped like the server's, as first-frame-carries-over.
 *
 * `base` is the world XZ offset of this segment's body, so successive segments walk
 * away from one another and "did the camera/trail/position move" is measurable.
 */
function fakeSegment({ frames = 5, base = 0, version = 1, intent = null } = {}) {
  const joints = new Float32Array(frames * JOINT_COUNT * 3);
  const roots = new Float32Array(frames * 3);
  const contacts = new Uint8Array(frames * CONTACT_COUNT);
  for (let frame = 0; frame < frames; frame += 1) {
    const z = base + frame;
    for (let j = 0; j < JOINT_COUNT; j += 1) {
      joints[(frame * JOINT_COUNT + j) * 3] = j;
      joints[(frame * JOINT_COUNT + j) * 3 + 1] = 0.8 + j * 0.1;
      joints[(frame * JOINT_COUNT + j) * 3 + 2] = z;
    }
    roots[frame * 3] = 0;
    roots[frame * 3 + 1] = 0.8;
    roots[frame * 3 + 2] = z;
    contacts[frame * CONTACT_COUNT] = frame % 2;
  }
  return {
    frames, jointCount: JOINT_COUNT, fps: STREAM_FPS,
    joints, roots, contacts, intentVersion: version,
    meta: { frames, embedding: {}, intent },
  };
}

/**
 * Play a queue the way the page does: one `advancePlayhead` per frame, with a
 * prefetch whenever the decision asks for one. The returned trace is the evidence
 * the monotonicity criteria are read off.
 */
function playRun({ segments, frames = 140, runaheadFrames = 4, loopSingleSegment = false }) {
  const queue = segments.slice();
  const trace = [];
  const assertClock = [];
  let playhead = 0;
  let held = false;
  for (let step = 0; step < frames; step += 1) {
    const tl = stream.timelineOf(queue, JOINT_COUNT);
    const decision = stream.advancePlayhead({
      segments: queue, timeline: tl, playhead, deltaSeconds: 1 / STREAM_FPS,
      fps: STREAM_FPS, runaheadFrames, loopSingleSegment,
    });
    if (decision.playhead < playhead) {
      // Only the explicit loop may do this, and it says so.
      assert.ok(decision.wrapped, `playhead went backwards without the loop: ${playhead} -> ${decision.playhead}`);
    }
    playhead = decision.playhead;
    held = decision.held;
    trace.push({
      step, playhead, held,
      depth: stream.bufferDepth(tl, playhead),
      points: stream.trailPoints(queue).length,
      timelineFrames: tl.frameCount,
    });
    // Runahead, exactly as the render loop does it: the decision is recomputed from
    // the *playhead* every frame, and the conditions the last segment used are reused
    // (`nextSegmentIntent`) rather than re-decided -- one sentence buys as many
    // segments as the runway needs. The harness must mirror the shipped loop, or it
    // would be playing a model the page does not have.
    const runahead = stream.shouldPrefetch({
      fetching: false, nextIntent: null, committedIntent: { version: 1 },
      timeline: tl, playhead, runaheadFrames,
    });
    const intent = stream.nextSegmentIntent({
      segments: queue, nextIntent: null, committedIntent: { version: 1 }, fetching: false,
    });
    if (runahead.fetch && intent.intent && queue.length < segments.length) {
      queue.push(segments[queue.length]);
      assertClock.push(SOURCE.includes('pullSegment({ quiet: !first });'));
    }
  }
  return { trace, held, playhead, queue, assertClock };
}

/** Five segments is the N >= 5 the criterion asks for. */
const FIVE = Array.from({ length: 5 }, (_, i) => fakeSegment({ frames: 6, base: i * 6, version: 1 }));

test('criterion 1: the playhead is monotone across five segments', () => {
  const { trace, queue } = playRun({ segments: FIVE, frames: 200 });
  assert.equal(queue.length, 5, 'all five segments must have been appended');
  for (let i = 1; i < trace.length; i += 1) {
    assert.ok(trace[i].playhead >= trace[i - 1].playhead,
      `playhead went backwards at frame ${i}: ${trace[i - 1].playhead} -> ${trace[i].playhead}`);
  }
  const last = trace[trace.length - 1];
  assert.ok(last.timelineFrames >= 26,
    `the run must span the whole queue, not the last clip: ${last.timelineFrames} frames`);
  assert.ok(last.playhead > 0);
});

test('criterion 1 negative control: the loop checkbox makes the playhead go back', () => {
  // The loop is a real code path on the shipped page, not a mock: with it on, the
  // position rewinds at the end of each segment and the monotonicity check above
  // must fail. This is what the criterion is falsified by.
  const { trace } = playRun({ segments: [FIVE[0]], frames: 60, loopSingleSegment: true });
  const backwards = trace.filter((entry, i) => i > 0 && entry.playhead < trace[i - 1].playhead);
  assert.ok(backwards.length > 0, 'with the loop on, at least one frame must rewind');
  const monotone = trace.every((entry, i) => i === 0 || entry.playhead >= trace[i - 1].playhead);
  assert.equal(monotone, false, 'the monotonicity criterion must go red when the loop is on');
});

test('criterion 1 negative control: a wrap at the segment end is a rewind', () => {
  // The other way this criterion can fail, and the one the page actually shipped:
  // `playhead -= frameCount - 1` at the end of a clip. modelled here on the same
  // inputs as the positive case, it produces a *decreasing* position while the loop
  // checkbox is off -- which is exactly what `advancePlayhead` must never do.
  const single = [fakeSegment({ frames: 6, base: 0, version: 1 })];
  const tl = stream.timelineOf(single, JOINT_COUNT);
  const wrap = (playhead) => {
    let next = playhead + 1;
    if (next > tl.maxFrame) next -= tl.maxFrame;   // the old loop, verbatim
    return next;
  };
  const wrapped = [0, 1, 2, 3, 4, 5].map((p) => wrap(p));
  assert.ok(wrapped[wrapped.length - 1] < wrapped[wrapped.length - 2],
    `the old wrap really does go backwards: ${wrapped.join(',')}`);
  // ... and the shipped advance, on the same input, does not.
  const shipped = [0, 1, 2, 3, 4, 5].map((p) => stream.advancePlayhead({
    segments: single, timeline: tl, playhead: p, deltaSeconds: 1 / STREAM_FPS,
    fps: STREAM_FPS, runaheadFrames: 4, loopSingleSegment: false,
  }).playhead);
  for (let i = 1; i < shipped.length; i += 1) {
    assert.ok(shipped[i] >= shipped[i - 1], `shipped advance went backwards: ${shipped.join(',')}`);
  }
  assert.equal(shipped[shipped.length - 1], tl.maxFrame, 'it parks on the last frame instead');
});

test('criterion 2: the trail only grows while segments are appended', () => {
  const { trace } = playRun({ segments: FIVE, frames: 200 });
  for (let i = 1; i < trace.length; i += 1) {
    assert.ok(trace[i].points >= trace[i - 1].points,
      `trail shrank at frame ${i}: ${trace[i - 1].points} -> ${trace[i].points}`);
    if (trace[i].timelineFrames > trace[i - 1].timelineFrames) {
      assert.ok(trace[i].points > trace[i - 1].points, 'a new segment must add trail points');
    }
  }
});

test('criterion 2 negative control: a naive re-append double-counts every seam', () => {
  // What the page used to do -- replace the trail with the newest clip's path -- and
  // what a naive "concatenate every segment's roots" does instead. Both are kept as
  // real functions so this control is not a mock.
  const naive = stream.trailUnderNaiveAppend(FIVE);
  const shipped = stream.trailPoints(FIVE);
  assert.equal(naive.length, 5 * 6, 'the naive path counts the carried-over frame twice');
  assert.equal(shipped.length, 5 * 6 - 4, 'the shipped path drops exactly one seam frame per seam');
  assert.notEqual(naive.length, shipped.length);

  // And the *replacement* failure mode: the trail reset to the newest segment.
  const replaced = stream.trailUnderNaiveAppend([FIVE[4]]);
  assert.ok(replaced.length < stream.trailUnderNaiveAppend(FIVE.slice(0, 4)).length,
    'replacing the trail with the newest clip makes the point count fall');
});

test('criterion 3: the camera may only be refitted once, or on an explicit Fit', () => {
  assert.equal(stream.shouldFitCamera({ haveSegments: false, userRequested: false }), true,
    'the first segment needs a frame');
  assert.equal(stream.shouldFitCamera({ haveSegments: true, userRequested: false }), false,
    'a later segment must not move the camera');
  assert.equal(stream.shouldFitCamera({ haveSegments: true, userRequested: true }), true,
    'pressing Fit always refits');
});

test('criterion 3 negative control: a per-segment refit really would move the camera', () => {
  // The negative control is measurable, not asserted: the focus of the whole-run
  // bounds moves by metres as segments are appended, so refitting per segment would
  // move the camera. The page does not, so its view state is bit-identical.
  const one = stream.timelineOf([FIVE[0]], JOINT_COUNT);
  const five = stream.timelineOf(FIVE, JOINT_COUNT);
  const centre = (bounds) => [0, 1, 2].map((axis) => (bounds.min[axis] + bounds.max[axis]) / 2);
  const before = centre(one.bounds);
  const after = centre(five.bounds);
  const moved = Math.abs(after[2] - before[2]);
  assert.ok(moved > 1, `the run's centre must actually move (moved ${moved.toFixed(3)} m on Z)`);

  // What "the camera did not move" means: appending changes no camera field. The
  // page's append path has no write to `cameraView` outside the fit branch.
  const appendBody = functionBody(SOURCE, 'appendSegment');
  const fitAt = appendBody.indexOf('shouldFitCamera(');
  assert.ok(fitAt >= 0, 'the fit must be conditional, not unconditional');
  assert.equal(appendBody.indexOf('cameraView.distance ='), -1,
    'nothing in the append path may set the camera directly');
  const perSegment = appendBody.replace('shouldFitCamera({ haveSegments: !first })', 'true');
  assert.notEqual(perSegment, appendBody, 'the negative control must actually change the source');
  assert.ok(/if \(true\) fitCameraToClip\(\);/.test(perSegment),
    'with the refit forced, the camera is refitted per segment (this is the red case)');
});

// ---------------------------------------------------------------------------
// the follow camera: the character stays in the picture, and the camera never jumps
// ---------------------------------------------------------------------------
//
// The symptom this section exists for: 0009 §二's "frame once, or on Fit" rule is
// correct about *re-fitting*, and the page implemented it as "frame once, then never
// move". A character who walks more than a few metres then walks out of the shot,
// and the reader is looking at an empty floor with the position counter still
// climbing. The fix must not re-introduce the jump 0009 forbids, so there are two
// criteria and three controls:
//
//   6. the character stays in the picture: the root *and* a head above it project
//      inside the frustum on every frame of a walk.
//   7. the camera never jumps: no frame moves the focus, or the camera position, by
//      `CAMERA_MAX_STEP_M` or more, and the damping never has to clamp.
//   8. following writes `focus` and nothing else -- the distance and the orbit stay
//      the reader's, so "no re-framing without a user action" still holds.
//
// The controls are real alternative implementations, not mocks: follow switched off,
// a per-segment `fitCameraToClip`, and `followFocusInstant`.

const WALK_STEP_M = 0.05;   // 1 m/s at the clip's 20 fps: a walk, not a teleport
const WALK_FRAMES = 40;     // one generation horizon
const WALK_SEGMENTS = 12;   // ~23 m of travel, enough to walk past a static camera
const VIEW_ASPECT = 16 / 9;
const HEAD_ABOVE_ROOT_M = 0.85;

/**
 * `fakeSegment`'s shape, walking `step` metres per frame along +Z.
 *
 * `fakeSegment` itself advances 1 m per frame -- 20 m/s, a teleport -- which is fine
 * for the queue tests and useless for a camera criterion.
 */
function walkSegment({ frames = WALK_FRAMES, base = 0, step = WALK_STEP_M } = {}) {
  const segment = fakeSegment({ frames });
  for (let frame = 0; frame < frames; frame += 1) {
    const z = base + frame * step;
    for (let j = 0; j < JOINT_COUNT; j += 1) {
      segment.joints[(frame * JOINT_COUNT + j) * 3 + 2] = z;
    }
    segment.roots[frame * 3 + 1] = 0.90;
    segment.roots[frame * 3 + 2] = z;
  }
  return segment;
}

/**
 * `count` consecutive walk segments.
 *
 * Segment N's frame 0 is the previous segment's last frame, carried over verbatim,
 * so each segment after the first starts `frames - 1` steps further along -- the
 * same off-by-one the whole playback model is built around.
 */
function walkQueue(count, { frames = WALK_FRAMES, step = WALK_STEP_M } = {}) {
  const stride = (frames - 1) * step;
  return Array.from({ length: count }, (unused, index) => (
    walkSegment({ frames, step, base: index * stride })
  ));
}

const distance3 = (a, b) => Math.hypot(a.x - b.x, a.y - b.y, a.z - b.z);

/**
 * Play a queue one frame at a time and report what the camera did each frame.
 *
 * The camera starts where the **shipped opening fit** put it -- the centre of the
 * first segment's bounds and that bounds' fit distance -- because that is the state
 * the page is really in when playback begins. `refit(frame)` optionally re-places
 * the camera before a frame is drawn, which is how the per-segment-refit control is
 * built without touching the page.
 */
function cameraRun({ segments, frames = null, follow = true, instant = false, refit = null,
  targetFrame = null }) {
  const timeline = stream.timelineOf(segments, JOINT_COUNT);
  const opening = stream.timelineOf(segments.slice(0, 1), JOINT_COUNT).bounds;
  const view = {
    yaw: 0.7,
    pitch: 0.22,
    distance: stream.fitDistanceFromBounds(opening),
    focus: { ...stream.fitFocusFromBounds(opening) },
  };
  const total = frames === null ? timeline.frameCount : frames;
  const trace = [];
  for (let frame = 0; frame < total; frame += 1) {
    if (refit) {
      const placed = refit(frame);
      if (placed) {
        view.focus = placed.focus;
        view.distance = placed.distance;
      }
    }
    // Where the character *is*, and where the follow *thinks* it is. They are the
    // same on the shipped page; the negative controls below pull them apart, and the
    // criterion is always about the character, never about the target.
    const actual = stream.focusTargetAt({
      roots: timeline.roots, frame, frameCount: timeline.frameCount,
    });
    const target = stream.focusTargetAt({
      roots: timeline.roots,
      frame: targetFrame ? targetFrame(frame) : frame,
      frameCount: timeline.frameCount,
    });
    const focusBefore = { ...view.focus };
    const eyeBefore = stream.cameraPositionFor(view);

    let focusStep = 0;
    let clamped = false;
    if (follow) {
      const stepped = instant
        ? stream.followFocusInstant(view.focus, target)
        : stream.followFocus(view.focus, target, 1 / STREAM_FPS);
      view.focus = stepped.focus;
      focusStep = stepped.distance;
      clamped = stepped.clamped;
    }

    const eye = stream.cameraPositionFor(view);
    const pelvis = stream.projectPoint(actual, view, { aspect: VIEW_ASPECT });
    const head = stream.projectPoint(
      { x: actual.x, y: actual.y + HEAD_ABOVE_ROOT_M, z: actual.z },
      view,
      { aspect: VIEW_ASPECT },
    );
    trace.push({
      frame,
      focusStep,
      focusJump: distance3(view.focus, focusBefore),
      cameraStep: distance3(eye, eyeBefore),
      clamped,
      eye,
      pelvis,
      head,
      onScreen: pelvis.onScreen && head.onScreen,
      ndc: Math.max(Math.abs(pelvis.x), Math.abs(pelvis.y), Math.abs(head.x), Math.abs(head.y)),
    });
  }
  return { timeline, view, trace };
}

test('the projection is the page\'s real one: a point at the focus is centred', () => {
  // The on-screen criterion below is only meaningful if the projection is right, so
  // check it against the things that are defined by hand: the focus projects to the
  // centre at the framing distance, the camera sits `distance` from the focus, and a
  // point behind the camera is not "in front".
  const view = { yaw: 0.7, pitch: 0.22, distance: 5, focus: { x: 0, y: 0.9, z: 0 } };
  const centre = stream.projectPoint(view.focus, view, { aspect: VIEW_ASPECT });
  assert.ok(Math.abs(centre.x) < 1e-9 && Math.abs(centre.y) < 1e-9, 'the focus is the centre');
  assert.ok(Math.abs(centre.depth - 5) < 1e-9, `depth is the framing distance (got ${centre.depth})`);
  assert.equal(centre.onScreen, true);

  const eye = stream.cameraPositionFor(view);
  assert.ok(Math.abs(distance3(eye, view.focus) - 5) < 1e-9, 'the eye is `distance` from the focus');

  // Behind the eye by 1 m along the view direction.
  const behind = stream.projectPoint({ x: eye.x + eye.x * 0.2, y: eye.y, z: eye.z + eye.z * 0.2 }, view, { aspect: VIEW_ASPECT });
  assert.equal(behind.inFront, false, 'a point behind the camera is not in front of it');
  assert.equal(behind.onScreen, false);

  // Well to the side: outside the frustum, and it must say so.
  const side = stream.projectPoint({ x: 40, y: 0.9, z: 0 }, view, { aspect: VIEW_ASPECT });
  assert.equal(side.onScreen, false, '40 m to the side is not on screen');
  assert.ok(Math.abs(side.x) > 1);
});

test('criterion 6: with the follow on, the character stays in the picture for the whole run', () => {
  const { timeline, trace } = cameraRun({ segments: walkQueue(WALK_SEGMENTS) });
  assert.ok(timeline.frameCount > 400, `the run must be long (${timeline.frameCount} frames)`);

  const missing = trace.filter((entry) => !entry.onScreen);
  const worst = trace.reduce((best, entry) => (entry.ndc > best.ndc ? entry : best), trace[0]);
  assert.deepEqual(
    missing.map((entry) => entry.frame),
    [],
    `the character left the frame on ${missing.length} frames; worst |ndc| = `
    + `${worst.ndc.toFixed(3)} at frame ${worst.frame}`,
  );
  // ... and it is not merely "on screen": it is well inside it.
  assert.ok(worst.ndc < 0.8, `the character must stay well inside the frustum (worst ${worst.ndc.toFixed(3)})`);
});

test('criterion 6 negative control: without the follow, the same run walks out of the shot', () => {
  // This is the page as it shipped: framed once at the first segment, then never
  // moved. On the same walk the character is on screen at the start and off it later
  // -- which is exactly the reader's "似乎不太对".
  const still = cameraRun({ segments: walkQueue(WALK_SEGMENTS), follow: false });
  const followed = cameraRun({ segments: walkQueue(WALK_SEGMENTS) });

  const early = still.trace.slice(0, 40);
  const late = still.trace.slice(-40);
  assert.ok(early.every((entry) => entry.onScreen), 'the shot is fine at the start');
  assert.ok(late.every((entry) => !entry.onScreen),
    'with the camera static, the character must be out of frame by the end');
  const firstGone = still.trace.find((entry) => !entry.onScreen);
  assert.ok(firstGone, 'there must be a first frame where the character is gone');
  assert.ok(Math.abs(firstGone.pelvis.depth) < 1e6);

  // The shipped follow does not: it is on screen on every frame of the same run.
  assert.ok(followed.trace.every((entry) => entry.onScreen),
    'the followed run must never lose the character');
  // Quantified: the static camera watches the character's |ndc| grow by an order of
  // magnitude, which is the measurable difference between the two policies.
  const stillWorst = Math.max(...late.map((entry) => entry.ndc));
  const followedWorst = Math.max(...followed.trace.map((entry) => entry.ndc));
  assert.ok(stillWorst > followedWorst * 3,
    `static worst |ndc| ${stillWorst.toFixed(2)} vs followed ${followedWorst.toFixed(2)}`);
});

test('criterion 6 negative control: a follow that chased a frozen frame loses the character too', () => {
  // The other plausible half-fix: run the follow, but read the root from the wrong
  // frame -- frame 0, or the last frame of whatever was loaded first. The camera then
  // tracks a *point in the past* perfectly, and the character walks out of frame just
  // as it did with no follow at all. The target frame is what makes the follow real.
  const segments = walkQueue(WALK_SEGMENTS);
  const frozen = cameraRun({ segments, targetFrame: () => 0 });
  assert.ok(frozen.trace.slice(0, 40).every((entry) => entry.onScreen), 'it starts fine');
  assert.ok(frozen.trace.slice(-40).every((entry) => !entry.onScreen),
    'and it ends with the character gone, exactly like no follow at all');

  const followed = cameraRun({ segments });
  assert.ok(followed.trace.every((entry) => entry.onScreen));
  const followBody = functionBody(SOURCE, 'followTargetNow');
  assert.ok(followBody.includes('Math.round(state.playhead)'),
    'the shipped target frame is the playhead, which is what the control above breaks');
});

test('criterion 7: the follow never jumps -- no frame reaches the threshold, and it never clamps', () => {
  const { trace } = cameraRun({ segments: walkQueue(WALK_SEGMENTS) });
  const maxFocus = Math.max(...trace.map((entry) => entry.focusStep));
  const maxCamera = Math.max(...trace.map((entry) => entry.cameraStep));
  const maxJump = Math.max(...trace.map((entry) => entry.focusJump));
  const clamped = trace.filter((entry) => entry.clamped).length;

  assert.ok(maxFocus < stream.CAMERA_MAX_STEP_M,
    `no frame may move the focus by the threshold: worst ${maxFocus.toFixed(3)} m`);
  assert.ok(maxCamera < stream.CAMERA_MAX_STEP_M,
    `no frame may move the camera by the threshold: worst ${maxCamera.toFixed(3)} m`);
  assert.ok(maxJump < stream.CAMERA_MAX_STEP_M,
    `no frame may jump the focus: worst ${maxJump.toFixed(3)} m`);
  assert.equal(clamped, 0, 'the damping alone must never need the clamp on a walk');

  // And the margin is real, not luck. The opening fit leaves a 1.1 m gap to the
  // character's first root; at tau = 0.30 s and dt = 0.05 s a step closes
  // 1 - e^{-1/6} = 15.4% of it, i.e. 0.17 m -- a third of the threshold.
  assert.ok(maxFocus < 0.25,
    `the damping must have room: worst ${maxFocus.toFixed(3)} m vs a 0.5 m threshold`);
  // The steady-state step is the root motion itself (0.05 m), not the gap.
  const steady = Math.max(...trace.slice(60).map((entry) => entry.focusStep));
  assert.ok(steady < 3 * WALK_STEP_M, `steady-state step ${steady.toFixed(3)} m is one frame of walking`);
});

test('criterion 7 negative control: a per-segment refit really does jump the camera', () => {
  // The failure 0009 §二 forbids, reconstructed: `fitCameraToClip()` at every seam,
  // against the *cumulative* bounds. The focus is thrown to the running clip's centre
  // (+1 m per seam here) and the distance to that clip's reach (5.1 -> 10.1 -> 15.2
  // -> ... -> 24, clamped), so the camera position jumps metres in one frame.
  const stride = WALK_FRAMES - 1;
  const refit = (frame) => {
    if (frame === 0 || frame % stride !== 0) return null;
    const upto = walkQueue(WALK_SEGMENTS).slice(0, frame / stride + 1);
    const bounds = stream.timelineOf(upto, JOINT_COUNT).bounds;
    return {
      focus: stream.fitFocusFromBounds(bounds),
      distance: stream.fitDistanceFromBounds(bounds),
    };
  };
  const { trace } = cameraRun({ segments: walkQueue(WALK_SEGMENTS), follow: true, refit });
  const worst = trace.reduce((best, entry) => (entry.cameraStep > best.cameraStep ? entry : best), trace[0]);
  assert.ok(worst.cameraStep >= stream.CAMERA_MAX_STEP_M,
    `a per-segment refit must trip the threshold (worst ${worst.cameraStep.toFixed(3)} m at frame ${worst.frame})`);
  assert.ok(worst.focusJump >= stream.CAMERA_MAX_STEP_M,
    `... and it jumps the focus too (${worst.focusJump.toFixed(3)} m)`);
  // The shipped policy, on the same queue, does neither.
  const shipped = cameraRun({ segments: walkQueue(WALK_SEGMENTS) });
  assert.ok(Math.max(...shipped.trace.map((entry) => entry.cameraStep)) < stream.CAMERA_MAX_STEP_M);
});

test('criterion 7 negative control: an instant follow snaps by the whole standing gap', () => {
  // The other way to "follow": set `focus = root` every frame. It tracks perfectly,
  // and it still jumps -- at the moment it takes over from the opening fit, by the
  // whole distance between the fit's centre and the character. Damping is the
  // difference, and this is what shows the threshold can tell them apart.
  const segments = walkQueue(WALK_SEGMENTS);
  const instant = cameraRun({ segments, instant: true });
  const damped = cameraRun({ segments });
  const instantWorst = Math.max(...instant.trace.map((entry) => entry.focusStep));
  const dampedWorst = Math.max(...damped.trace.map((entry) => entry.focusStep));

  assert.ok(instantWorst >= stream.CAMERA_MAX_STEP_M,
    `an instant snap must trip the threshold (${instantWorst.toFixed(3)} m)`);
  assert.ok(dampedWorst < stream.CAMERA_MAX_STEP_M,
    `the shipped damping must not (${dampedWorst.toFixed(3)} m)`);
  assert.ok(instantWorst > 4 * dampedWorst,
    `the gap closed in one frame vs a sixth of it per frame: ${instantWorst.toFixed(3)} vs ${dampedWorst.toFixed(3)}`);
  // Both stay on screen -- this control is about the jump, not the tracking.
  assert.ok(instant.trace.every((entry) => entry.onScreen));

  // And `followFocus` is what the page calls, not the control.
  const followBody = functionBody(SOURCE, 'applyFollow');
  assert.ok(followBody.includes('followFocus('), 'the render path must use the damped follow');
  assert.ok(!followBody.includes('followFocusInstant('), 'the instant variant is a control, not a page path');
});

test('criterion 8: following writes the focus and nothing else', () => {
  // "No re-framing without a user action" (0009 §二) has to survive the follow, so
  // the follow path may not touch the distance or the orbit. The render loop is the
  // only caller, and this is its contract.
  const followBody = functionBody(SOURCE, 'applyFollow');
  assert.ok(followBody.length > 0, 'applyFollow must exist');
  assert.equal(followBody.indexOf('cameraView.distance'), -1,
    'the follow may not set the distance');
  assert.equal(followBody.indexOf('cameraView.yaw'), -1, 'the follow may not set the yaw');
  assert.equal(followBody.indexOf('cameraView.pitch'), -1, 'the follow may not set the pitch');
  assert.ok(followBody.includes('cameraView.focus.set('), 'the follow moves the focus');

  // ... and it is actually *called*, every rendered frame. A `followFocus` that is
  // never invoked is a camera that never follows.
  const frameBody = functionBody(SOURCE, 'frame');
  assert.ok(frameBody.includes('applyFollow('), 'the render loop must run the follow');
  assert.ok(frameBody.includes('camera.lookAt(cameraView.focus)'),
    'and it must still aim at the focus');

  // The follow target is the frame *the playhead is on* -- not frame 0, not the last
  // frame. A frozen target is a camera that watches the character walk away, which
  // is the criterion-6 negative control again with extra steps.
  const targetBody = functionBody(SOURCE, 'followTargetNow');
  assert.ok(targetBody.includes('Math.round(state.playhead)'),
    'the follow target must be the frame the playhead is on');

  // The only writes to the camera outside the follow and the reader's own handlers
  // are in `fitCameraToClip` (which is what the reader presses) and `resetCameraView`.
  const distanceWrites = [...SOURCE.matchAll(/cameraView\.distance =/g)].length;
  assert.ok(distanceWrites <= 3, `distance is written in the reader's handlers only (${distanceWrites} sites)`);
});

test('criterion 8: a reader Fit lands on the character, the opening framing on the clip', () => {
  // `fitPlacement` is the whole decision, so it can be run rather than read.
  const segments = walkQueue(WALK_SEGMENTS);
  const timeline = stream.timelineOf(segments, JOINT_COUNT);
  const bounds = timeline.bounds;
  const lastRoot = stream.focusTargetAt({
    roots: timeline.roots, frame: timeline.frameCount - 1, frameCount: timeline.frameCount,
  });

  const requested = stream.fitPlacement({ userRequested: true, bounds, target: lastRoot });
  assert.equal(distance3(requested.focus, lastRoot), 0, 'a reader Fit lands on the character');
  assert.equal(requested.distance, stream.fitDistanceFromBounds(bounds));

  const opening = stream.fitPlacement({ userRequested: false, bounds, target: lastRoot });
  const centre = stream.fitFocusFromBounds(bounds);
  assert.equal(distance3(opening.focus, centre), 0, 'the opening framing lands on the clip');
  assert.equal(distance3(opening.focus, lastRoot) > 10, true, 'and that is not the character, late in a run');

  // No character yet (nothing planned): the reader's Fit falls back to the clip
  // rather than crashing or leaving the camera somewhere undefined.
  const noTarget = stream.fitPlacement({ userRequested: true, bounds, target: null });
  assert.equal(distance3(noTarget.focus, centre), 0);
  assert.equal(stream.fitPlacement({ userRequested: true, bounds: null, target: null }), null);
});

test('criterion 8 negative control: a Fit that framed the run instead of the character would jump', () => {
  // The plausible mistake, made testable by `fitPlacement`: pass `userRequested:
  // false` (the opening framing) when the reader pressed Fit. On a 23 m run the
  // whole-run centre is metres from where the character actually is.
  const segments = walkQueue(WALK_SEGMENTS);
  const timeline = stream.timelineOf(segments, JOINT_COUNT);
  const bounds = timeline.bounds;
  const lastRoot = stream.focusTargetAt({
    roots: timeline.roots, frame: timeline.frameCount - 1, frameCount: timeline.frameCount,
  });
  const swapped = stream.fitPlacement({ userRequested: false, bounds, target: lastRoot });
  const shipped = stream.fitPlacement({ userRequested: true, bounds, target: lastRoot });
  const gap = distance3(swapped.focus, lastRoot);
  assert.ok(gap > 10, `the swapped Fit is metres from the character (${gap.toFixed(1)} m)`);
  assert.ok(gap >= stream.CAMERA_MAX_STEP_M, 'so it is a jump, not a re-frame');
  assert.equal(distance3(shipped.focus, lastRoot), 0, 'the shipped Fit has nothing to jump to');

  const fitBody = functionBody(SOURCE, 'fitCameraToClip');
  assert.ok(fitBody.includes('fitPlacement({'), 'the fit must go through the pure decision');
  assert.ok(fitBody.includes('userRequested: Boolean(given.userRequested)'),
    'and it must pass the reader\'s intent through');
  const clickWiring = SOURCE.slice(SOURCE.indexOf('els.fitCamera.addEventListener'));
  assert.ok(clickWiring.slice(0, 120).includes('userRequested: true'),
    'the Fit button must be wired as a reader action, not as the opening frame');
  const appendBody = functionBody(SOURCE, 'appendSegment');
  assert.ok(appendBody.includes('fitCameraToClip();'),
    'the opening framing is the reader-free call, so it gets the clip, not the character');
});

test('criterion 4: a run out of buffer holds the last frame and says so', () => {
  // One segment, 41 frames, and no more segments to append: playback must park on
  // the last frame. It must not wrap, and it must say the buffer ran out.
  const single = [fakeSegment({ frames: 41, base: 0, version: 1 })];
  const tl = stream.timelineOf(single, JOINT_COUNT);
  let playhead = 0;
  let firstHold = null;
  for (let step = 0; step < 60; step += 1) {
    const decision = stream.advancePlayhead({
      segments: single, timeline: tl, playhead, deltaSeconds: 1, fps: STREAM_FPS,
      runaheadFrames: 4, loopSingleSegment: false,
    });
    playhead = decision.playhead;
    if (decision.held && firstHold === null) firstHold = step;
    assert.ok(decision.playhead <= tl.maxFrame, 'the playhead may never pass the last frame');
  }
  assert.notEqual(firstHold, null, 'the run must reach and report a hold');
  assert.equal(playhead, tl.maxFrame, 'it holds on the last frame');

  const note = stream.bufferNote({ timeline: tl, playhead, fps: STREAM_FPS, held: true });
  assert.equal(note.state, 'dry');
  assert.equal(note.depth, 0);
  assert.equal(note.text, stream.BUFFER_EMPTY_HINT);
  assert.match(note.text, /缓冲耗尽/);
  assert.match(note.text, /不循环/);
});

test('criterion 4 negative control: the loop would hide the dry buffer', () => {
  // With the loop on, the same exhausted run never reports a hold -- it rewinds and
  // pretends there is something to play. That is exactly the dishonesty 0007 ⑤
  // forbids, and it is why the honest branch is the default.
  const single = [fakeSegment({ frames: 41, base: 0, version: 1 })];
  const tl = stream.timelineOf(single, JOINT_COUNT);
  let playhead = 0;
  let held = 0;
  for (let step = 0; step < 60; step += 1) {
    const decision = stream.advancePlayhead({
      segments: single, timeline: tl, playhead, deltaSeconds: 1, fps: STREAM_FPS,
      runaheadFrames: 4, loopSingleSegment: true,
    });
    playhead = decision.playhead;
    if (decision.held) held += 1;
  }
  assert.equal(held, 0, 'the loop never reports a hold, so the dry-buffer warning never appears');
  const note = stream.bufferNote({
    timeline: tl, playhead, fps: STREAM_FPS, held: false, loopSingleSegment: true,
  });
  assert.equal(note.state, 'loop');
  assert.match(note.text, /这不是连续流/);
});

test('runahead is the only thing that asks for a segment, and a miss is never a rewind', () => {
  const tl = stream.timelineOf(FIVE, JOINT_COUNT);
  // The playhead is the *only* input that decides. With runway left, nothing is asked
  // for -- and no remembered "last decision" is consulted, because there is none.
  const buffered = stream.shouldPrefetch({
    fetching: false, nextIntent: null, committedIntent: { version: 1 },
    timeline: tl, playhead: 0, runaheadFrames: 4,
  });
  assert.equal(buffered.fetch, false);
  assert.equal(buffered.reason, stream.STREAM_HOLD.BUFFERED);
  assert.equal(buffered.wantMore, false);
  // ... and inside the runahead window it is.
  const low = stream.shouldPrefetch({
    fetching: false, nextIntent: null, committedIntent: { version: 1 },
    timeline: tl, playhead: tl.maxFrame - 2, runaheadFrames: 4,
  });
  assert.equal(low.fetch, true);
  assert.equal(low.reason, null);
  // One request at a time -- but only until its deadline.
  const inflight = stream.shouldPrefetch({
    fetching: true, fetchAgeMs: 10, timeoutMs: 15000, nextIntent: null,
    committedIntent: { version: 1 }, timeline: tl, playhead: tl.maxFrame - 2,
    runaheadFrames: 4,
  });
  assert.equal(inflight.reason, stream.STREAM_HOLD.GENERATING);
  assert.equal(inflight.expired, false);
  // At the deadline the in-flight request stops being a reason to wait: `expired` is
  // the shell's instruction to abandon it by name and let the next frame try again.
  const stale = stream.shouldPrefetch({
    fetching: true, fetchAgeMs: 15000, timeoutMs: 15000, nextIntent: null,
    committedIntent: { version: 1 }, timeline: tl, playhead: tl.maxFrame - 2,
    runaheadFrames: 4,
  });
  assert.equal(stale.expired, true);
  assert.equal(stale.fetch, false);
  assert.equal(stale.reason, stream.STREAM_HOLD.TIMEOUT);
  assert.equal(stream.STREAM_HOLD.TIMEOUT, 'prefetch_timeout');
  // No conditions at all is the one refusal that is not about buffering.
  assert.equal(stream.shouldPrefetch({ fetching: false, timeline: tl }).reason,
    stream.STREAM_HOLD.NO_INTENT);
  // The explicit loop is not a stream and must not grow the queue behind the reader.
  assert.equal(stream.shouldPrefetch({
    fetching: false, nextIntent: null, committedIntent: { version: 1 },
    timeline: tl, playhead: tl.maxFrame, loopSingleSegment: true,
  }).reason, stream.STREAM_HOLD.LOOPING);
});

// ---------------------------------------------------------------------------
// the prefetch controller: a stream that must never stop feeding itself
// ---------------------------------------------------------------------------
//
// The reader's symptom was "角色彻底不动": `segments_generated`/`frames_generated`
// frozen (10 segments / 800 frames, identical 4 s later), the process alive and still
// answering `/api/health`, and **no further `POST /api/plan`**. The player was out of
// material (`held: true`) and -- this is the bug -- it had no way left to ask for
// more. Two independent wedges produced that, and neither is visible in a browser
// without a human watching it:
//
//   1. `nextSegmentIntent` answered `intent: null` for *unchanged* conditions, and
//      `pullSegment` refuses a null intent -- so one sentence bought exactly one
//      segment, and after that the runahead path refused, quietly, every frame.
//   2. `shouldPrefetch` read a remembered `lastDecision` and a `fetching` flag, so a
//      request that never settled suppressed every future request forever.
//
// The harness below plays the stream with the **shipped pure functions** against a
// **stub `/api/plan`** in plain `node`: no DOM, no fetch, no browser, no three.js.
// Only the *network* is stubbed -- every decision is the page's own code. It advances
// >= 20 segments and asserts the three things the reader could not see: the queue
// keeps growing, the playhead never goes back, and no hold outlives a bounded window.
//
// Each assertion has a negative control that makes the **same verdict** go red --
// prefetch switched off, the old `intent: null` restored, the in-flight flag wedged
// with no deadline, and a mid-stream failure with the flag left set. A criterion that
// cannot fail is not a criterion.

/** A segment long enough to be a real clip: 40 frames = the 2.0 s minimum at 20 fps. */
const SOAK_SEGMENT_FRAMES = 40;
const SOAK_MS_PER_FRAME = 1000 / STREAM_FPS;

/**
 * `nextSegmentIntent`, with the shipped bug available as an explicit control.
 *
 * `reuseIntent: false` reproduces the exact line that froze the page -- `intent: null`
 * for unchanged conditions -- so the negative control is a real alternative
 * implementation and not a mock of one (the same technique the loop/wrap controls
 * above use).
 */
function soakIntent({ queue, intent, committed, reuseIntent }) {
  if (!reuseIntent && intent && committed && intent.version === committed.version) {
    return { intent: null, commit: null, changed: false, reason: null };
  }
  return stream.nextSegmentIntent({
    segments: queue, nextIntent: intent, committedIntent: committed, fetching: false,
  });
}

/**
 * Play a stream against a stub `/api/plan`, in the order the render loop uses: one
 * `advancePlayhead` per frame, then one prefetch decision per frame, then the request
 * settles after `latencyFrames` (or never, for the wedge controls).
 *
 * Options:
 *   * `wedgedAttempts` -- the first N requests never settle at all. Only the deadline
 *     can clear the flag, which is the failure the reader hit.
 *   * `stuckFlag` -- the deadline fires but the release does *not* clear the in-flight
 *     flag. This is the "永久卡住" case: a deadline that cannot release anything is
 *     no better than no deadline, and it must make the soak go red.
 *   * `stallAfterFailure` -- a *failed* request leaves the in-flight flag set forever.
 *     With a finite `timeoutMs` the deadline recovers it; with `timeoutMs: 1e9` it
 *     models the shipped page, which had no deadline at all.
 *   * `prefetchEnabled: false` -- the page with the prefetch loop switched off.
 *   * `reuseIntent: false` -- the shipped `intent: null`.
 *   * `plan(attempt)` -- return `null` for a clip, or a named error string.
 */
function soakRun(options = {}) {
  const given = options || {};
  const maxFrames = given.maxFrames === undefined ? 2000 : given.maxFrames;
  const latencyFrames = given.latencyFrames === undefined ? 3 : given.latencyFrames;
  const timeoutMs = given.timeoutMs === undefined ? stream.PREFETCH_TIMEOUT_MS : given.timeoutMs;
  const runaheadFrames = given.runaheadFrames === undefined
    ? stream.RUNAHEAD_FRAMES : given.runaheadFrames;
  const segmentFrames = given.segmentFrames === undefined ? SOAK_SEGMENT_FRAMES : given.segmentFrames;
  const prefetchEnabled = given.prefetchEnabled !== false;
  const reuseIntent = given.reuseIntent !== false;
  const stallAfterFailure = given.stallAfterFailure === true;
  const stuckFlag = given.stuckFlag === true;
  const plan = given.plan || (() => null);
  let wedgesLeft = given.wedgedAttempts === undefined ? 0 : given.wedgedAttempts;

  const queue = [];
  const trace = [];
  const events = [];
  const intent = { version: 1, body: { prompt: 'walk' } };
  let committed = null;
  let playhead = 0;
  let held = false;
  let heldStreak = 0;
  let worstHeldStreak = 0;
  let monotone = true;
  let fetching = false;
  let fetchStartedAt = 0;
  let flight = null;            // {settleAt, error}; settleAt Infinity = never settles
  let retries = 0;              // since the last appended segment
  let totalRetries = 0;
  let lastPrefetchError = null;
  let attempt = 0;

  const fail = (step, name) => {
    retries += 1;
    totalRetries += 1;
    lastPrefetchError = { name, at: step * SOAK_MS_PER_FRAME, retries };
    events.push({ step, kind: 'error', name });
  };
  const succeed = (step) => {
    queue.push(fakeSegment({
      frames: segmentFrames,
      base: queue.length * segmentFrames,
      version: committed ? committed.version : 1,
    }));
    retries = 0;
    lastPrefetchError = null;
    events.push({ step, kind: 'append', segments: queue.length });
  };

  for (let step = 0; step < maxFrames; step += 1) {
    const now = step * SOAK_MS_PER_FRAME;
    // 1. the answer to the request in flight, if it is due.
    if (flight && flight.settleAt <= step) {
      const settled = flight;
      flight = null;
      fetching = false;
      if (settled.error) {
        fail(step, settled.error);
        if (stallAfterFailure) {
          // The control: the failure is recorded and the flag is left set forever.
          fetching = true;
          fetchStartedAt = now;
          flight = { settleAt: Infinity, error: null };
        }
      } else {
        succeed(step);
      }
    }
    // 2. one frame of playback.
    const tl = queue.length ? stream.timelineOf(queue, JOINT_COUNT) : null;
    const previous = playhead;
    const decision = stream.advancePlayhead({
      segments: queue, timeline: tl, playhead, deltaSeconds: 1 / STREAM_FPS,
      fps: STREAM_FPS, runaheadFrames, loopSingleSegment: false,
    });
    playhead = decision.playhead;
    held = decision.held;
    if (playhead < previous) monotone = false;
    heldStreak = held ? heldStreak + 1 : 0;
    worstHeldStreak = Math.max(worstHeldStreak, heldStreak);
    // 3. the prefetch controller, exactly the block the render loop runs.
    if (prefetchEnabled) {
      const controller = stream.shouldPrefetch({
        fetching,
        fetchAgeMs: fetching ? now - fetchStartedAt : 0,
        timeoutMs,
        nextIntent: intent,
        committedIntent: committed,
        timeline: tl,
        playhead,
        runaheadFrames,
      });
      if (controller.expired) {
        // The deadline: release by name, and the next frame is free to ask again.
        // `stuckFlag` models a release that does not actually clear the flag.
        if (!stuckFlag) fetching = false;
        flight = null;
        attempt += 1;
        fail(step, stream.STREAM_HOLD.TIMEOUT);
        events.push({ step, kind: 'timeout' });
      } else if (controller.fetch) {
        const resolution = soakIntent({ queue, intent, committed, reuseIntent });
        if (resolution && resolution.intent) {
          if (resolution.commit) committed = resolution.commit;
          attempt += 1;
          fetching = true;
          fetchStartedAt = now;
          const wedged = wedgesLeft > 0;
          if (wedged) wedgesLeft -= 1;
          flight = wedged
            ? { settleAt: Infinity, error: null }
            : { settleAt: step + latencyFrames, error: plan(attempt) };
        }
      }
    }
    trace.push({ step, playhead, held, segments: queue.length, fetching, retries });
  }

  return {
    segments: queue.length,
    playhead,
    steps: trace.length,
    monotone,
    worstHeldStreak,
    heldFrames: trace.filter((entry) => entry.held).length,
    retries,
    totalRetries,
    lastPrefetchError: lastPrefetchError ? lastPrefetchError.name : null,
    attempts: attempt,
    timeouts: events.filter((event) => event.kind === 'timeout').length,
    errors: events.filter((event) => event.kind === 'error').length,
    appends: events.filter((event) => event.kind === 'append').length,
    events,
    trace,
  };
}

/**
 * The soak criteria as a predicate, so every negative control can assert that *this*
 * verdict goes red instead of re-implementing the check it is meant to falsify.
 */
function soakVerdict(result, options = {}) {
  const given = options || {};
  const targetSegments = given.targetSegments === undefined ? 24 : given.targetSegments;
  const maxHeldStreak = given.maxHeldStreak === undefined
    ? stream.RUNAHEAD_FRAMES : given.maxHeldStreak;
  const reasons = [];
  if (result.segments < targetSegments) {
    reasons.push(`the queue only reached ${result.segments} segment(s), need ${targetSegments}`);
  }
  if (!result.monotone) reasons.push('the playhead went backwards');
  if (result.worstHeldStreak > maxHeldStreak) {
    reasons.push(`a hold lasted ${result.worstHeldStreak} frames, over ${maxHeldStreak}`);
  }
  return { ok: reasons.length === 0, reasons };
}

test('soak: 24 segments of continuous playback, and the queue never stops growing', () => {
  const result = soakRun({ maxFrames: 2000 });
  const verdict = soakVerdict(result, { targetSegments: 24 });
  assert.ok(verdict.ok, `the soak must pass: ${verdict.reasons.join('; ')}`);
  assert.ok(result.segments >= 24, `${result.segments} segments`);
  assert.equal(result.monotone, true, 'the playhead is monotone across every append');
  assert.equal(result.worstHeldStreak, 0,
    'with runahead 40 and a 3-frame request, playback must never run dry at all');
  assert.equal(result.errors, 0);
  assert.equal(result.appends, result.segments, 'every append is accounted for');
  // The seam really is a continuation: each segment adds frames-1 global frames.
  const tl = stream.timelineOf(
    Array.from({ length: result.segments }, (_, i) => fakeSegment({
      frames: SOAK_SEGMENT_FRAMES, base: i * SOAK_SEGMENT_FRAMES,
    })),
    JOINT_COUNT,
  );
  assert.equal(tl.frameCount, result.segments * SOAK_SEGMENT_FRAMES - (result.segments - 1),
    'the dropped seam frame is one per seam, and only one');

  // The shipped loop is the thing under test: the decision is recomputed from the
  // playhead every frame, and no remembered decision may take part in it.
  const loop = functionBody(SOURCE, 'frame');
  assert.match(loop, /shouldPrefetch\(/, 'the render loop asks the controller');
  assert.match(loop, /plan\.expired/, 'the deadline is read in the loop it protects');
  assert.match(loop, /releaseInFlight\(\)/);
  // The code may not *use* a remembered decision (the comments name the one that was
  // removed, which is why this matches the property usage and not the bare word).
  assert.doesNotMatch(loop, /lastDecision\s*[:,]/,
    'no remembered decision may take part in the per-frame prefetch decision');
  assert.doesNotMatch(loop, /decision\.wantMore/,
    'the runahead rule is recomputed from the playhead, not read off the last frame');
});

test('soak: a mid-stream failure is named, retried, and costs a bounded hold', () => {
  // Every seventh request fails. The failure is at the moment the buffer is thinnest,
  // so this is the case where a page that does not retry goes dry -- and the retry has
  // to happen without a human pressing anything.
  const result = soakRun({
    maxFrames: 2000,
    latencyFrames: 20,
    plan: (attempt) => (attempt % 7 === 0 ? stream.STREAM_HOLD.FAILED : null),
  });
  const verdict = soakVerdict(result, { targetSegments: 24, maxHeldStreak: stream.RUNAHEAD_FRAMES });
  assert.ok(verdict.ok, `a retried failure must still stream: ${verdict.reasons.join('; ')}`);
  assert.ok(result.errors >= 1, `the fixture must really fail (got ${result.errors})`);
  assert.equal(result.lastPrefetchError, null, 'a recovered stream must not still say it failed');
  assert.ok(result.totalRetries >= 1, `${result.totalRetries} retries`);
  assert.equal(result.monotone, true);
});

test('soak: the deadline is what makes a wedged request recoverable', () => {
  // The first request never settles at all -- the flag can only be cleared by the
  // deadline -- and a later failure leaves the flag set the same way. With the
  // deadline (20 frames here, 15 s by default) the stream recovers and keeps growing.
  const result = soakRun({
    maxFrames: 2600,
    timeoutMs: 20 * SOAK_MS_PER_FRAME,
    wedgedAttempts: 1,
    stallAfterFailure: true,
    plan: (attempt) => (attempt % 5 === 0 ? stream.STREAM_HOLD.FAILED : null),
  });
  const verdict = soakVerdict(result, { targetSegments: 24, maxHeldStreak: stream.RUNAHEAD_FRAMES });
  assert.ok(verdict.ok, `the deadline must recover the stream: ${verdict.reasons.join('; ')}`);
  assert.ok(result.timeouts >= 1, `the deadline must actually fire (got ${result.timeouts})`);
  assert.equal(stream.PREFETCH_TIMEOUT_MS, 15000,
    'the shipped default is the documented 15 s deadline');
});

test('negative control: with prefetch switched off the stream is at a standstill', () => {
  // The "no prefetch" page. It must fail the growth criterion -- or the soak is
  // measuring something other than the prefetch controller.
  const result = soakRun({ prefetchEnabled: false, maxFrames: 600 });
  assert.equal(result.segments, 0, 'nothing can be appended if nothing is asked for');
  const verdict = soakVerdict(result, { targetSegments: 24 });
  assert.equal(verdict.ok, false, 'the growth criterion must be able to fail');
  assert.match(verdict.reasons.join(' '), /only reached 0 segment/);
});

test('negative control: the shipped "unchanged intent" refusal buys exactly one segment', () => {
  // Wedge 1, verbatim: `intent: null` for unchanged conditions. One segment, then a
  // silent refusal every frame while `/api/health` keeps being polled -- the reader's
  // "10 segments then frozen", reproduced in four lines.
  const result = soakRun({ reuseIntent: false, maxFrames: 2000 });
  assert.equal(result.segments, 1, 'the old answer buys exactly one segment');
  const verdict = soakVerdict(result, { targetSegments: 24 });
  assert.equal(verdict.ok, false, 'and the same verdict must notice');
  assert.ok(result.trace.filter((entry) => entry.segments === 1).length > 100,
    'the stream sits at one segment for the rest of the run: no growth, no error, no stop');

  // The shipped function reuses the committed conditions instead, and commits nothing.
  const committed = { version: 3 };
  const reused = stream.nextSegmentIntent({
    segments: [fakeSegment({})], nextIntent: { version: 3 }, committedIntent: committed,
    fetching: false,
  });
  assert.equal(reused.intent, committed, 'the conditions that are playing are returned');
  assert.equal(reused.commit, null, 'nothing new is committed: no mid-clip condition change');
  assert.equal(reused.changed, false);
  // ... and the same when nothing is pending at all (the state a `hold` leaves).
  assert.equal(stream.nextSegmentIntent({
    segments: [fakeSegment({})], nextIntent: null, committedIntent: committed, fetching: false,
  }).intent, committed);
  assert.match(functionBody(SOURCE, 'nextSegmentIntent'),
    /return \{ intent: committed, commit: null, changed: false, reason: null \};/,
    'the reuse branch is the shipped line, not a test-only copy');
});

test('negative control: an in-flight flag that can never be cleared stops the stream forever', () => {
  // Wedge 2: a request that never settles, with no deadline to abandon it. `fetching`
  // stays true, so `shouldPrefetch` answers `stream_segment_in_flight` on every frame
  // for the rest of the run -- the exact "process alive, health polled, no
  // `POST /api/plan`" state. `timeoutMs: 1e9` models the page as it shipped.
  const wedged = soakRun({ wedgedAttempts: Infinity, timeoutMs: 1e9, maxFrames: 600 });
  assert.equal(wedged.segments, 0);
  assert.equal(wedged.attempts, 1, 'exactly one request ever leaves');
  assert.equal(soakVerdict(wedged, { targetSegments: 24 }).ok, false);

  const stalled = soakRun({
    stallAfterFailure: true, timeoutMs: 1e9, maxFrames: 2000, latencyFrames: 20,
    plan: (attempt) => (attempt % 3 === 0 ? stream.STREAM_HOLD.FAILED : null),
  });
  const verdict = soakVerdict(stalled, { targetSegments: 24 });
  assert.equal(verdict.ok, false, 'a stuck flag must be visible to the soak');
  assert.ok(stalled.segments < 24, `the queue must stall (reached ${stalled.segments})`);
  assert.ok(stalled.worstHeldStreak > stream.RUNAHEAD_FRAMES,
    `and it must sit held, not merely slow (${stalled.worstHeldStreak} frames)`);
});

test('negative control: a deadline that cannot clear the flag is still a permanent hold', () => {
  // "永久卡住" in its purest form: the deadline fires on every frame and the release
  // never actually clears the flag, so no request can ever leave. A deadline that
  // cannot release anything is no better than no deadline -- and the soak must say so.
  const stuck = soakRun({
    stuckFlag: true, stallAfterFailure: true, maxFrames: 1500,
    timeoutMs: 20 * SOAK_MS_PER_FRAME, latencyFrames: 3,
    plan: (attempt) => (attempt % 4 === 0 ? stream.STREAM_HOLD.FAILED : null),
  });
  const verdict = soakVerdict(stuck, { targetSegments: 24 });
  assert.equal(verdict.ok, false, 'a stuck flag must be visible to the soak');
  assert.ok(stuck.segments < 24, `the queue must stall (reached ${stuck.segments})`);
  assert.ok(stuck.segments >= 1, 'the stall must happen mid-stream, not before anything played');
  assert.ok(stuck.timeouts > 10, `the deadline keeps firing (${stuck.timeouts} times)`);
  assert.ok(stuck.worstHeldStreak > stream.RUNAHEAD_FRAMES,
    `and the player sits held (${stuck.worstHeldStreak} frames)`);

  // With a release that really clears the flag, the same wedge recovers: that pair is
  // the whole value of requirement 2.
  const recovered = soakRun({
    wedgedAttempts: 1, maxFrames: 2600, timeoutMs: 20 * SOAK_MS_PER_FRAME,
  });
  assert.equal(soakVerdict(recovered, { targetSegments: 24 }).ok, true);
});

test('the stream row tells 取 / 取失败 / 真的没料 apart, with the count and the name', () => {
  const tl = stream.timelineOf(
    Array.from({ length: 4 }, (_, i) => fakeSegment({ frames: 40, base: i * 40 })), JOINT_COUNT,
  );
  // Playhead positions are chosen so that exactly one reading is the true one: `thin`
  // is inside the runahead window with runway left, `dry` is the last buffered frame.
  const base = {
    timeline: tl, fps: STREAM_FPS, runaheadFrames: stream.RUNAHEAD_FRAMES,
    nextIntent: { version: 2 },
  };
  const thin = { ...base, playhead: tl.maxFrame - 10 };
  const dryBase = { ...base, playhead: tl.maxFrame };

  const fetching = stream.prefetchNote({ ...thin, fetching: true, retries: 0 });
  assert.equal(fetching.state, stream.PREFETCH_STATE.FETCHING);
  assert.match(fetching.text, /正在取下一段/);

  const failed = stream.prefetchNote({
    ...thin, retries: 3, lastPrefetchError: { name: 'prefetch_timeout' },
  });
  assert.equal(failed.state, stream.PREFETCH_STATE.FAILED);
  assert.match(failed.text, /取下一段失败，正在重试/);
  assert.match(failed.text, /prefetch_timeout/, 'the named reason must be on the row');
  assert.match(failed.text, /重试 3 次/, 'and so must the count');

  const dry = stream.prefetchNote({
    ...dryBase, held: true, retries: 2, lastPrefetchError: { name: 'prefetch_failed' },
  });
  assert.equal(dry.state, stream.PREFETCH_STATE.DRY);
  assert.match(dry.text, /缓冲耗尽（持帧）/, 'the honest "no material" reading is kept');
  assert.match(dry.text, /prefetch_failed/, 'with the last failure named beside it');
  assert.match(dry.text, /重试 2 次/);

  const blocked = stream.prefetchNote({ timeline: tl, playhead: tl.maxFrame, fps: STREAM_FPS });
  assert.equal(blocked.state, stream.PREFETCH_STATE.BLOCKED);
  assert.match(blocked.text, /没有可预取的条件/);

  const loop = stream.prefetchNote({ ...dryBase, loopSingleSegment: true });
  assert.equal(loop.state, stream.PREFETCH_STATE.LOOP);
  assert.match(loop.text, /不是连续流/);

  const idle = stream.prefetchNote({
    ...base, playhead: 0, retries: 1, lastPrefetchError: { name: 'prefetch_failed' },
  });
  assert.equal(idle.state, stream.PREFETCH_STATE.IDLE,
    'plenty of runway is not a failure reading even when the last attempt failed');
  assert.match(idle.text, /帧 \/ .* s/);
  assert.match(idle.text, /最后失败 prefetch_failed/, 'the last failure is still named');

  // The small print carries the numbers, so a reading is auditable after the fact.
  assert.match(failed.help, /重试 3 次/);
  assert.match(failed.help, /最后一次失败：prefetch_timeout/);
  assert.match(failed.help, /在飞 否/);
  assert.match(stream.prefetchNote({ ...thin, fetching: true }).help, /在飞 是/);
});

test('the hold path re-arms the stream instead of clearing it, and says which', () => {
  // A `hold` is "that sentence did not ask for motion", not "stop the stream". The
  // shipped page nulled both intents here, which is the second wedge: no intent means
  // no request can ever leave again.
  const rearmed = stream.holdResolution({
    lastGoodIntent: { version: 2, text: 'walk' }, reason: 'non_motion',
  });
  assert.equal(rearmed.rearmed, true);
  assert.equal(rearmed.committedIntent.version, 2, 'the playing conditions are re-armed');
  assert.equal(rearmed.nextIntent, null, 'the sentence itself is dropped');
  assert.equal(rearmed.code, 'prefetch_intent_rearmed');

  const first = stream.holdResolution({ lastGoodIntent: null, reason: 'non_motion' });
  assert.equal(first.rearmed, false);
  assert.equal(first.committedIntent, null);
  assert.equal(first.code, stream.STREAM_HOLD.NO_INTENT,
    'a hold on the very first request has no stream to continue, and says so by name');

  // ... and the branch that used to clear both intents is the one that now re-arms.
  const body = functionBody(SOURCE, 'pullSegment');
  assert.match(body, /holdResolution\(/);
  assert.doesNotMatch(body, /state\.committedIntent = null/);
  assert.doesNotMatch(body, /state\.nextIntent = null/);
  assert.match(body, /prefetch hold/);
});

test('the manual retry releases the wedge and asks again without resetting the run', () => {
  const body = functionBody(SOURCE, 'retryPrefetchNow');
  assert.match(body, /releaseInFlight\(\)/);
  assert.match(body, /pullSegment\(\{ quiet: false \}\)/);
  assert.doesNotMatch(body, /resetStream/, 'retry must never clear what is on screen');
  assert.match(SOURCE, /els\.retryPrefetch\.addEventListener\('click', \(\) => retryPrefetchNow\(\)\)/);
  assert.match(INDEX_SOURCE, /id="retry-prefetch"/);
  assert.match(INDEX_SOURCE, /id="stream-retry-help"/);
  assert.match(SOURCE, /els\.streamRetryHelp\.textContent = prefetch\.help/);

  // `releaseInFlight` is the one place the flag is cleared by hand, and it invalidates
  // the outstanding request so a late answer cannot append after its replacement.
  const release = functionBody(SOURCE, 'releaseInFlight');
  assert.match(release, /state\.requestId \+= 1/);
  assert.match(release, /state\.fetching = false/);
  assert.match(release, /state\.fetchAbort\.abort\(\)/);
});

test('every way a prefetch can end is named, and none of them is a silent stop', () => {
  const body = functionBody(SOURCE, 'pullSegment');
  // The deadline is measured from when the request left, not from when the loop
  // noticed it.
  assert.match(body, /state\.fetchStartedAt = performance\.now\(\)/);
  assert.match(body, /signal: controller\.signal/);
  // A failure is named, recorded and reported -- never swallowed.
  assert.match(body, /notePrefetchFailure\(name, error\.message\)/);
  assert.match(body, /error\.name === 'AbortError' \? STREAM_HOLD\.TIMEOUT : STREAM_HOLD\.FAILED/);
  // The flag may only be cleared by the request that owns it: an abandoned request
  // settling late must not wipe its replacement's flag.
  assert.match(body, /if \(requestId === state\.requestId\) \{/);

  const note = functionBody(SOURCE, 'notePrefetchFailure');
  assert.match(note, /state\.retries \+= 1/);
  assert.match(note, /state\.lastPrefetchError = \{/);
  assert.match(note, /noteClientEvent\('console\.warn'/,
    'the named failure must reach /api/client-log, or the server cannot see it');
  assert.doesNotMatch(note, /state\.fetching = false/,
    'the failure path does not clear the flag itself: releaseInFlight does, once');

  // The render loop treats an expired request as an instruction to release and retry.
  const loop = functionBody(SOURCE, 'frame');
  assert.match(loop, /notePrefetchFailure\(\s*STREAM_HOLD\.TIMEOUT/);
  assert.match(loop, /setStatus\(streamHoldText\(STREAM_HOLD\.TIMEOUT\), true\)/);
});

test('the snapshot carries the prefetch readings, so a stall is answerable from the log', () => {
  // Loaded here rather than shared with the client-log section below: this test is
  // about the two fields the *prefetch* controller added, and it must not depend on
  // where in this file the other region happens to be defined.
  const log = loadClientLog();
  const snapshot = log.clientSnapshot({
    playhead: 800, timelineFrames: 800, segments: 10, buffered: 0, held: true,
    prefetchRetries: 4, prefetchError: 'prefetch_timeout', href: '/viewer',
  });
  assert.equal(snapshot.segments, 10);
  assert.equal(snapshot.held, true);
  assert.equal(snapshot.buffered, 0);
  assert.equal(snapshot.prefetch_retries, 4);
  assert.equal(snapshot.prefetch_error, 'prefetch_timeout');
  // ... and they are declared on both sides of the wire, not merely emitted.
  for (const field of ['prefetch_retries', 'prefetch_error']) {
    assert.ok(log.CLIENT_LOG_SNAPSHOT_FIELDS.includes(field), `${field} declared`);
  }
  // The server's allowed kinds are what a named prefetch failure is reported under.
  assert.ok(log.CLIENT_LOG_KINDS.includes('console.warn'));
  assert.match(SOURCE, /prefetchError: state\.lastPrefetchError \? state\.lastPrefetchError\.name : null/);
});

test('a new intent waits for the next segment boundary (0007 3)', () => {
  // While a segment is being fetched, a newer intent must not be committed: it would
  // land inside the clip being generated.
  const pending = { version: 2, prompt: 'walk faster', text: 'walk faster' };
  const committed = { version: 1, prompt: 'walk', text: 'walk' };
  const during = stream.nextSegmentIntent({
    segments: [fakeSegment({})], nextIntent: pending, committedIntent: committed, fetching: true,
  });
  assert.equal(during.intent, null);
  assert.equal(during.reason, stream.STREAM_HOLD.GENERATING);
  // Once the fetch is done, the newer intent is what the *next* segment uses.
  const after = stream.nextSegmentIntent({
    segments: [fakeSegment({})], nextIntent: pending, committedIntent: committed, fetching: false,
  });
  assert.equal(after.intent.version, 2);
  assert.equal(after.changed, true);
  // And the page says where it will land rather than applying it silently.
  const note = stream.playbackIntentNote({ nextIntent: pending, committedIntent: committed });
  assert.equal(note.state, 'pending');
  assert.equal(note.applies, 'next');
  assert.match(note.text, /下一段/);
  assert.match(note.text, /不在片段中途换条件/);
});

test('the seam frame is the same frame on both sides of an append', () => {
  // `/api/plan` carries the previous clip's last frame over verbatim as frame 0 of
  // the new clip (0009's "0.000 mm"). The concatenation drops that duplicate, so the
  // global frame at the seam is one frame and not two -- and the two clips agree on
  // the pose there, which is what makes the append invisible.
  const a = fakeSegment({ frames: 6, base: 0, version: 1 });
  const b = fakeSegment({ frames: 6, base: 6, version: 1 });
  // The server's carry: b's frame 0 is a's last frame.
  b.joints.set(a.joints.subarray((a.frames - 1) * JOINT_COUNT * 3), 0);
  b.roots.set(a.roots.subarray((a.frames - 1) * 3), 0);
  const tl = stream.timelineOf([a, b], JOINT_COUNT);
  assert.equal(tl.frameCount, 11, 'two 6-frame clips share one seam frame');
  assert.equal(tl.droppedSeamFrames, 1);
  const seam = 5;
  for (let j = 0; j < JOINT_COUNT * 3; j += 1) {
    assert.equal(tl.poses[seam * JOINT_COUNT * 3 + j], a.joints[(a.frames - 1) * JOINT_COUNT * 3 + j],
      `the seam pose must be bit-identical at joint component ${j}`);
  }
  // The base of segment 1 is exactly the seam frame, and frame 0 of the run is
  // segment 0's frame 0: no frame of the run is a restart.
  assert.equal(stream.segmentBase([a, b], 1), 5,
    'segment 1 starts at the seam frame: it contributes frames-1 global frames');
  assert.equal(stream.segmentAtFrame([a, b], 4).index, 0);
  assert.equal(stream.segmentAtFrame([a, b], 5).index, 0, 'the seam frame belongs to the segment that owns it');
  assert.equal(stream.segmentAtFrame([a, b], 6).index, 1);
});

test('negative control: clipping the playhead to the newest segment would restart it', () => {
  // The old page kept `playhead` inside the newest clip and reset it to 0 on every
  // install. Modelled here, that is a position that falls back to 0 at the append --
  // the "character was reset" the reader saw.
  const a = fakeSegment({ frames: 6, base: 0, version: 1 });
  const b = fakeSegment({ frames: 6, base: 6, version: 1 });
  const globalBefore = 5;                      // the seam frame, in global terms
  const localAfterReset = 0;                   // what the old model showed
  assert.ok(localAfterReset < globalBefore, 'the old model goes backwards at the seam');
  const tl = stream.timelineOf([a, b], JOINT_COUNT);
  assert.ok(tl.frameCount > 6, 'the queue is longer than the newest clip');
  assert.equal(stream.segmentBase([a, b], 1), 5,
    'the new segment starts where the old one ended, so its frame 0 is not a new global frame');
});

test('criterion 5: appending does not change the count of any frame already shown', () => {
  // The client half of the "0.000 mm" criterion. The server's half is asserted in
  // `tests/test_viewer_api.py` (`carried_frame_joint_mm == 0.0`); this is the half
  // that used to fail, because the page replaced the clip instead of appending to it.
  // Every segment the server returns is `frames` long, so appending clips must add
  // `frames - 1` global frames each -- a different count would mean the client dropped
  // or duplicated a frame at the seam, and the character would jump by that much.
  const clips = FIVE;
  let queue = [];
  let previousCount = 0;
  let previousPoses = null;
  for (const clip of clips) {
    const globalBefore = previousCount;
    queue = queue.concat([clip]);
    const tl = stream.timelineOf(queue, JOINT_COUNT);
    const added = tl.frameCount - globalBefore;
    const expected = globalBefore === 0 ? clip.frames : clip.frames - 1;
    assert.equal(added, expected,
      `appending a ${clip.frames}-frame clip must add ${expected} global frames, added ${added}`);
    if (previousPoses !== null) {
      // Nothing already drawn may change: frame k of the run has the same value it had
      // before the append. This is what "the trajectory only grows" means on the wire.
      for (let k = 0; k < previousCount * JOINT_COUNT * 3; k += 1) {
        if (previousPoses[k] !== tl.poses[k]) {
          assert.fail(`append changed global frame content at component ${k}`);
        }
      }
      // And the seam frame itself is literally the same numbers on both sides.
      const seamComponent = (previousCount - 1) * JOINT_COUNT * 3;
      const a = clips[queue.length - 2];
      assert.equal(
        tl.poses[seamComponent],
        a.joints[(a.frames - 1) * JOINT_COUNT * 3],
        'the seam frame must be the previous clip\'s own last frame',
      );
    }
    previousCount = tl.frameCount;
    previousPoses = tl.poses;
  }
  assert.equal(previousCount, 5 * 6 - 4, 'five 6-frame clips share four seam frames');
});

test('the timeline region is pure: it runs with no document, no fetch, no state', () => {
  const trap = new Proxy({}, {
    get(_target, key) { throw new Error(`the pure timeline touched ${String(key)}`); },
  });
  const isolated = loadTimeline(vm.createContext({
    document: trap, window: trap, fetch: trap, state: trap, els: trap,
    THREE: trap, performance: trap, requestAnimationFrame: trap, localStorage: trap,
  }));
  const tl = isolated.timelineOf([fakeSegment({ frames: 4, base: 0 })], JOINT_COUNT);
  assert.equal(tl.frameCount, 4);
  assert.equal(isolated.bufferDepth(tl, 0), 3);
  assert.equal(isolated.RUNAHEAD_FRAMES, 40);
  assert.match(isolated.BUFFER_EMPTY_HINT, /不循环/);
});

test('negative control: the loop is off by default in the markup and in the code', () => {
  // Loop is an explicit option, so it must not ship checked -- and the page must read
  // its checkbox rather than hardcode a loop anywhere.
  assert.match(INDEX_SOURCE, /<input id="loop-toggle" type="checkbox">/);
  assert.doesNotMatch(INDEX_SOURCE, /id="loop-toggle"[^>]*checked/);
  assert.match(INDEX_SOURCE, /this repeats one clip/);
  assert.match(SOURCE, /loopSingleSegment: els\.loopToggle\.checked/);
  // The advance itself must not contain an unconditional wrap.
  const body = functionBody(SOURCE, 'updatePlayhead');
  assert.doesNotMatch(body, /playhead\s*-=/, 'the page must not rewind the playhead by arithmetic');
  assert.doesNotMatch(body, /%\s*.*frameCount/, 'and must not modulo it into a loop');
});

// ---------------------------------------------------------------------------
// the client log: what the page says about itself, and what it may not say
// ---------------------------------------------------------------------------
//
// The page has no terminal. `POST /api/client-log` is how a renderer exception
// becomes visible at all -- which is why the two things that could go wrong with it
// are both tested here:
//
//   * it must be **bounded and named**, or a page that is failing becomes a page that
//     is also flooding, and the server's log fills with events nobody can grep;
//   * it must be a **whitelist**, or the diagnostic channel becomes the leak. The page
//     holds no key, but an error message is text the page did not write -- a failed
//     `fetch('https://…?key=…')` rejects with the whole URL in it.

const CLIENT_LOG_START = '// --- pure-client-log: start';
const CLIENT_LOG_END = '// --- pure-client-log: end';
const CLIENT_LOG_SENTINEL = 'ts_test_key_do_not_use_0123456789';

/** The marked client-log region of the shipped page, verbatim. */
function pureClientLog(source) {
  return markedRegion(source, CLIENT_LOG_START, CLIENT_LOG_END);
}

function loadClientLog(context = vm.createContext({})) {
  return vm.runInContext(
    `${pureClientLog(SOURCE)}\n({ CLIENT_LOG_MAX_EVENTS, CLIENT_LOG_TEXT_CHARS, `
    + `CLIENT_LOG_KINDS, CLIENT_LOG_SNAPSHOT_FIELDS, clientLogText, clientLogRedact, `
    + `clientLogEntry, clientLogPush, clientLogPath, clientSnapshot });`,
    context,
    { filename: 'viewer/app.js#pure-client-log' },
  );
}

const clientLogRegion = loadClientLog();

test('the client log is bounded: the queue caps at fifty and counts what fell out', () => {
  let queue = [];
  let dropped = 0;
  const cap = clientLogRegion.CLIENT_LOG_MAX_EVENTS;
  for (let i = 0; i < cap + 7; i += 1) {
    const pushed = clientLogRegion.clientLogPush(
      queue, clientLogRegion.clientLogEntry({ kind: 'error', text: `event ${i}` }),
    );
    queue = pushed.queue;
    dropped += pushed.dropped;
  }
  assert.equal(queue.length, cap, 'the queue is bounded at the documented maximum');
  assert.equal(dropped, 7, 'and it says how many it lost');
  // Oldest first out: the newest events are the ones near the symptom.
  assert.equal(queue[queue.length - 1].text, `event ${cap + 6}`);
  assert.equal(queue[0].text, 'event 7');

  // The same bound holds against the *server's* limit, so a full batch is never
  // refused for being too big.
  assert.equal(clientLogRegion.CLIENT_LOG_MAX_EVENTS, ardy_viewer_max_events());
  assert.ok(clientLogRegion.CLIENT_LOG_TEXT_CHARS < 600,
    'the page truncates below the server’s limit, so it is never refused for length');
});

/** The server's own ceiling, read out of the shipped Python. */
function ardy_viewer_max_events() {
  const source = readFileSync(join(HERE, '..', 'scripts', 'ardy_viewer.py'), 'utf8');
  const match = source.match(/^CLIENT_LOG_MAX_EVENTS = (\d+)$/m);
  assert.ok(match, 'scripts/ardy_viewer.py must define CLIENT_LOG_MAX_EVENTS');
  return Number(match[1]);
}

test('a log line cannot be forged: control characters are collapsed, not passed through', () => {
  const forged = clientLogRegion.clientLogText('first line\n[ardy_viewer] client-log error fake');
  assert.doesNotMatch(forged, /\n/, 'a newline would let a message write a second line');
  assert.match(forged, /first line/);
  assert.match(forged, /fake/);

  // Truncation is marked, because an unmarked half-message reads as the whole one.
  const long = clientLogRegion.clientLogText('x'.repeat(clientLogRegion.CLIENT_LOG_TEXT_CHARS + 25));
  assert.match(long, /…\[\+25 chars\]$/);
  assert.equal(long.length > clientLogRegion.CLIENT_LOG_TEXT_CHARS, true);
  // Exactly at the cap is untouched: no marker on a message that fits.
  const exact = clientLogRegion.clientLogText('y'.repeat(clientLogRegion.CLIENT_LOG_TEXT_CHARS));
  assert.equal(exact.length, clientLogRegion.CLIENT_LOG_TEXT_CHARS);
  assert.doesNotMatch(exact, /chars\]/);
});

test('an event the page cannot name is refused, not relabelled', () => {
  // The server refuses an unknown kind by name. The page must therefore not invent
  // one: relabelling a `console.log` as an `error` would put a name on a report that
  // does not fit it -- worse than sending nothing.
  const named = clientLogRegion.clientLogEntry({ kind: 'error', text: 'ok', at: 12.6, line: 3 });
  assert.equal(named.kind, 'error');
  assert.equal(named.at, 13, 'the timestamp is whole milliseconds');
  assert.equal(named.line, 3);
  assert.equal(clientLogRegion.clientLogEntry({ kind: 'console.log', text: 'nope' }), null);
  assert.equal(clientLogRegion.clientLogEntry({ kind: undefined, text: 'nope' }), null);

  // A null entry is counted as lost rather than vanishing.
  const pushed = clientLogRegion.clientLogPush([], null);
  assert.deepEqual(plain(pushed.queue), []);
  assert.equal(pushed.dropped, 1);

  for (const kind of clientLogRegion.CLIENT_LOG_KINDS) {
    assert.ok(clientLogRegion.clientLogEntry({ kind, text: 'x' }), `${kind} must be sendable`);
  }
});

test('the snapshot is a whitelist: nothing the page was not written to send can ride along', () => {
  const poisoned = {
    playhead: 411.6,
    timelineFrames: 680,
    segments: 6,
    buffered: 12,
    held: false,
    playing: true,
    fps: 19.8,
    intentVersion: 3,
    route: 'compose',
    routeProbability: 0.81,
    camera: { frames: 400, clamped: 0, maxStep: 0.17, lastStep: 0.05, userJumps: 1 },
    screen: [0.12, -0.3, 5.1],
    root: [0, 0.9, 20.4],
    href: 'http://127.0.0.1:8850/viewer?prompt=secret#frag',
    // ... the things that must never be sent:
    secret: CLIENT_LOG_SENTINEL,
    apiKey: CLIENT_LOG_SENTINEL,
    TYPESAFE_API_KEY: CLIENT_LOG_SENTINEL,
    environment: { HOME: '/Users/someone' },
    localStorage: { token: CLIENT_LOG_SENTINEL },
  };
  const snapshot = clientLogRegion.clientSnapshot(poisoned);
  const text = JSON.stringify(snapshot);
  assert.doesNotMatch(text, new RegExp(CLIENT_LOG_SENTINEL));
  assert.doesNotMatch(text, /Users\/someone/);
  for (const field of ['secret', 'apiKey', 'TYPESAFE_API_KEY', 'environment', 'localStorage']) {
    assert.equal(field in snapshot, false, `${field} must not be in the snapshot`);
  }
  // Named fields only, and every one of them is one the region declared.
  for (const key of Object.keys(snapshot)) {
    assert.ok(clientLogRegion.CLIENT_LOG_SNAPSHOT_FIELDS.includes(key), `undeclared field ${key}`);
  }
  assert.deepEqual(
    Object.keys(snapshot).sort(), [...clientLogRegion.CLIENT_LOG_SNAPSHOT_FIELDS].sort(),
    'the snapshot is exactly the declared whitelist, so a new field is a deliberate act',
  );
  // The values that *are* sent are shaped, not passed through: a triple is three
  // numbers a test can assert on.
  assert.deepEqual(plain(snapshot.root), [0, 0.9, 20.4]);
  assert.equal(snapshot.playhead, 412);
  assert.equal(snapshot.camera.max_step_m, 0.17);
  assert.equal(snapshot.route, 'compose');
});

test('the snapshot carries the URL path only, because a query string can carry a secret', () => {
  assert.equal(clientLogRegion.clientLogPath('http://127.0.0.1:8850/?key=abc#x'), '/');
  assert.equal(clientLogRegion.clientLogPath('http://127.0.0.1:8850/viewer?a=1'), '/viewer');
  assert.equal(clientLogRegion.clientLogPath('/plain'), '/plain');
  assert.equal(clientLogRegion.clientLogPath(''), null);
  assert.equal(clientLogRegion.clientLogPath(`/?token=${CLIENT_LOG_SENTINEL}`), '/');
  // The path is the only part that survives -- so a key in the query cannot be sent
  // even if something put one there.
  assert.doesNotMatch(String(clientLogRegion.clientLogPath(`/?token=${CLIENT_LOG_SENTINEL}`)),
    new RegExp(CLIENT_LOG_SENTINEL));
});

test('negative control: a naive "send everything" snapshot really does carry the key', () => {
  // The check above only means something if it can fail. A snapshot built by
  // spreading the page's state -- the obvious implementation -- carries every
  // poisoned field, so the whitelist assertion is doing real work.
  const poisoned = { playhead: 1, secret: CLIENT_LOG_SENTINEL, route: 'compose' };
  const naive = { ...poisoned };
  assert.match(JSON.stringify(naive), new RegExp(CLIENT_LOG_SENTINEL),
    'the naive snapshot must leak, or the whitelist test proves nothing');
  assert.doesNotMatch(JSON.stringify(clientLogRegion.clientSnapshot(poisoned)),
    new RegExp(CLIENT_LOG_SENTINEL));
});

test('negative control: a key in an error message is masked before it leaves the page', () => {
  // The realistic path: an error message is text the page did not write. A failed
  // request rejects with its whole URL, query string and all.
  const message = `TypeError: Failed to fetch https://api.example/v1?key=${CLIENT_LOG_SENTINEL}`;
  assert.match(message, new RegExp(CLIENT_LOG_SENTINEL), 'the fixture must carry the key');
  const masked = clientLogRegion.clientLogText(message);
  assert.doesNotMatch(masked, new RegExp(CLIENT_LOG_SENTINEL));
  assert.match(masked, /\[redacted:typesafe-key\]/);
  assert.match(masked, /Failed to fetch/, 'and it is masked, not dropped');

  // The other shapes, each masked under its own name.
  const shapes = {
    'openai-key': 'sk-live_ABCDEFGHIJKLMNOPQRSTUVWXYZ012345',
    'github-token': 'ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789',
    'aws-key': 'AKIAIOSFODNN7EXAMPLE',
    'bearer-token': 'Bearer abcdefghijklmnopqrstuvwxyz',
    'slack-token': 'xoxb-1234567890-abcdefghijkl',
    'jev-key': 'jev_ABCDEFGHIJKLMNOP',
  };
  for (const [label, token] of Object.entries(shapes)) {
    const out = clientLogRegion.clientLogText(`boom ${token} end`);
    assert.ok(out.includes(`[redacted:${label}]`), `${label} must be masked (got ${out})`);
    assert.doesNotMatch(out, new RegExp(token.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')));
  }

  // A JWT-shaped token, which is the one that is *three* dot-separated segments.
  const jwt = 'eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r_wW1g';
  assert.match(clientLogRegion.clientLogText(`auth ${jwt}`), /\[redacted:jwt\]/);
});

test('the client log region is pure: it runs with no document, no fetch, no state', () => {
  const trap = new Proxy({}, {
    get(_target, key) { throw new Error(`the pure client log touched ${String(key)}`); },
  });
  const isolated = loadClientLog(vm.createContext({
    document: trap, window: trap, fetch: trap, state: trap, els: trap, console: trap,
    location: trap, performance: trap, setTimeout: trap, localStorage: trap,
  }));
  const entry = isolated.clientLogEntry({ kind: 'error', text: 'still pure' });
  assert.equal(entry.text, 'still pure');
  assert.equal(isolated.clientSnapshot({ playhead: 1 }).playhead, 1);
});

test('the page actually installs the hooks, and a key cannot reach them', () => {
  // The decision functions are useless if nothing calls them. This is the wiring
  // half: all three event families, a flush on unload, and the heartbeat that makes
  // the snapshot exist *before* the reader says "不太对".
  const install = functionBody(SOURCE, 'installClientLog');
  assert.ok(install.includes("'console.error', 'console.warn'"), 'console.error/warn are captured');
  assert.ok(install.includes("addEventListener('error'"), 'window.onerror is captured');
  assert.ok(install.includes("addEventListener('unhandledrejection'"), 'rejections are captured');
  assert.ok(install.includes("addEventListener('pagehide'"), 'the last chance to send is taken');
  assert.ok(install.includes('original(...args)'), 'and the console calls still reach the console');

  const boot = functionBody(SOURCE, 'boot');
  assert.ok(boot.includes('installClientLog();'), 'the hooks are installed at boot');
  assert.ok(boot.indexOf('installClientLog();') < boot.indexOf('loadHealth()'),
    'and before the first request, so a boot failure is reportable');
  assert.ok(boot.includes('startClientLogHeartbeat();'));

  const heartbeat = functionBody(SOURCE, 'startClientLogHeartbeat');
  assert.match(heartbeat, /snapshotOnly: true/, 'the heartbeat sends a snapshot, not events');

  // The page holds no key: nothing in what it renders mentions one, and the log
  // region names the *shapes* it masks rather than any value.
  const region = pureClientLog(SOURCE);
  assert.doesNotMatch(region, /TYPESAFE_API_KEY|JEV_API_KEY/, 'the page must not name the key vars');
  assert.match(region, /CLIENT_LOG_SECRET_SHAPES/, 'it masks by shape instead');
});

test('the snapshot whitelist is the same list on both sides of the wire', () => {
  // Two hand-written whitelists drift. The failure mode of drift here is quiet and
  // one-directional: the page builds a field the server does not know, and the server
  // drops it -- so the very field added to diagnose a symptom is the one that never
  // arrives. `screen` and `root` were both in that position when this test was added.
  const source = readFileSync(join(HERE, '..', 'scripts', 'ardy_viewer.py'), 'utf8');
  const match = source.match(/^CLIENT_LOG_SNAPSHOT_FIELDS = \(([\s\S]*?)\)$/m);
  assert.ok(match, 'scripts/ardy_viewer.py must declare CLIENT_LOG_SNAPSHOT_FIELDS as a tuple');
  const server = [...match[1].matchAll(/"([^"]+)"/g)].map((found) => found[1]);
  assert.ok(server.length >= 10, `the server list looks parsed (${server.length} names)`);
  assert.deepEqual(
    [...clientLogRegion.CLIENT_LOG_SNAPSHOT_FIELDS].sort(),
    [...server].sort(),
    'the page and the server must whitelist the same snapshot fields',
  );

  // ... and the two fields this round added are actually emitted, not merely declared.
  const snapshot = clientLogRegion.clientSnapshot({
    playhead: 1, camera: { clamped: 2 }, screen: [0.1, 0.2, 5], root: [0, 0.9, 21],
    href: '/viewer?x=1',
  });
  assert.deepEqual(plain(snapshot.screen), [0.1, 0.2, 5]);
  assert.deepEqual(plain(snapshot.root), [0, 0.9, 21]);
  assert.equal(snapshot.camera.clamped, 2);
  assert.equal(snapshot.url, '/viewer', 'the path reaches the snapshot, without the query');
});
