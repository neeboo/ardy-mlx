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
    bodyFromPlanBody: /planBody\(resolved, \{ seed, duration, noise \}\)/.test(source),
    // ... and `generateClip` has exactly one caller left: the guarded entry point.
    singleGenerateCallSite: generateDecls === 1 && generateCalls === 2
      && source.includes('return generateClip(resolved);'),
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
      return pending >= 0 && pending < body.indexOf('generateClip(resolved)');
    })(),
    // ... and redrawn from the *server's* verdict, not from the page's guess at one.
    bubbleFromVerdict: /showBubble\(bubbleFromPlan\(payload\)\)/.test(source),
    // A hold keeps the clip on screen: no clip is installed and the branch returns.
    holdInstallsNothing: (() => {
      const body = functionBody(source, 'generateClip');
      const hold = body.indexOf("payload.kind === 'hold'");
      const install = body.indexOf("installClip(payload, 'mine')");
      return hold >= 0 && install > hold && /return;/.test(body.slice(hold, install));
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
    .replace('planBody(resolved, { seed, duration, noise })',
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
