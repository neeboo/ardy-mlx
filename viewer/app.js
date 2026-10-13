// viewer/app.js -- ardy-mlx browser viewer.
//
// Written for this repository. The page drives three endpoints on
// `scripts/ardy_viewer.py`:
//
//   GET  /api/health                     -> device, parameters, cold start, cached prompts, skeleton
//   POST /api/plan  {prompt, seed, ...}  -> one generated clip (posed_joints, contacts, timing)
//   GET  /api/reference?duration=&prompt= -> the golden clip for that exact duration, or a named error
//
// Two rules the page enforces rather than hides:
//
//   * a prompt is free text. If it has a cached vector the server uses it; if it
//     does not, the server encodes it here with the local MLX text encoder and
//     caches the result. A failure -- no encoder, or an encoder that refuses -- is
//     shown verbatim by name (`local_encoder_unavailable` / `local_encoder_failed`)
//     and never becomes a different prompt's clip;
//   * a vector's *source* is always on screen. A reference bf16 vector is the
//     conditioning signal the golden clip came from; a locally-encoded one is a
//     different, equally valid condition. The HUD names which one this clip used,
//     so a local clip can never be mistaken for a golden comparison;
//   * `duration` must be a whole number of ARDY generation horizons. The input
//     steps by one horizon and the server rejects anything else, so the page
//     cannot quietly round a request into a different clip.
//
// Frames arrive base64-encoded float32 (`posed_joints`, `root_positions`), so what
// is drawn is the server's float32 bit-for-bit, with no JSON round-trip through
// a decimal representation.

import * as THREE from './vendor/three.module.min.js';
import { Rig, clipBounds } from './rig.js';

const viewport = document.querySelector('#viewport');
const html = document.documentElement;

const els = {
  promptSelect: document.querySelector('#prompt-select'),
  promptHelp: document.querySelector('#prompt-help'),
  promptText: document.querySelector('#prompt-text'),
  promptTextGo: document.querySelector('#prompt-text-go'),
  promptTextHelp: document.querySelector('#prompt-text-help'),
  seedInput: document.querySelector('#seed-input'),
  seedNext: document.querySelector('#seed-next'),
  durationInput: document.querySelector('#duration-input'),
  durationStep: document.querySelector('#duration-step'),
  durationHelp: document.querySelector('#duration-help'),
  noiseSelect: document.querySelector('#noise-select'),
  noiseHelp: document.querySelector('#noise-help'),
  generate: document.querySelector('#generate'),
  reload: document.querySelector('#reload'),
  playPause: document.querySelector('#play-pause'),
  frameLabel: document.querySelector('#frame-label'),
  frameSlider: document.querySelector('#frame-slider'),
  resetCamera: document.querySelector('#reset-camera'),
  fitCamera: document.querySelector('#fit-camera'),
  goldenToggle: document.querySelector('#golden-toggle'),
  goldenHelp: document.querySelector('#golden-help'),
  contactsToggle: document.querySelector('#contacts-toggle'),
  trailToggle: document.querySelector('#trail-toggle'),
  status: document.querySelector('#status'),
  errorBox: document.querySelector('#error-box'),
  hud: {
    clip: document.querySelector('#hud-clip'),
    model: document.querySelector('#hud-model'),
    frames: document.querySelector('#hud-frames'),
    generate: document.querySelector('#hud-generate'),
    fps: document.querySelector('#hud-fps'),
    noise: document.querySelector('#hud-noise'),
    embedding: document.querySelector('#hud-embedding'),
    source: document.querySelector('#hud-source'),
    encode: document.querySelector('#hud-encode'),
  },
  stats: {
    model: document.querySelector('#stat-model'),
    device: document.querySelector('#stat-device'),
    params: document.querySelector('#stat-params'),
    coldStart: document.querySelector('#stat-coldstart'),
    generate: document.querySelector('#stat-generate'),
    frames: document.querySelector('#stat-frames'),
    windows: document.querySelector('#stat-windows'),
    fps: document.querySelector('#stat-fps'),
    clipFps: document.querySelector('#stat-clipfps'),
    noise: document.querySelector('#stat-noise'),
    embedding: document.querySelector('#stat-embedding'),
    prompt: document.querySelector('#stat-prompt'),
    source: document.querySelector('#stat-source'),
    quantization: document.querySelector('#stat-quantization'),
    encode: document.querySelector('#stat-encode'),
    total: document.querySelector('#stat-total'),
    encoder: document.querySelector('#stat-encoder'),
    endpoint: document.querySelector('#stat-endpoint'),
  },
};

// ---------------------------------------------------------------------------
// scene
// ---------------------------------------------------------------------------
const scene = new THREE.Scene();
scene.fog = new THREE.FogExp2(0x080a0f, 0.035);
const camera = new THREE.PerspectiveCamera(42, 1, 0.02, 200);
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.outputColorSpace = THREE.SRGBColorSpace;
viewport.append(renderer.domElement);

scene.add(new THREE.HemisphereLight(0xcde4ff, 0x25322a, 2.4));
const keyLight = new THREE.DirectionalLight(0xffffff, 3.0);
keyLight.position.set(3.5, 7, 4);
scene.add(keyLight);
const rimLight = new THREE.DirectionalLight(0x6be6ba, 1.1);
rimLight.position.set(-4, 3, -3.5);
scene.add(rimLight);

const floor = new THREE.Mesh(
  new THREE.PlaneGeometry(60, 60),
  new THREE.MeshStandardMaterial({ color: 0x0d121a, roughness: 0.96, metalness: 0.02 }),
);
floor.rotation.x = -Math.PI / 2;
floor.position.y = -0.014;
scene.add(floor);
const grid = new THREE.GridHelper(24, 48, 0x3a4b5d, 0x1d2733);
scene.add(grid);
// 1 m reference squares: the grid alone does not say how far "forward" is.
const metreGrid = new THREE.GridHelper(24, 24, 0x24414f, 0x24414f);
metreGrid.position.y = 0.001;
scene.add(metreGrid);

// ---------------------------------------------------------------------------
// camera (drag to orbit, wheel to zoom, double-click to reset)
// ---------------------------------------------------------------------------
const cameraView = {
  yaw: 0.7, pitch: 0.22, distance: 4.6, dragging: false, x: 0, y: 0,
  focus: new THREE.Vector3(0, 0.75, 0),
};
const desiredCamera = new THREE.Vector3();

function resetCameraView() {
  cameraView.yaw = 0.7;
  cameraView.pitch = 0.22;
  cameraView.distance = 4.6;
  cameraView.focus.set(0, 0.75, 0);
}

function fitCameraToClip() {
  const clip = state.clip;
  if (!clip || !clip.jointCount || !clip.frameCount) return;
  const box = clipBounds(clip.joints, clip.frameCount, clip.jointCount);
  if (box.isEmpty()) return;
  box.getCenter(cameraView.focus);
  cameraView.focus.y = Math.max(0.45, cameraView.focus.y);
  const size = box.getSize(new THREE.Vector3());
  const reach = Math.max(size.x, size.y, size.z, 0.6);
  cameraView.distance = THREE.MathUtils.clamp(reach * 2.6, 2.2, 24);
}

function resize() {
  const width = Math.max(1, viewport.clientWidth);
  const height = Math.max(1, viewport.clientHeight);
  renderer.setSize(width, height, false);
  camera.aspect = width / height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(viewport);

renderer.domElement.addEventListener('pointerdown', (event) => {
  cameraView.dragging = true;
  cameraView.x = event.clientX;
  cameraView.y = event.clientY;
  renderer.domElement.setPointerCapture(event.pointerId);
});
renderer.domElement.addEventListener('pointermove', (event) => {
  if (!cameraView.dragging) return;
  cameraView.yaw -= (event.clientX - cameraView.x) * 0.006;
  cameraView.pitch = THREE.MathUtils.clamp(
    cameraView.pitch + (event.clientY - cameraView.y) * 0.004, -0.08, 1.15);
  cameraView.x = event.clientX;
  cameraView.y = event.clientY;
});
const endDrag = (event) => {
  cameraView.dragging = false;
  if (event.pointerId !== undefined && renderer.domElement.hasPointerCapture(event.pointerId)) {
    renderer.domElement.releasePointerCapture(event.pointerId);
  }
};
renderer.domElement.addEventListener('pointerup', endDrag);
renderer.domElement.addEventListener('pointercancel', endDrag);
renderer.domElement.addEventListener('wheel', (event) => {
  cameraView.distance = THREE.MathUtils.clamp(
    cameraView.distance * Math.exp(event.deltaY * 0.001), 1.2, 30);
  event.preventDefault();
}, { passive: false });
renderer.domElement.addEventListener('dblclick', resetCameraView);

// ---------------------------------------------------------------------------
// state
// ---------------------------------------------------------------------------
const state = {
  health: null,
  skeleton: null,
  clip: null,        // {joints, roots, contacts, frameCount, jointCount, meta}
  golden: null,      // same shape, or null
  goldenKey: null,   // `${duration}|${prompt}` the loaded golden answers
  rig: null,
  goldenRig: null,
  playhead: 0,
  playing: true,
  lastTime: performance.now(),
  fpsWindow: [],
  fps: 0,
  busy: false,
  requestId: 0,
};

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------
function setStatus(text, isError = false) {
  els.status.textContent = text;
  els.status.dataset.error = isError ? 'true' : 'false';
  html.dataset.viewerStatus = isError ? 'error' : 'ready';
}

function showError(name, message, detail) {
  els.errorBox.hidden = false;
  els.errorBox.innerHTML = '';
  const head = document.createElement('div');
  head.innerHTML = `<code>${name}</code>`;
  const body = document.createElement('div');
  body.textContent = message;
  els.errorBox.append(head, body);
  if (detail) {
    const extra = document.createElement('div');
    extra.textContent = detail;
    els.errorBox.append(extra);
  }
}

function clearError() {
  els.errorBox.hidden = true;
  els.errorBox.textContent = '';
}

function fmtMs(ms) {
  if (ms === null || ms === undefined) return '—';
  if (ms < 1000) return `${ms.toFixed(0)} ms`;
  return `${(ms / 1000).toFixed(2)} s`;
}

function fmtCount(value) {
  if (value === null || value === undefined) return '—';
  return value.toLocaleString('en-US');
}

function shortHash(hash, keep = 12) {
  if (!hash) return '—';
  return hash.length <= keep * 2 ? hash : `${hash.slice(0, keep)}…${hash.slice(-6)}`;
}

/** Decode a base64 string of little-endian float32 into a Float32Array. */
function decodeFloat32(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
}

/** Decode a base64 string of 0/1 bytes into a Uint8Array. */
function decodeUint8(b64) {
  const binary = atob(b64);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

async function getJson(url, options) {
  const response = await fetch(url, options);
  let payload = null;
  try {
    payload = await response.json();
  } catch (error) {
    throw Object.assign(new Error(`${url} did not return JSON (HTTP ${response.status})`), {
      errorName: 'not_json',
    });
  }
  if (!response.ok) {
    const err = new Error(payload && payload.message ? payload.message : `HTTP ${response.status}`);
    err.errorName = (payload && payload.error) || `http_${response.status}`;
    err.payload = payload;
    throw err;
  }
  return payload;
}

// ---------------------------------------------------------------------------
// server round trips
// ---------------------------------------------------------------------------
async function loadHealth() {
  const health = await getJson('/api/health');
  state.health = health;
  state.skeleton = health.skeleton;
  populatePrompts(health);
  populateDuration(health);
  els.stats.device.textContent = `${health.device.name} (${health.device.backend})`;
  els.stats.model.textContent = health.model;
  els.stats.params.textContent = fmtCount(health.parameters);
  els.stats.coldStart.textContent = fmtMs(health.cold_start_s * 1000);
  els.hud.model.textContent = health.model;
  updateEncoderHelp(health);
  html.dataset.viewerStatus = 'ready';
  return health;
}

/** Say exactly what the local encoder is, in the words `/api/health` used.
 *
 * The point of this line is that "you can type any prompt" is a claim about a
 * specific artifact at a specific precision. If the encoder is missing the panel
 * says so, and says which prompts still work (the cached ones).
 */
function updateEncoderHelp(health) {
  const encoder = health.encoder || { enabled: false, state: 'disabled' };
  const details =
    `${encoder.encoder_id || 'no local encoder'} · ${encoder.quantization || '—'}`;
  els.stats.encoder.textContent =
    encoder.available ? `${details}` : `${encoder.state}: ${encoder.reason || '—'}`;
  els.stats.encoder.title = encoder.repo || '';
  if (encoder.available) {
    els.promptText.disabled = false;
    els.promptTextGo.disabled = false;
    els.promptTextHelp.textContent =
      `Any prompt works. A prompt with no cached vector is encoded here by the local MLX `
      + `text encoder (${encoder.quantization}), about half a second once, then cached. `
      + `A locally-encoded vector is close to — not identical to — the reference bf16 one, `
      + `so its clip is a valid sample that is NOT aligned with the golden clip.`;
    els.promptTextHelp.classList.remove('warn');
  } else {
    els.promptText.disabled = true;
    els.promptTextGo.disabled = true;
    els.promptTextHelp.textContent =
      `No local text encoder is usable (${encoder.reason || encoder.state}). Only the `
      + `cached prompts below can be run; a new prompt fails by name as `
      + `local_encoder_unavailable rather than being substituted.`;
    els.promptTextHelp.classList.add('warn');
  }
}

function populatePrompts(health) {
  els.promptSelect.innerHTML = '';
  if (!health.embeddings.length) {
    const option = document.createElement('option');
    option.value = '';
    option.textContent = 'no cached embedding';
    els.promptSelect.append(option);
    els.promptSelect.disabled = true;
    els.generate.disabled = true;
    els.promptHelp.textContent = 'No cached embedding is on disk. See embeddings/README.md.';
    return;
  }
  for (const entry of health.embeddings) {
    const option = document.createElement('option');
    option.value = entry.prompt_hash;
    option.textContent = `${entry.text}  ·  ${sourceLabel(entry)}`;
    option.title = entry.quantization
      ? `${entry.encoder_id} · ${entry.quantization}`
      : entry.encoder_id;
    els.promptSelect.append(option);
  }
  els.promptSelect.disabled = false;
  els.generate.disabled = false;
  updatePromptHelp();
}

function selectedEmbedding() {
  const hash = els.promptSelect.value;
  return (state.health?.embeddings ?? []).find((entry) => entry.prompt_hash === hash) ?? null;
}

/** A short, always-visible name for where a cached vector came from. */
function sourceLabel(entry) {
  if (entry.source === 'reference') return 'reference bf16';
  if (entry.source === 'local') return `local MLX ${entry.bits ? entry.bits + '-bit' : ''}`.trim();
  return `unlabelled (${entry.source})`;
}

function updatePromptHelp() {
  const entry = selectedEmbedding();
  if (!entry) {
    els.promptHelp.textContent = '—';
    return;
  }
  els.promptHelp.textContent =
    `${sourceLabel(entry)} · ${entry.encoder_id} · prompt ${entry.prompt_hash} · `
    + `embedding ${entry.embedding_sha256.slice(0, 16)}… · ${entry.tokens} token(s)`;
}

// --- pure-prompt-guard: start (no DOM, no fetch, no state) -------------------
//
// Everything between these two markers is a pure function of plain data, and is
// evaluated and tested on its own by `tests/viewer_js_guard.test.mjs` under node.
// The page has no browser test harness, so the *decision* to send a request is
// kept here -- where a test can run it -- and the call sites only adapt DOM values
// into these arguments.
//
// Why this exists at all: `/api/plan` reads `params.get("prompt")`, and
// `JSON.stringify` **drops undefined values**. A request built from a prompt that
// never resolved is therefore not "a request with an empty prompt", it is a body
// with no `prompt` key at all: `{"seed":0,"duration":2,"noise":"golden"}`. The
// server answers that correctly and by name -- 400 `no_prompt_given` -- but it is
// the page's mistake, and it must not be reachable from any control.

/** What the page says when there is nothing to plan yet. Never an error box. */
const NO_PROMPT_HINT = 'Type a prompt, or pick a cached embedding from the list.';

/**
 * Which prompt a `/api/plan` body would carry -- or `null` when there is none.
 *
 * The free-text box wins when it has content after trimming (whitespace is not a
 * prompt); otherwise the selected cached embedding's hash. Pure.
 */
function resolveRequest(input) {
  const given = input || {};
  const typed = typeof given.typed === 'string' ? given.typed.trim() : '';
  if (typed) return { source: 'text', prompt: typed, text: typed, entry: null };
  const entry = given.entry;
  const hash = entry && typeof entry.prompt_hash === 'string' ? entry.prompt_hash.trim() : '';
  if (!hash) return null;
  return {
    source: 'cached',
    prompt: hash,
    text: typeof entry.text === 'string' ? entry.text : hash,
    entry,
  };
}

/**
 * The one guard: a resolved request is planned only when it names a prompt.
 *
 * A caller that skips this and builds the body anyway cannot produce a *nearly*
 * right request -- it produces one that is refused with `no_prompt_given`. That is
 * what makes this check load-bearing rather than decorative, and
 * `tests/viewer_js_guard.test.mjs` asserts both halves of that sentence.
 */
function shouldPlan(resolved) {
  return !!resolved && typeof resolved.prompt === 'string' && resolved.prompt.trim() !== '';
}

/** The exact body `/api/plan` is given. Pure, so a test can assert on it. */
function planBody(resolved, options) {
  const given = options || {};
  return {
    prompt: resolved ? resolved.prompt : undefined,
    seed: given.seed,
    duration: given.duration,
    noise: given.noise,
  };
}

/**
 * Whether `/api/reference` should be asked for this duration at all.
 *
 * `/api/health` already reports `golden_references`, so a duration without a
 * golden run is a fact the page was told, not something to discover with a 404 in
 * the console. `available: false` means the run is configured but its `out/*.npz`
 * is not on disk -- also a 404, also known in advance. Pure.
 */
function referenceDecision(input) {
  const given = input || {};
  const runs = Array.isArray(given.goldenReferences) ? given.goldenReferences : [];
  // `Number(null)` is 0, so a malformed entry is filtered by shape rather than
  // coerced: matching duration 0 to a null record would load the wrong clip's guard.
  const durationOf = (run) => (
    run && run.duration_s !== null && run.duration_s !== undefined
      && Number.isFinite(Number(run.duration_s))
      ? Number(run.duration_s)
      : null
  );
  const seconds = Number(given.duration);
  if (!Number.isFinite(seconds)) {
    return { fetch: false, reason: 'no_duration', message: 'Choose a duration first.' };
  }
  const entry = runs.find((run) => durationOf(run) === seconds) || null;
  if (!entry) {
    const known = runs
      .map(durationOf)
      .filter((value) => value !== null)
      .sort((a, b) => a - b)
      .map((value) => `${value.toFixed(1)} s`)
      .join(', ');
    return {
      fetch: false,
      reason: 'no_golden_reference_for_duration',
      message: `No golden clip exists for ${seconds.toFixed(1)} s`
        + (known ? ` (the reference runs on disk are: ${known})` : '')
        + '.',
    };
  }
  if (entry.available === false) {
    return {
      fetch: false,
      reason: 'golden_reference_missing',
      message: `A golden run for ${seconds.toFixed(1)} s is configured, but its out/*.npz `
        + 'is not on disk.',
    };
  }
  return { fetch: true, reason: null, message: null };
}
// --- pure-prompt-guard: end --------------------------------------------------

/**
 * The DOM adapter around `resolveRequest`: the free-text box when it has content,
 * else the selected cached embedding.
 */
function requestedPrompt() {
  return resolveRequest({ typed: els.promptText.value, entry: selectedEmbedding() });
}

function populateDuration(health) {
  const step = health.duration_step_s;
  els.durationInput.step = String(step);
  els.durationInput.min = String(step);
  if (Number(els.durationInput.value) % step !== 0) {
    els.durationInput.value = String(step);
  }
  els.durationHelp.textContent =
    `Must be a whole number of ${health.horizon}-frame horizons `
    + `(${step.toFixed(2)} s at ${health.fps} fps). Anything else is rejected by the server `
    + `as duration_not_horizon_multiple.`;
  const golden = health.golden_references.map((r) => `${r.duration_s.toFixed(1)}s`).join(', ');
  els.noiseHelp.textContent =
    `golden injects the reference x_T recorded with artifacts/golden-run* (available for: `
    + `${golden || 'none'}). random is MLX's own RNG, which is not CUDA's Philox stream, so a `
    + `random clip is a valid sample but not a reference reproduction.`;
  updateGoldenHelp();
}

function updateGoldenHelp() {
  if (!state.health) return;
  const duration = Number(els.durationInput.value);
  const decision = referenceDecision({
    goldenReferences: state.health.golden_references,
    duration,
  });
  if (!decision.fetch) {
    // The same decision `loadReference` makes, so the sentence on screen and the
    // request that is (not) sent cannot disagree.
    els.goldenHelp.textContent =
      `${decision.message} The toggle reports that by name instead of showing a different `
      + `clip, and the page does not ask for it.`;
    els.goldenHelp.classList.add('warn');
    return;
  }
  const entry = state.health.golden_references.find((r) => r.duration_s === duration);
  els.goldenHelp.textContent =
    `A golden clip exists for ${duration.toFixed(1)} s (${entry.npz_path}); `
    + `it is the reference implementation's own output.`;
  els.goldenHelp.classList.remove('warn');
}

/**
 * The single entry point for every trigger that would (re)plan a clip.
 *
 * `trigger` names the control in the status line, so "which control asked" is
 * visible on the page rather than guessed from a console trace. When there is
 * nothing to plan the page says one plain sentence and builds no request: this is
 * a page-side "not yet", not a server failure, so it never appears as a red box.
 */
async function planIfReady(trigger) {
  const resolved = requestedPrompt();
  if (!shouldPlan(resolved)) {
    clearError();
    setStatus(`Nothing to generate yet (${trigger}). ${NO_PROMPT_HINT}`);
    return null;
  }
  return generateClip(resolved);
}

async function generateClip(resolved = requestedPrompt()) {
  if (state.busy) return null;
  if (!shouldPlan(resolved)) {
    // Defence in depth. Every control reaches this function through `planIfReady`,
    // but the body builder is the last place that can still notice a prompt that
    // never resolved -- and a body whose `prompt` is undefined is not a body with
    // an empty prompt, it is one with no `prompt` key, which the server refuses as
    // 400 `no_prompt_given`.
    clearError();
    setStatus(NO_PROMPT_HINT);
    return null;
  }
  const duration = Number(els.durationInput.value);
  const seed = Number(els.seedInput.value);
  const noise = els.noiseSelect.value;
  state.busy = true;
  els.generate.disabled = true;
  clearError();
  // The encoding step is invisible to the user otherwise: a cache miss costs about
  // half a second before sampling even starts, and the status line says so rather
  // than making the whole request look slow.
  setStatus(
    `Generating ${duration.toFixed(1)} s with ${noise} noise`
    + (resolved.source === 'text'
      ? ` — encoding ${JSON.stringify(resolved.text)} locally first…` : '…'));
  const requestId = ++state.requestId;
  const startedAt = performance.now();
  try {
    const payload = await getJson('/api/plan', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(planBody(resolved, { seed, duration, noise })),
    });
    if (requestId !== state.requestId) return;
    installClip(payload, 'mine');
    const timing = payload.timing || {};
    const encoded = timing.encode_ms;
    setStatus(
      `Generated ${payload.frames} frames in ${fmtMs(timing.total_ms ?? payload.generate_ms)}`
      + (encoded ? ` (encoded the prompt in ${fmtMs(encoded)} first)` : ' (prompt was cached)')
      + `.`);
    // A prompt that was just encoded is now in the cache, so refresh the list and
    // select it -- otherwise the dropdown would still show the old set and the next
    // Generate would look like it had ignored the text box.
    await loadHealth().catch(() => {});
    syncPromptSelect(payload.embedding && payload.embedding.prompt_hash);
  } catch (error) {
    if (requestId !== state.requestId) return;
    const name = error.errorName || 'request_failed';
    showError(name, error.message, error.payload && error.payload.detail);
    setStatus(`Failed: ${name}`, true);
    els.hud.generate.textContent = fmtMs(performance.now() - startedAt);
  } finally {
    state.busy = false;
    els.generate.disabled = false;
  }
}

/** Point the dropdown at a prompt hash when it is present; leave it alone otherwise. */
function syncPromptSelect(promptHash) {
  if (!promptHash) return;
  const option = [...els.promptSelect.options].find((item) => item.value === promptHash);
  if (option) {
    els.promptSelect.value = promptHash;
    updatePromptHelp();
  }
}

/** Drop the loaded golden clip and its rig. Leaves the checkbox alone. */
function dropGolden() {
  state.golden = null;
  state.goldenKey = null;
  if (state.goldenRig) {
    state.goldenRig.dispose();
    state.goldenRig = null;
  }
}

/** Hide the golden overlay and say the clip on screen is this machine's own. */
function hideGoldenOverlay() {
  if (state.goldenRig) state.goldenRig.setVisible(false);
  els.hud.clip.textContent = state.clip ? `mine (${state.clip.duration.toFixed(1)} s)` : '—';
}

/**
 * Every control that changes which duration/prompt the reference would be for.
 *
 * Picking from the list, changing the duration and stepping it all invalidate the
 * loaded reference; they are one function because they were three copies of the
 * same two lines, and the copy that forgets to re-check is the one that asks for a
 * clip that does not exist.
 */
function refreshGoldenForControls() {
  state.goldenKey = null;
  if (els.goldenToggle.checked) loadReference(true);
}

async function loadReference(force = false) {
  if (!state.health) return null;
  const duration = Number(els.durationInput.value);
  // `/api/health` already said which durations have a golden run. Asking anyway is a
  // 404 in the console for a fact this page was handed -- so the request is not made
  // and the page says the same thing the server would have.
  const decision = referenceDecision({
    goldenReferences: state.health.golden_references,
    duration,
  });
  if (!decision.fetch) {
    els.goldenToggle.checked = false;
    dropGolden();
    hideGoldenOverlay();
    updateGoldenHelp();
    setStatus(decision.message);
    return null;
  }
  // The golden clip belongs to one prompt. Compare against the clip actually on
  // screen when there is one -- the text box may name a prompt the dropdown does not
  // list yet -- so the reference is never loaded for a different prompt than the
  // motion beside it.
  const entry = selectedEmbedding();
  const shown = state.clip && state.clip.meta ? state.clip.meta.embedding : null;
  const promptHash = (shown && shown.prompt_hash) || (entry ? entry.prompt_hash : '');
  const key = `${duration}|${promptHash}`;
  if (!force && state.goldenKey === key && state.golden) return state.golden;
  try {
    const payload = await getJson(
      `/api/reference?duration=${encodeURIComponent(duration)}`
      + `&prompt=${encodeURIComponent(promptHash)}`);
    state.golden = payload;
    state.goldenKey = key;
    installClip(payload, 'golden');
    els.hud.clip.textContent = `mine + golden (${payload.duration_s.toFixed(1)} s)`;
    return payload;
  } catch (error) {
    const name = error.errorName || 'request_failed';
    dropGolden();
    showError(name, error.message, error.payload && error.payload.detail);
    setStatus(`Golden reference unavailable: ${name}`, true);
    return null;
  }
}

// ---------------------------------------------------------------------------
// clip installation
// ---------------------------------------------------------------------------
function decodeClip(payload) {
  return {
    joints: decodeFloat32(payload.posed_joints_b64),
    roots: decodeFloat32(payload.root_positions_b64),
    contacts: payload.foot_contacts_b64 ? decodeUint8(payload.foot_contacts_b64) : null,
    frameCount: payload.frames,
    jointCount: payload.joints,
    fps: payload.fps,
    duration: payload.duration_s,
    meta: payload,
  };
}

function installClip(payload, role) {
  const clip = decodeClip(payload);
  if (role === 'golden') {
    if (state.goldenRig) state.goldenRig.dispose();
    state.golden = clip;
    state.goldenRig = new Rig(scene, state.skeleton, {
      boneRadius: 0.010,
      jointRadius: 0.011,
      rootRadius: 0.018,
      trailColor: 0xf4b860,
      contactColor: 0xf4b860,
      contactRadius: 0.038,
    });
    state.goldenRig.setTrail(clip.roots, clip.frameCount);
    state.goldenRig.setOpacity(0.5);
    state.goldenRig.setVisible(els.goldenToggle.checked);
  } else {
    if (state.rig) state.rig.dispose();
    state.clip = clip;
    state.rig = new Rig(scene, state.skeleton, {});
    state.rig.setTrail(clip.roots, clip.frameCount);
    state.rig.setVisible(true);
    state.playhead = 0;
    els.frameSlider.max = String(Math.max(0, clip.frameCount - 1));
    els.frameSlider.value = '0';
    // The camera is only refitted for a fresh clip, never per frame.
    fitCameraToClip();
    updateHudForClip(payload);
  }
  updateFrameLabel();
}

function updateHudForClip(payload) {
  els.hud.model.textContent = payload.model;
  els.hud.frames.textContent = `${payload.frames} @ ${payload.fps} fps`;
  els.hud.generate.textContent = fmtMs(payload.generate_ms);
  els.hud.noise.textContent = payload.noise.mode === 'golden'
    ? `golden x_T (${payload.noise.source_files.length} window file(s))`
    : `random (seed ${payload.noise.seed})`;
  els.hud.noise.className = `hud-value ${payload.noise.mode === 'golden' ? 'is-golden' : 'is-mine'}`;
  els.hud.embedding.textContent = `${payload.embedding.prompt_hash} · ${payload.embedding.embedding_sha256.slice(0, 12)}…`;
  els.hud.embedding.title = payload.embedding.embedding_path;

  // Where the conditioning vector came from. This is the one label that must never
  // be wrong: "reference bf16" means the clip is comparable with the golden one,
  // "local MLX" means it is a different, equally valid sample.
  const provenance = payload.embedding.provenance || {};
  const isReference = payload.embedding.source_kind === 'reference';
  els.hud.source.textContent = isReference
    ? 'reference bf16 (golden)'
    : `${provenance.label || payload.embedding.source_kind}${payload.embedding.bits ? ` ${payload.embedding.bits}-bit` : ''}`;
  els.hud.source.className = `hud-value ${isReference ? 'is-golden' : 'is-mine'}`;
  els.hud.source.title = provenance.note || '';

  const timing = payload.timing || {};
  els.hud.encode.textContent = timing.encode_ms === null || timing.encode_ms === undefined
    ? 'cached'
    : fmtMs(timing.encode_ms);
  els.hud.encode.className = 'hud-value';

  els.stats.generate.textContent = fmtMs(payload.generate_ms);
  els.stats.frames.textContent = fmtCount(payload.frames);
  els.stats.windows.textContent = String(payload.windows);
  els.stats.clipFps.textContent = `${payload.fps} fps`;
  els.stats.noise.textContent = payload.noise.mode === 'golden' ? 'golden x_T' : `random (seed ${payload.noise.seed})`;
  els.stats.embedding.textContent = payload.embedding.embedding_sha256;
  els.stats.prompt.textContent = `${payload.embedding.text} (${payload.embedding.prompt_hash})`;
  els.stats.source.textContent = provenance.label || payload.embedding.source_kind;
  els.stats.source.className = isReference ? 'is-golden' : 'is-mine';
  els.stats.source.title = provenance.note || '';
  els.stats.quantization.textContent = payload.embedding.quantization || '—';
  els.stats.encode.textContent = timing.encode_ms === null || timing.encode_ms === undefined
    ? 'not needed (cached)'
    : fmtMs(timing.encode_ms);
  els.stats.total.textContent = fmtMs(timing.total_ms);
  els.stats.endpoint.textContent = payload.served_by;
}

function updateFrameLabel() {
  const clip = state.clip;
  const frame = clip ? Math.min(clip.frameCount - 1, Math.max(0, Math.round(state.playhead))) : 0;
  els.frameLabel.textContent = `frame ${frame} / ${clip ? clip.frameCount - 1 : 0}`;
}

// ---------------------------------------------------------------------------
// playback
// ---------------------------------------------------------------------------
function updatePlayhead(deltaSeconds) {
  const clip = state.clip;
  if (!clip || !state.playing) return;
  state.playhead += deltaSeconds * clip.fps;
  if (state.playhead > clip.frameCount - 1) {
    state.playhead -= clip.frameCount - 1;   // loop
    if (state.playhead < 0) state.playhead = 0;
  }
  els.frameSlider.value = String(Math.round(state.playhead));
  updateFrameLabel();
}

function pushFpsSample(count) {
  state.fpsWindow.push(count);
  if (state.fpsWindow.length > 45) state.fpsWindow.shift();
  state.fps = state.fpsWindow.reduce((a, b) => a + b, 0) / state.fpsWindow.length;
}

// ---------------------------------------------------------------------------
// render loop
// ---------------------------------------------------------------------------
let renderedFrames = 0;
let fpsClock = performance.now();

function frame() {
  requestAnimationFrame(frame);
  const now = performance.now();
  const delta = Math.min(0.1, (now - state.lastTime) / 1000);
  state.lastTime = now;

  updatePlayhead(delta);

  if (state.clip && state.rig) {
    state.rig.setPose(
      state.clip.joints,
      Math.round(state.playhead),
      state.clip.frameCount,
      els.contactsToggle.checked ? state.clip.contacts : null,
    );
  }
  if (state.golden && state.goldenRig) {
    state.goldenRig.setPose(
      state.golden.joints,
      Math.round(state.playhead),
      state.golden.frameCount,
      els.contactsToggle.checked ? state.golden.contacts : null,
    );
  }

  // Camera: orbit around the focus point with a fixed convention (yaw 0 looks
  // down -Z, +Y up), so dragging behaves the same whatever the clip does.
  desiredCamera.set(
    cameraView.focus.x + cameraView.distance * Math.cos(cameraView.pitch) * Math.sin(cameraView.yaw),
    cameraView.focus.y + cameraView.distance * Math.sin(cameraView.pitch),
    cameraView.focus.z + cameraView.distance * Math.cos(cameraView.pitch) * Math.cos(cameraView.yaw),
  );
  camera.position.copy(desiredCamera);
  camera.lookAt(cameraView.focus);

  renderer.render(scene, camera);
  renderedFrames += 1;
  if (now - fpsClock >= 500) {
    pushFpsSample(renderedFrames / ((now - fpsClock) / 1000));
    renderedFrames = 0;
    fpsClock = now;
    els.hud.fps.textContent = `${state.fps.toFixed(0)} fps (page)`;
    els.stats.fps.textContent = `${state.fps.toFixed(1)} fps`;
  }
}

// ---------------------------------------------------------------------------
// wiring
// ---------------------------------------------------------------------------
// Every control that can plan a clip goes through `planIfReady`, so the guard is
// one function that all of them share rather than a check each of them repeats.
els.generate.addEventListener('click', () => planIfReady('generate'));
// Enter is the whole interaction for the text box: type a prompt, press Enter.
els.promptText.addEventListener('keydown', (event) => {
  if (event.key !== 'Enter') return;
  event.preventDefault();
  planIfReady('enter');
});
els.promptTextGo.addEventListener('click', () => planIfReady('go'));
els.promptText.addEventListener('input', () => {
  // Typing is a statement of intent: clear the failure from the previous attempt so
  // the panel does not still be shouting about a prompt that is no longer on screen.
  if (!els.errorBox.hidden) clearError();
});
els.reload.addEventListener('click', async () => {
  // `/api/health` is what fills the dropdown, and this handler used to fire the
  // request without waiting for it: the guard then read a control that had not been
  // populated yet. The round trip finishes first, so what the guard reads is what
  // the user can see.
  await loadHealth().catch(() => {});
  await planIfReady('reload');
});
els.promptSelect.addEventListener('change', () => {
  updatePromptHelp();
  // Picking from the list is a different request from what is typed, so clear the box
  // rather than leave two answers to "which prompt am I generating?" on screen.
  els.promptText.value = '';
  refreshGoldenForControls();
});
els.seedInput.addEventListener('change', () => {
  if (els.noiseSelect.value === 'random') return;
  els.noiseHelp.textContent =
    'Seed is ignored while golden noise is selected: the reference x_T is injected verbatim.';
});
els.noiseSelect.addEventListener('change', () => {
  updateDurationConstraints();
});
els.durationInput.addEventListener('change', () => {
  updateDurationConstraints();
  refreshGoldenForControls();
});
els.durationStep.addEventListener('click', () => {
  const step = state.health ? state.health.duration_step_s : 2;
  els.durationInput.value = String(Number(els.durationInput.value) + step);
  updateDurationConstraints();
  refreshGoldenForControls();
});
els.seedNext.addEventListener('click', () => {
  els.seedInput.value = String(Number(els.seedInput.value) + 1);
});
els.playPause.addEventListener('click', () => {
  state.playing = !state.playing;
  els.playPause.textContent = state.playing ? 'Pause' : 'Play';
  els.playPause.setAttribute('aria-pressed', String(state.playing));
});
els.frameSlider.addEventListener('input', () => {
  state.playhead = Number(els.frameSlider.value);
  state.playing = false;
  els.playPause.textContent = 'Play';
  els.playPause.setAttribute('aria-pressed', 'false');
  updateFrameLabel();
});
els.resetCamera.addEventListener('click', resetCameraView);
els.fitCamera.addEventListener('click', fitCameraToClip);
els.contactsToggle.addEventListener('change', () => {
  if (state.rig) state.rig.setVisible(true);
  if (state.goldenRig) state.goldenRig.setVisible(els.goldenToggle.checked);
});
els.trailToggle.addEventListener('change', () => {
  if (state.rig) state.rig.trail.visible = els.trailToggle.checked;
  if (state.goldenRig) state.goldenRig.trail.visible = els.trailToggle.checked;
});
els.goldenToggle.addEventListener('change', async () => {
  if (!els.goldenToggle.checked) {
    hideGoldenOverlay();
    setStatus('Reference hidden. The clip on screen is this machine\'s own output.');
    return;
  }
  setStatus('Loading the golden reference…');
  // `loadReference` owns the decision for both cases: it either loads the clip, or
  // unchecks this box and says, in the same words the server would have used, that
  // this duration has no golden run -- without a request and without a 404.
  await loadReference(false);
});

function updateDurationConstraints() {
  const step = state.health ? state.health.duration_step_s : 2;
  const raw = Number(els.durationInput.value);
  if (!Number.isFinite(raw) || raw <= 0 || Math.abs(raw % step) > 1e-9) {
    els.durationHelp.classList.add('warn');
    els.durationHelp.textContent =
      `${raw} s is not a whole multiple of the ${step.toFixed(2)} s horizon; the server will `
      + `reject it as duration_not_horizon_multiple.`;
  } else {
    els.durationHelp.classList.remove('warn');
    els.durationHelp.textContent =
      `${raw.toFixed(2)} s = ${Math.round(raw / step)} horizon(s) of `
      + `${state.health ? state.health.horizon : 40} frames.`;
  }
  updateGoldenHelp();
}

// ---------------------------------------------------------------------------
// boot
// ---------------------------------------------------------------------------
window.addEventListener('error', (event) => {
  showError('page_error', String(event.message).slice(0, 400));
});

async function boot() {
  resize();
  requestAnimationFrame(frame);
  try {
    const health = await loadHealth();
    const encoder = health.encoder || {};
    setStatus(
      `Ready. ${health.embeddings.length} cached prompt(s); cold start `
      + `${fmtMs(health.cold_start_s * 1000)}. `
      + (encoder.available
        ? `Local text encoder ready (${encoder.quantization}).`
        : 'No local text encoder: only cached prompts.'));
    updateDurationConstraints();
    await planIfReady('boot');
  } catch (error) {
    setStatus('Could not reach the viewer server.', true);
    showError(error.errorName || 'boot_failed', error.message);
  }
}

boot();
