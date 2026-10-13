# The browser viewer

`scripts/ardy_viewer.py` serves a small page (`viewer/`) that samples ARDY motion
**live on this machine** with the real MLX checkpoint, plays it back with a 27-joint
skeleton, and can overlay the golden reference clip for the same duration.

```sh
./.venv/bin/python scripts/ardy_viewer.py --port 8850
# -> http://127.0.0.1:8850/
```

Everything below is meant to be checkable. Where a claim has a number, the number
came from a run on this machine (recorded in §7), and where something is not true,
it is in §6 rather than left for the reader to discover.

---

## 1. Starting and stopping

```sh
# the normal way (nohup so it survives the shell)
cd ~/dev/ardy-mlx
nohup ./.venv/bin/python scripts/ardy_viewer.py --port 8850 > /tmp/ardy-viewer.log 2>&1 &
curl -s http://127.0.0.1:8850/api/health | head -c 400

# or in the foreground, with the log on the terminal
./.venv/bin/python scripts/ardy_viewer.py --port 8850
```

| flag | default | meaning |
|---|---|---|
| `--host` | `127.0.0.1` | bind address. **Anything but loopback is refused** (`exit 2`): the viewer has no authentication and no TLS. |
| `--port` | `8850` | port |
| `--model` | `ARDY-Core-RP-20FPS-Horizon40` | checkpoint preset (any of `io.PRESETS`) |
| `--device` | `gpu` | MLX device |
| `--allow-cpu` | off | required for `--device cpu`; the golden comparisons were all measured on the GPU |
| `--embeddings-dir` | `artifacts/embeddings` | the cached-embedding directory (§3) |
| `--checkpoint-dir` | `checkpoints/<model>` | checkpoint root |
| `--no-preload` | off | serve before loading weights; the page then reports `model_unavailable` |

Startup prints the URL and the load time, and the process keeps running after the
shell exits under `nohup`. `Ctrl-C` (or `SIGTERM` to the process) shuts it down.

Requirements: `mlx`, `numpy` and the standard library — nothing else. The page
loads three.js from `viewer/vendor/`, never from a CDN. The converted weights must
exist (`scripts/ardy_convert.py`); without them the server still starts and
`/api/health` says exactly what is missing instead of failing obscurely.

### Threading, and why it is not negotiable

The MLX GPU stream is **thread-local**. Loading the model on one thread and
evaluating on another fails immediately — measured on this machine:

```
RuntimeError: There is no Stream(gpu, 1) in current thread.
```

So the server is a plain single-threaded `http.server.HTTPServer` (not the
`Threading` variant), and `main()` starts **one** worker thread that both loads the
weights and serves every request on the socket it bound. That thread is the only
place MLX is ever touched. The consequence is that requests are handled one at a
time: a generation blocks the next request until it finishes.

### Security

Loopback only, and no authentication, by construction. It is a local development
viewer: anything that can reach `127.0.0.1:8850` can generate motion with the
checkpoint on this machine. Do not expose the port.

---

## 2. The HTTP interface

Three routes plus static files. Every failure carries a stable machine-readable
`error` name, and `POST /api/plan` covers the whole generation path.

### `GET /api/health`

```json
{
  "status": "ok",                       // "unavailable" when the model failed to load
  "error": null,                        // the loader error, verbatim, when it did
  "model": "ARDY-Core-RP-20FPS-Horizon40",
  "device": {"name": "Device(gpu, 0)", "backend": "gpu", "platform": "darwin", "mlx_version": "0.32.3"},
  "parameters": 191253304,
  "cold_start_s": 0.19,                 // weight load, including one mx.eval
  "load_seconds": {"tokenizer_s": 0.013, "denoiser_s": 0.164, "total_s": 0.177},
  "fps": 20, "horizon": 40, "num_frames_per_token": 4, "motion_rep_dim": 330,
  "duration_step_s": 2.0,               // horizon / fps: the only legal duration step
  "duration_min_s": 2.0,
  "num_base_steps": 10,
  "default_cfg_weight": [2.0, 2.0],
  "skeleton": {
    "name": "cskel27",
    "jointNames": ["Hips", "Spine", "... 27 entries"],
    "jointParents": [-1, 0, 1, "... 27 entries"],
    "rootIndex": 0,
    "neutralJoints": [[0.0, 0.0, 0.0], "... 27 x 3"],
    "footJointNames": ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"],
    "footJointIdx": [25, 26, 21, 22]
  },
  "embeddings": [
    {"prompt_hash": "e9cf8141cac79f35", "text": "A person walks forward.",
     "path": "artifacts/embeddings/e9cf8141cac79f35.npy",
     "embedding_sha256": "9105d788...", "shape": [1, 1, 4096], "tokens": 1,
     "has_prompt_text": true, "source": "artifacts/golden-run/raw/text_feat.npy"}
  ],
  "embeddings_dir": "artifacts/embeddings",
  "golden_references": [
    {"duration_s": 2.0, "path": "artifacts/golden-run", "available": true},
    {"duration_s": 4.0, "path": "artifacts/golden-run-4s", "available": true}
  ],
  "limits": {"text_encoder": "...", "postprocess": "...", "rng": "...", "auth": "...", "threading": "..."},
  "served_by": "scripts/ardy_viewer.py (127.0.0.1)"
}
```

`skeleton` is read straight out of `ardy_mlx.model.skeleton` (`core_skeleton()`), so
the page's joint names and parents cannot drift from the FK that produced
`posed_joints`. Nothing on the client hardcodes a second joint table.

### `POST /api/plan`

```json
{"prompt": "e9cf8141cac79f35",   // a prompt hash, or the exact prompt text
 "seed": 0,                       // integer, or omitted
 "duration": 2.0,                 // seconds; must be a whole number of horizons
 "noise": "golden",               // "golden" (default) or "random"
 "diffusion_steps": 10,           // optional, 1..num_base_steps
 "cfg_weight": [2.0, 2.0],        // optional; 1 value = text only, constraint 0.0
 "include_frames": false}         // optional; adds the raw 330-wide frames
```

`GET /api/plan?prompt=&seed=&duration=&noise=` accepts the same parameters as query
strings (no `cfg_weight` list). Body and query are merged, the body winning.

Response:

```json
{
  "kind": "plan",
  "model": "ARDY-Core-RP-20FPS-Horizon40",
  "frames": 40, "joints": 27, "fps": 20,
  "duration_s": 2.0, "duration_requested_s": 2.0,
  "windows": 1,                    // autoregressive windows
  "num_autoregressive_steps": 1,
  "offsets": [0, 40],              // frame index where each window starts
  "history_frames": 160,           // crop_history_length actually used
  "diffusion_steps": 10, "cfg_weight": [2.0, 2.0],
  "seed": 0,
  "generate_ms": 350.4,            // wall time of the sampling call alone
  "posed_joints_b64": "…",         // float32 [frames, 27, 3] LE, base64
  "root_positions_b64": "…",       // float32 [frames, 3] LE, base64
  "foot_contacts_b64": "…",        // uint8 [frames, 4], 0/1
  "foot_contact_order": ["left_heel", "left_toe", "right_heel", "right_toe"],
  "foot_joint_names": ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"],
  "embedding": {"text": "A person walks forward.", "prompt_hash": "e9cf8141cac79f35",
                "embedding_sha256": "9105d788…", "embedding_path": "artifacts/embeddings/…npy",
                "has_prompt_text": true},
  "noise": {"mode": "golden", "seed": 0, "windows": 1,
            "source_files": ["artifacts/golden-run/raw/denoise_step0_x_in.npy"], "note": "…"},
  "postprocess": false,
  "reference": {"repo": "nv-tlabs/ardy", "commit": "693f74d"},
  "served_by": "127.0.0.1 scripts/ardy_viewer.py"
}
```

`posed_joints` is world-space joint positions, Y-up, metres, from the real
`CoreSkeleton27` FK over the generated feature frames — the same value
`scripts/ardy_generate.py` writes into its npz. `root_positions` is the root joint,
and `posed_joints[:, 0, :] == root_positions` exactly.

The three arrays are **base64 little-endian float32**, not JSON number lists: the
bytes that leave the server are the bytes MLX produced. (`include_frames` adds
`frames_330_b64`, the contract's `(T, 330)` feature block, at ~4x the payload.)

### `GET /api/reference?duration=2.0&prompt=<hash or text>`

Returns the reference implementation's own clip for that exact duration, in the same
shape as `/api/plan` but with `"kind": "golden"`, `npz_path`, `npz_sha256` and a
`source` block. With `prompt`, it also checks that the golden clip was generated
from that prompt and answers `golden_reference_prompt_mismatch` (404) if not —
comparing two different prompts would not be a comparison.

### Error names

| `error` | status | when |
|---|---|---|
| `no_cached_embedding_for_prompt` | 400 | the prompt has no cached embedding (§3). Never substituted. |
| `no_prompt_given` | 400 | no `prompt` at all |
| `duration_not_horizon_multiple` | 400 | `duration` is not a whole number of horizons |
| `invalid_duration` | 400 | missing, zero, negative, NaN or infinite |
| `unknown_noise_mode` | 400 | `noise` is neither `golden` nor `random` |
| `no_golden_noise_for_duration` | 400 | no recorded reference `x_T` for that duration |
| `invalid_parameter`, `invalid_json`, `invalid_cfg_weight` | 400 | malformed request |
| `no_golden_reference_for_duration` | 404 | no golden clip for that exact duration |
| `golden_reference_prompt_mismatch` | 404 | the golden clip belongs to another prompt |
| `golden_reference_missing` | 404 | the run directory exists but has no npz |
| `not_found` | 404 | unknown path (static files are an exact whitelist) |
| `method_not_allowed` | 405 | anything but `GET`/`POST` |
| `body_too_large` | 413 | request body over 1 MiB |
| `model_unavailable` | 503 | the checkpoint is not loaded; `/api/health` has the reason |
| `internal_error` | 500 | anything unforeseen, with a traceback in `detail` |

---

## 3. Prompts, and why there is a dropdown instead of a text box

**v1 has no text encoder.** The port consumes a *precomputed* `[B, L, 4096]`
embedding; LLM2Vec + Llama-3-8B was never ported (it is the ~16 GB the reference
run had to keep on the CPU — `artifacts/golden-run/RUN-RECORD.md` §2). So the page
offers the prompts that already have an embedding on disk, and nothing else.

* The cache lives in `artifacts/embeddings/<prompt-hash>.npy` (+ an optional `.json`
  sidecar carrying the prompt text). `<prompt-hash>` is
  `prompt_hash(prompt)` — the first 16 hex chars of `sha256(prompt.encode())`, the
  same convention `save_outputs` uses.
* A request may name the prompt by hash (`GET`-friendly) or by its exact text; the
  text is hashed, so both paths agree by construction.
* **A miss is an error**: `400 no_cached_embedding_for_prompt`, naming the file it
  looked for. The server never falls back to another prompt and the page shows the
  error name verbatim.

`embeddings/README.md` has the recipe for adding one — it has to run on a machine
that can run the reference text encoder.

---

## 4. The page

The right-hand panel is the whole control surface; the canvas is the clip.

* **Prompt** — the cached embeddings from `/api/health`, labelled with their text and
  hash. Nothing here can invent a prompt.
* **Seed** — used only by `random` noise. With `golden` the reference `x_T` is
  injected and the seed changes nothing (the help text says so).
* **Duration** — in seconds, stepped by one horizon (2.0 s for Core/20FPS/H40). A
  value that is not a whole multiple is flagged in the panel *and* refused by the
  server with `duration_not_horizon_multiple`.
* **Initial noise** — `golden` (the reference's own `x_T`) or `random` (MLX's RNG).
* **Generate & play** — posts to `/api/plan` and plays the result.
* **Playback** — pause/resume, a frame slider that scrubs and pauses, and
  `Playback fps` measured over the last ~45 rendered frames (that is page frame rate,
  not the clip's 20 fps, which is listed separately as `Clip fps (model)`).
* **Camera** — drag to orbit, wheel to zoom, double-click to reset, `Fit clip` to
  frame the current clip's bounding box.
* **Ground grid** — a 24 m grid plus 1 m subdivisions; the clip's root path is drawn
  as a trail at ground height.
* **Foot-contact markers** — an octahedron per contact joint, shown while its channel
  is set. The four channels are `(left_heel, left_toe, right_heel, right_toe)` and
  attach to `LeftFoot`/`LeftToeBase`/`RightFoot`/`RightToeBase`, whose indices arrive
  with the skeleton from the server.
* **Reference (golden)** — loads `/api/reference` for the current duration and draws
  it as a translucent amber skeleton beside the mint-green generated one; both play
  on the same playhead. If no golden clip exists for that duration the toggle reports
  `no_golden_reference_for_duration` and shows **nothing** rather than a nearby clip.
* **HUD** (over the canvas) — clip, model, frames, generate time, playback fps, which
  `x_T` was injected, and the embedding fingerprint. The panel repeats those and adds
  device, parameter count, cold start, window count, model fps, the full embedding
  sha256 and the prompt hash.

---

## 5. What is reused, and what is new

**Copied byte-for-byte** (MIT, from `~/dev/motionbricks-mlx/viewer/vendor/`):

| file | sha256 |
|---|---|
| `viewer/vendor/three.module.min.js` | `852c06b9fe936cf8ebc2870c86370d1015c310b04de98cdfc07f4efaf6afd2af` |
| `viewer/vendor/three.core.min.js` | `4183cee05f0aa093682fdf551363a16bb20bd92b68e14f7576905f45c461ba82` |
| `viewer/vendor/THREE-LICENSE.txt` | `df2a91696186fe58c92518a25d48bc59c8ee2edc7cac9ca6d19a5a0d599a2a05` |

(`cp -p`; the digests are of both the source files and the **staged git blobs**, so
what is in the repository is byte-for-byte what was copied. The licence text ships
beside them. three.js does not print a clean revision string in these bundles; the
shipped header is `Copyright © 2010-2025 three.js authors` and the core bundle is
the r185 line, so identify it by digest.)

The root `.gitattributes` marks `viewer/vendor/**` as `-text`. That is not
decoration: this machine has `core.autocrlf=true`, which rewrites line endings when
a file is added (`CRLF` -> `CRCRLF`) and silently breaks the recorded digests. The
attribute is what keeps the byte-identity claim checkable.

**New, written for this repository**: every other file here — `viewer/index.html`,
`viewer/app.js`, `viewer/rig.js`, `viewer/style.css`, `scripts/ardy_viewer.py`, the
tests, this document. The MotionBricks page's *look* (a full-bleed canvas with a dark
control rail) is the reference for the layout; none of its markup, CSS or
JavaScript was copied.

In particular `viewer/rig.js` is **not** the MotionBricks `SkeletonRig`. That class
poses a bone hierarchy (root position + per-joint quaternions) for a hardcoded
34-joint G1 skeleton; ARDY has 27 joints and this viewer draws `posed_joints` —
world joint positions — so there are no quaternions to interpolate at all. The joint
hierarchy is not written down in JavaScript either: `app.js` passes the names and
parents that `/api/health` sends from `ardy_mlx.model.skeleton`.

**Reused by import, not copied**:

* `ardy_mlx.model.ardy.ArdyBatch` — the window loop, recentring, requantisation and
  detokenisation. The viewer drives it exactly as `scripts/ardy_generate.py
  --backend mlx` does (`tests/test_viewer_service.py` asserts the two produce
  bit-identical frames from the same noise).
* `ardy_mlx.sampling.pipeline.build_mlx_runner` — **the one loader** both entry
  points call. It was factored out of the CLI for this round so the two cannot drift
  apart; `scripts/ardy_generate.py --backend mlx` now calls it too. Nothing in the
  path was rewritten or duplicated.
* `ardy_mlx.model.skeleton.posed_joints_from_features` — the CoreSkeleton27 FK.
* `ardy_mlx.sampling.pipeline.resolve_noise_windows` — the injected-noise validator
  the CLI's `--noise-file` also uses, so "a wrong `x_T` fails loudly" means the same
  thing in both.

Two small read-plus-additive changes were made to shared code, both recorded here:

* `src/ardy_mlx/sampling/pipeline.py`: added `MlxRunner`, `build_mlx_runner` and
  `mlx_parameter_count` (the shared loader). No existing function changed.
* `src/ardy_mlx/model/nn.py`: added `WeightStore.total_elements()`, a read-only
  count that does not mark tensors used (so `/api/health` can report a parameter
  count without making `WeightStore.unused()` lie).
* `scripts/ardy_generate.py`: `_run_mlx_backend` now calls the shared loader instead
  of carrying its own copy of it. Behaviour and output are unchanged — the CLI's
  golden-reproduction tests still pass.

---

## 6. Boundaries: what is *not* true

1. **There is no text encoder.** The prompt list is the set of precomputed
   embeddings on disk. A prompt without one cannot be run, full stop.
2. **Only 2.0 s and 4.0 s have a golden reference.** They are the repository's only
   two authorised reference runs. For any other duration the reference toggle shows
   nothing and says why.
3. **A `golden`-noise clip is not a byte-for-byte reference reproduction.** It is the
   reference's own `x_T` through this port: measured `posed_joints` residues of
   `5.178e-06` (2.0 s) and `4.813e-06` (4.0 s) against the golden npz. The `random`
   noise mode is *not* comparable at all — MLX has no Philox stream (README boundary 5).
4. **No post-processing.** The reference's C++ `MotionCorrection` (trajectory
   correction + foot IK) is not built here, so clips retain **foot skating**, exactly
   as both golden clips do (`--no-postprocess`). The foot-contact markers show the
   model's own contact *labels*, not a skate-free constraint being satisfied.
5. **The viewer only ever drew `posed_joints`.** Joint limits, contacts as a hard
   constraint, `--constraints`, and every G1 preset are untested here: only the Core
   27-joint checkpoint is on disk, and the page's duration step and skeleton come
   from that preset.
6. **Not real-time.** A 2.0 s clip costs ~0.35 s of sampling on this machine and a
   4.0 s clip ~0.8 s, plus the HTTP round trip and the browser's own decode; the page
   blocks on the request (no streaming, no partial playback). The playback fps in the
   HUD is the *page's* frame rate.
7. **Single request at a time.** One thread owns MLX, so a second request waits. That
   is a deliberate trade for not aborting the process.
8. **No authentication and no TLS.** Loopback only. Anything that can reach the port
   can generate motion with the checkpoint.
9. **The embedding sidecar is documentation.** Nothing verifies that a `.npy` really
   is the encoding of the text its sidecar names. A mislabelled file produces a
   perfectly convincing clip under the wrong prompt, and no test can catch it.
10. **The playback is a slerp-free replay of stored joint positions.** Scrubbing and
    playback interpolate the *positions* frame-to-frame; the animation is only as
    smooth as the 20 fps clip. Bones are drawn between joint positions, so a limb's
    roll about its own axis is not visible.

---

## 7. Measured on this machine

Apple Silicon, macOS 26.5.2 (`arm64`), MLX 0.32.3, Metal, checkpoint
`ARDY-Core-RP-20FPS-Horizon40` (191,253,304 parameters), served by
`scripts/ardy_viewer.py --port 8850` and driven with `curl` against
`POST /api/plan` with golden noise.

| quantity | value |
|---|---|
| weight load (`cold_start_s`, printed at startup) | **0.22 s** (tokenizer 0.013 s + denoiser 0.164 s) |
| `POST /api/plan`, 2.0 s / 40 frames / 1 window | **113–152 ms** sampling over 5 warm requests; **248 ms** on the first request after startup, **350 ms** when the server had just started and the host was busy |
| `POST /api/plan`, 4.0 s / 80 frames / 2 windows | **281–315 ms** sampling; **794 ms** in the earlier, busy-host measurement |
| `GET /api/reference?duration=2.0` | **10 ms**, 18,732 bytes |
| `GET /api/plan?…&noise=random` (2.0 s) | 216 ms wall, including 40 ms of HTTP and decode |
| `posed_joints` vs `artifacts/golden-run/out/golden_core_h40_2s_seed0.npz` (golden noise) | **5.178e-06** |
| `posed_joints` vs `artifacts/golden-run-4s/out/golden_core_h40_4s_seed0.npz` (golden noise) | **4.813e-06** |
| `root_positions` vs the 2.0 s golden npz | 5.141e-06 |
| `foot_contacts` vs the 2.0 s golden npz | identical (0 of 160 entries differ) |

The sampling times move by ~3x between runs on this host (it is shared and its load
average was 13–20 during the earlier measurements), so read the spread as the honest
number rather than the best one. What does **not** move is the alignment: the two
`posed_joints` residues are exactly the values `artifacts/align-report.json` records
for these durations — the viewer goes through the same code path, so it lands on the
same numbers rather than a new, more flattering set.

## 8. Tests

```sh
./.venv/bin/python -m pytest tests/test_viewer_api.py tests/test_viewer_http.py \
    tests/test_viewer_negative_controls.py tests/test_viewer_service.py -q
```

| file | needs weights? | covers |
|---|---|---|
| `tests/test_viewer_api.py` | no | prompt lookup (hash / text / full sha256 / miss), duration rules, noise selection, payload shape, golden-reference parsing, the shared loader's error messages |
| `tests/test_viewer_http.py` | no | the real `HTTPServer` over a loopback socket: routes, status codes, error names, static whitelist and traversal probes, body limits, `405`s, and that the server is **not** the threading variant |
| `tests/test_viewer_negative_controls.py` | partly | deliberate faults that must be refused **or** must visibly change the clip: a missing embedding, a partial horizon, a missing golden clip, a wrong-window `x_T`, a partial multi-window injection, the wrong prompt, the wrong seed, and `random` noise failing to reproduce the reference |
| `tests/test_viewer_service.py` | yes | the shipped command over HTTP with real weights and a golden comparison, the CLI/viewer frame-identity check, and the 2.0 s / 4.0 s golden residues |

The tests that need the converted checkpoint or the golden dumps **skip with a
reason** when they are absent, because neither is committed (`NOTICE`).
`tests/test_viewer_api.py` and `tests/test_viewer_http.py` need neither: a fake
runner returns correctly-shaped feature blocks, and the HTTP tests fake the FK in
numpy so no test ever touches MLX from a non-MLX thread.
