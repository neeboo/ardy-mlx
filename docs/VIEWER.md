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
| `--llm2vec-repo` | `~/dev/llm2vec-mlx` (or `$ARDY_LLM2VEC_REPO`) | the sibling text-encoder checkout this viewer shells out to. **Never imported** — see §3. |
| `--no-local-encoder` | off | disable local encoding; every prompt must then have a cached vector, exactly as v1 behaved |
| `--encoder-bits` | `4` | which quantized encoder artifact to run (`4` or `8`); the full precision plan is read from the artifact's manifest |
| `--encoder-weights-dir` | `<repo>/weights` | override the encoder's weights directory |
| `--encoder-interpreter` | `<repo>/.venv/bin/python` | override the interpreter (only the sibling `.venv` is ever tried by default) |
| `--encoder-timeout` | `600` | give up on one encoder subprocess after this many seconds |
| `--no-preload` | off | serve before loading weights; the page then reports `model_unavailable` |

Startup prints the URL and the load time, and the process keeps running after the
shell exits under `nohup`. `Ctrl-C` (or `SIGTERM` to the process) shuts it down.

Requirements: `mlx`, `numpy` and the standard library — nothing else, **in this
repository**. The page loads three.js from `viewer/vendor/`, never from a CDN. The
converted weights must exist (`scripts/ardy_convert.py`); without them the server
still starts and `/api/health` says exactly what is missing instead of failing
obscurely.

The local text encoder lives in a **different repository** (`~/dev/llm2vec-mlx`)
with its own virtualenv, and is run as a subprocess (§3). If that checkout is absent,
the viewer still starts, still serves every cached prompt, and reports
`encoder.state = "unavailable"` with the path it looked for.

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
     "has_prompt_text": true, "source": "reference",
     "encoder_id": "reference-llm2vec-bf16", "quantization": null,
     "reference_source": "pancat-linux-ci: LLM2Vec + Meta-Llama-3-8B-Instruct ..."}
  ],
  "embeddings_dir": "artifacts/embeddings",
  "embeddings_detail": {
    "count": 1, "variants": 1,
    "cache_keys": [["e9cf8141cac79f35", "reference-llm2vec-bf16"]],
    "by_source": {"reference": 1, "local": 0},
    "local_dirs": [], "skipped": []
  },
  "encoder": {
    "enabled": true, "available": true, "state": "ready", "reason": null,
    "encoder_id": "mlx-llm2vec-4-bit-group-32-last-2-layers-embed-tokens-at-8-bit",
    "quantization": "4-bit group 32, last-2 layers + embed_tokens at 8-bit",
    "bits": 4,
    "repo": "/Users/…/dev/llm2vec-mlx",
    "interpreter": "/Users/…/dev/llm2vec-mlx/.venv/bin/python",
    "encode_script": "/Users/…/dev/llm2vec-mlx/scripts/encode.py",
    "weights_dir": "/Users/…/dev/llm2vec-mlx/weights",
    "manifest": "/Users/…/dev/llm2vec-mlx/weights/manifest-4bit.json",
    "probe_ms": 0.23, "checks": {"repo": {"…": "…"}, "…": "…"},
    "note": "a subprocess of the sibling checkout llm2vec-mlx, reached through its .venv and scripts/encode.py; nothing is imported, so the two repositories stay decoupled"
  },
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

`encoder` is the *probe*, not a trial encode: it checks that the sibling checkout,
its `scripts/encode.py` and the `<bits>`-bit manifest are all present and readable
(§3). The states are `ready`, `unavailable` (with `reason` naming what is missing)
and `disabled`. No weights are loaded to answer a `GET`.

`embeddings` lists each prompt **once**, as the vector a request would actually use
(the reference when one exists), and `embeddings_detail.cache_keys` lists every
`(prompt_hash, encoder_id)` pair that is addressable — a prompt can have several.

### `POST /api/plan`

```json
{"prompt": "A person waves.",    // any prompt text, or a cached prompt hash
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
  "timing": {                      // the request, split into its two costs
    "encode_ms": 974.2,            // the encoder subprocess; null when the vector was cached
    "generate_ms": 350.4,
    "total_ms": 1330.1,            // server-side wall time for the whole request
    "embedding_source": "encoded", // "cache" (a vector was already there) or "encoded"
    "cache_hit": false
  },
  "posed_joints_b64": "…",         // float32 [frames, 27, 3] LE, base64
  "root_positions_b64": "…",       // float32 [frames, 3] LE, base64
  "foot_contacts_b64": "…",        // uint8 [frames, 4], 0/1
  "foot_contact_order": ["left_heel", "left_toe", "right_heel", "right_toe"],
  "foot_joint_names": ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"],
  "embedding": {"text": "A person walks forward.", "prompt_hash": "e9cf8141cac79f35",
                "embedding_sha256": "9105d788…", "embedding_path": "artifacts/embeddings/…npy",
                "has_prompt_text": true,
                "source": "reference",              // "reference" or "local"
                "source_kind": "reference",         // "reference" | "local-mlx" | "unlabelled (…)"
                "encoder_id": "reference-llm2vec-bf16",
                "quantization": null,               // e.g. "4-bit group 32, …" for a local vector
                "bits": null, "group_size": null,
                "provenance": {"kind": "reference", "label": "reference bf16 (golden)",
                               "note": "the reference machine's own bf16 vector …"}},
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

`embedding.source` is the one field a consumer must not ignore: `"reference"` means
the clip is conditioned on the reference machine's own bf16 vector and is therefore
comparable with the golden clip; `"local"` means this machine encoded it, which is a
**different** (equally valid) condition. `timing.encode_ms` is the encoder
subprocess alone, `null` when the vector came out of the cache — the page shows both
so "1.3 s" never has to be read as "generation is slow".

### `GET /api/reference?duration=2.0&prompt=<hash or text>`

Returns the reference implementation's own clip for that exact duration, in the same
shape as `/api/plan` but with `"kind": "golden"`, `npz_path`, `npz_sha256` and a
`source` block. With `prompt`, it also checks that the golden clip was generated
from that prompt and answers `golden_reference_prompt_mismatch` (404) if not —
comparing two different prompts would not be a comparison.

### Error names

| `error` | status | when |
|---|---|---|
| `local_encoder_unavailable` | 503 | the prompt is not cached **and** the local encoder cannot run (missing checkout/script/manifest). Never substituted. |
| `local_encoder_failed` | 502 | the local encoder ran and refused: a non-zero exit, an over-`max_length` prompt, an unreadable or wrongly-shaped vector. The encoder's own `stderr` is in `detail`. |
| `no_cached_embedding_for_prompt` | 400 | the prompt has no cached vector and this viewer has no local encoder configured (`--no-local-encoder`, or `ViewerApp(local_encoder=None)`). Never substituted. |
| `encoder_variant_not_cached` | 400 | the prompt *is* cached, but not from the encoder that was asked for — the two are different artifacts and are not interchangeable |
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

## 3. Prompts: a text box, two sources of vector, and one rule about precedence

The page takes **any prompt**. If the prompt has no cached vector the server encodes
it here, in about a second, and caches it; the next request for that prompt is a
cache hit. The dropdown stays, listing every vector that is already on disk.

There are two kinds of vector and they are **not interchangeable**:

| | reference bf16 | local MLX |
|---|---|---|
| produced by | LLM2Vec + Llama-3-8B on the reference machine, CPU bf16, `batch_size=1` | the port in `~/dev/llm2vec-mlx`, affine-quantized, on this Mac's GPU |
| lives at | `artifacts/embeddings/<prompt-hash>.npy` | `artifacts/embeddings/<encoder-id>/<prompt-hash>.npy` |
| used for | the clip the golden npz was generated from — the only comparable one | a different, equally valid condition |
| chosen when | a prompt has one — **always** | a prompt has no reference vector |
| `source_kind` in `/api/plan` | `reference` | `local-mlx` |

**The precedence rule is the first row of that table.** A prompt that has a reference
vector is *always* conditioned on it, and the HUD says so. That is what keeps a
viewer clip comparable with the golden reference at all: conditioning on a local
vector for a prompt whose golden clip exists would produce a convincing clip that can
never be compared with anything, and nothing on screen would say why.

### The cache key is `(prompt_hash, encoder_id)`

A prompt hash alone is not a cache key, because two encoders can both encode the same
prompt and produce different vectors. So:

* `<prompt-hash>` is `prompt_hash(prompt)` — the first 16 hex chars of
  `sha256(prompt.encode())`, the convention `save_outputs` uses.
* `encoder_id` names the encoder **and its exact precision**, e.g.
  `mlx-llm2vec-4-bit-group-32-last-2-layers-embed-tokens-at-8-bit`. The full
  quantization string (`4-bit group 32, last-2 layers + embed_tokens at 8-bit`) is
  written into every sidecar next to it.
* The local vector's directory **is** the `encoder_id` half. One prompt can therefore
  hold a reference vector and several precisions of local vector at once, and no
  install can overwrite another encoder's file.
* The top level is the reference convention: a `.npy` there is a reference vector by
  construction, which is why every pre-existing cache keeps working unchanged.
* A sidecar whose `encoder_id` contradicts its directory is **refused**, not
  relabelled — see `tests/test_viewer_local_encoder.py::test_negative_a_mislabelled_file_in_an_encoders_directory_is_refused`.

### Why the quantization has to be named in full

"4-bit" is not enough to identify the artifact. The 4-bit encoder that clears the
frozen acceptance floor is 4-bit affine **at group 32 with the last two transformer
layers and `embed_tokens` held at 8 bits**; plain 4-bit at group 32 measures a
minimum cosine of **0.9828** against the bf16 reference and does **not** clear the
**0.985** floor. Two different artifacts, two different behaviours — so the sidecar
records the plan, not the nominal width, and `/api/plan` reports it in
`embedding.quantization`.

The name is read out of the artifact's own manifest
(`recipe.quantization_plan` + `quantization.mixed_precision`), never hardcoded, so a
rebuilt artifact with a different plan names itself correctly. A manifest that does
not record the plan yields `None` rather than the nominal width.

### The encoder is a subprocess, not an import

`~/dev/llm2vec-mlx` is a **separate repository** with its own virtualenv, and this
viewer does not import it, vendor it, or add it to `sys.path`. It runs:

```
<llm2vec-mlx>/.venv/bin/python <llm2vec-mlx>/scripts/encode.py \
    --text "<prompt>" --bits 4 --out <tmp>/<prompt-hash>.npy --weights-dir <…> --quiet
```

and reads back the two files that command writes — the `.npy` and its `.json`
sidecar. That is the entire coupling. Three reasons it is a subprocess rather than an
import:

1. the two repositories stay independent; a change on either side fails by name at
   the seam instead of breaking somewhere untraceable;
2. the encoder is a second, much larger model (5.2 GB of weights at 4-bit) and must
   not share this process, its memory, or its MLX GPU stream;
3. the cost is visible. `timing.encode_ms` is the subprocess, reported per request.

Only the sibling checkout's **own** `.venv` is ever tried. This viewer's
`sys.executable` is deliberately not a candidate: it is a different virtualenv on a
different dependency set, and using it would turn "the encoder is missing" into "the
encoder ran under the wrong python".

### What happens when the encoder is not there

The probe (`encoder` in `/api/health`) checks three things and reports which are
missing, without loading any weights:

| state | meaning | what a request for an uncached prompt gets |
|---|---|---|
| `ready` | interpreter, `scripts/encode.py` and the `<bits>`-bit manifest all present | it is encoded |
| `unavailable` | one of the three is missing; `reason` names it and `checks` lists every candidate tried | `503 local_encoder_unavailable`, with the missing paths in `detail` |
| `disabled` | `--no-local-encoder`, or `ViewerApp(local_encoder=None)` | `400 no_cached_embedding_for_prompt` |

And when the encoder runs but refuses — a prompt over the model's `max_length`, a
crashed child, an unreadable or non-`(1, L, 4096)` vector — the request gets
`502 local_encoder_failed` with the encoder's own `stderr` tail in `detail`. Measured
example:

```json
{"error": "local_encoder_failed",
 "message": "the local encoder exited 3 for prompt hash a532f2b3a994c000",
 "detail": "prompt hash: a532f2b3a994c000
encoder exit code: 3
encoder refused: prompt needs 2010 tokens, over the reference's max_length 512; truncation would change the pooled vector, so the encoder refuses to guess"}
```

**Nothing in that path substitutes.** There is no branch that falls back to another
prompt's vector, to another encoder's vector for the same prompt, or to "the nearest"
cached prompt. Every one of those would produce a perfectly plausible clip under the
wrong conditioning signal, which is the failure this repository treats as the worst
one available. The negative controls that hold that line are in
`tests/test_viewer_negative_controls.py` and `tests/test_viewer_local_encoder.py`.

### The vector sidecar is documentation

Nothing verifies that a `.npy` really is the encoding of the text its sidecar names.
A mislabelled file produces a perfectly convincing clip under the wrong prompt, and
no test can catch it (see `embeddings/README.md`, which is kept as the reference-vector
recipe). The `(prompt_hash, encoder_id)` key
narrows the blast radius — it makes it impossible to confuse *which encoder* produced
a vector — but it cannot make the sidecar true.

---

## 4. The page

The right-hand panel is the whole control surface; the canvas is the clip.

* **Prompt (free text)** — type any prompt and press **Enter** (or `Go`). If the
  prompt has no vector, the server encodes it locally first; the status line says so
  while it happens and the HUD shows how long it took. The label under the box comes
  from `/api/health`'s `encoder` block, so it says what precision is in use — or, when
  the encoder is missing, that only the cached prompts can run.
* **Cached text embeddings (dropdown)** — every vector on disk from
  `/api/health.embeddings`, each labelled **`reference bf16`** or **`local MLX 4-bit`**,
  with the full `encoder_id` and quantization in its tooltip. Picking one clears the
  text box: there should never be two answers to "which prompt is being generated?".
  A prompt with a reference vector is listed (and used) as the reference even when a
  local vector for it also exists.
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
* **Reference (golden)** — still only available for the two durations that have a
  reference run on disk (2.0 s and 4.0 s). For any other duration the toggle reports
  `no_golden_reference_for_duration` and draws **nothing** rather than a nearby clip.
  When the clip on screen was conditioned on a *local* vector, the overlay is still
  loaded — it is the reference implementation's clip for that duration and prompt —
  but the two are **not** aligned, and the HUD's `vector` field is what says so.
* **HUD** (over the canvas) — clip, model, frames, generate time, playback fps, which
  `x_T` was injected, the **vector source** (`reference bf16 (golden)` in amber, or
  `local MLX 4-bit` in mint), the **encode time** (or `cached`), and the embedding
  fingerprint. The panel repeats those and adds device, parameter count, cold start,
  window count, model fps, the full embedding sha256, the prompt hash, the vector
  source, the quantization, the encode time, the server-side total, and what the local
  encoder is.

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

**New for the local encoder** (nothing in the model path was touched):

* `src/ardy_mlx/sampling/local_encoder.py` — the whole subprocess adapter: probing,
  the `(prompt_hash, encoder_id)` naming, the sidecar validation, and the named
  failures. It imports nothing from `llm2vec-mlx`, and the sibling repository was not
  modified.
* `scripts/ardy_viewer.py` — `EmbeddingCache` now keys entries by
  `(prompt_hash, encoder_id)`, scans one directory per encoder, and installs an
  encoded vector atomically with its sidecar. `ViewerApp` grew the
  resolve→encode→cache path and the `timing`/`embedding.source` fields.

Two small read-plus-additive changes were made to shared code, both recorded here:

* `src/ardy_mlx/sampling/pipeline.py`: added `MlxRunner`, `build_mlx_runner` and
  `mlx_parameter_count` (the shared loader). No existing function changed.
* `src/ardy_mlx/model/nn.py`: added `WeightStore.total_elements()`, a read-only
  count that does not mark tensors used (so `/api/health` can report a parameter
  count without making `WeightStore.unused()` lie).
* `src/ardy_mlx/sampling/pipeline.py`: added `sha256_text()`, the full-digest form of
  `prompt_hash()`, so the viewer resolves a 64-hex prompt selector on the same
  encoding the 16-char cache name uses. No existing function changed.
* `scripts/ardy_generate.py`: `_run_mlx_backend` now calls the shared loader instead
  of carrying its own copy of it. Behaviour and output are unchanged — the CLI's
  golden-reproduction tests still pass.

---

## 6. Boundaries: what is *not* true

1. **A locally-encoded vector does not align with the golden reference.** The local
   MLX encoder is available and works (measured in §7), but it is a *different
   artifact* from the reference bf16 encoder: 4-bit affine at group 32 with the last
   two transformer layers and `embed_tokens` held at 8 bits. Against the encoder
   repository's 16 reference vectors it measures a minimum cosine of **0.9875765**
   (8-bit: **0.9997453**), both clearing that repository's frozen 0.985 floor. Close
   is not the same as aligned: diffusion is chaotic in its conditioning signal, so two
   nearby text vectors give two clips that are **different but equally valid**, not
   two clips that agree. A local vector therefore **cannot** reproduce, or be compared
   with, a golden clip. This is why a reference vector always wins when a prompt has
   one, and why the HUD names the vector's source on every clip.
2. **The reference bf16 vectors are pinned to the reference machine.** They come from
   the one authorised encoder-only run and are never recomputed here. They are the
   only vectors that make a golden comparison meaningful, so they are preferred by
   construction: a prompt that has one is reported as `source_kind: "reference"` no
   matter what else is cached for it.
3. **Encoding is once per prompt, then cached.** The first request for an uncached
   prompt pays a whole encoder subprocess: measured p50 **974 ms**, p90 **993 ms**
   here (§7). The vector is then on disk, and every later request for that prompt is a
   ~110 ms cache hit. There is no batching across requests, no persistent encoder
   process, and no cross-prompt reuse — each distinct prompt costs one encode. (The
   encoder's own warm-process encode is ~0.4 s; this viewer cannot use that, because
   it starts a fresh subprocess each time, and the subprocess number is the honest
   one.)
4. **Only 2.0 s and 4.0 s have a golden reference.** They are the repository's only
   two authorised reference runs. For any other duration the reference toggle shows
   nothing and says why.
5. **A `golden`-noise clip is not a byte-for-byte reference reproduction.** It is the
   reference's own `x_T` through this port: measured `posed_joints` residues of
   `5.178e-06` (2.0 s) and `4.813e-06` (4.0 s) against the golden npz. The `random`
   noise mode is *not* comparable at all — MLX has no Philox stream (README boundary 5).
6. **No post-processing.** The reference's C++ `MotionCorrection` (trajectory
   correction + foot IK) is not built here, so clips retain **foot skating**, exactly
   as both golden clips do (`--no-postprocess`). The foot-contact markers show the
   model's own contact *labels*, not a skate-free constraint being satisfied.
7. **The viewer only ever drew `posed_joints`.** Joint limits, contacts as a hard
   constraint, `--constraints`, and every G1 preset are untested here: only the Core
   27-joint checkpoint is on disk, and the page's duration step and skeleton come
   from that preset.
8. **Not real-time.** A 2.0 s clip costs ~110 ms of sampling warm (~130 ms on a busy
   host) and a 4.0 s clip ~257 ms, plus, on a cache miss, a whole encoder subprocess
   (~1.0 s). The page blocks on the request (no streaming, no partial playback). The
   playback fps in the HUD is the *page's* frame rate.
9. **Single request at a time.** One thread owns MLX, so a second request waits. That
   is a deliberate trade for not aborting the process. The encoder is a subprocess, so
   its ~1 s does not hold the MLX stream — but it does hold the request.
10. **No authentication and no TLS.** Loopback only. Anything that can reach the port
    can generate motion with the checkpoint — and can now also run the text encoder,
    which writes a vector into the cache.
11. **The embedding sidecar is documentation.** Nothing verifies that a `.npy` really
    is the encoding of the text its sidecar names. A mislabelled file produces a
    perfectly convincing clip under the wrong prompt, and no test can catch it. The
    `(prompt_hash, encoder_id)` key closes the *encoder* half of that hole — two
    encoders' vectors can no longer be confused, and a file whose sidecar contradicts
    its directory is refused — but it cannot make the prompt text true.
12. **The playback is a slerp-free replay of stored joint positions.** Scrubbing and
    playback interpolate the *positions* frame-to-frame; the animation is only as
    smooth as the 20 fps clip. Bones are drawn between joint positions, so a limb's
    roll about its own axis is not visible.

---

## 7. Measured on this machine

Apple Silicon, macOS 26.5.2 (`arm64`), 10 cores, MLX 0.32.3, Metal, checkpoint
`ARDY-Core-RP-20FPS-Horizon40` (191,253,304 parameters), served by
`scripts/ardy_viewer.py --port 8850` and driven over HTTP against `POST /api/plan`
with golden noise. The local encoder was `llm2vec-mlx` at 4-bit, in its own
`.venv`.

**Load average during these runs: 8.41 / 9.15 / 10.07** (1 / 5 / 15 min) at the
start and **8.96 / 9.23 / 10.08** at the end. This host is shared and busy; that
context is part of the number, not a footnote to it.

### Sampling and the golden comparison

| quantity | value |
|---|---|
| weight load (`cold_start_s`, printed at startup) | **0.13–0.22 s** (tokenizer 0.013 s + denoiser 0.164 s) |
| `POST /api/plan`, 2.0 s / 40 frames / 1 window, warm cache | **p50 108 ms** (5 requests, min 106, max 108); 129 ms p50 across the first 20 warm requests on the busier host |
| `POST /api/plan`, 4.0 s / 80 frames / 2 windows, warm cache | **p50 257 ms** (min 254, max 268) |
| `GET /api/reference?duration=2.0` | **10 ms**, 18,732 bytes |
| `posed_joints` vs `artifacts/golden-run/out/golden_core_h40_2s_seed0.npz` (golden noise) | **5.178e-06** |
| `posed_joints` vs `artifacts/golden-run-4s/out/golden_core_h40_4s_seed0.npz` (golden noise) | **4.813e-06** |
| `root_positions` vs the 2.0 s golden npz | 5.141e-06 |
| `foot_contacts` vs the 2.0 s golden npz | identical (0 of 160 entries differ) |

What does **not** move is the alignment: the `posed_joints` residue is exactly the
value `artifacts/align-report.json` records, re-measured at the end of this round
(`5.1781535e-06`) — the viewer goes through the same code path, so it lands on the
same number rather than a new, more flattering one.

### Local encoding, over HTTP, 20 never-cached prompts

Each prompt encoded exactly once (4-bit, this machine), then generated at 2.0 s.
20 prompts, one request each, serially:

| quantity | value |
|---|---|
| `timing.encode_ms` (the whole encoder subprocess) | **p50 974 ms**, p90 993 ms, min 958, max 1067 |
| `timing.generate_ms` (sampling alone) | p50 129 ms, p90 138 ms |
| `timing.total_ms` (server-side, whole request) | **p50 1114 ms**, p90 1142 ms |
| client wall clock | p50 1116 ms, p90 1149 ms |
| warm-cache request, same prompt | **p50 111 ms**, `encode_ms: null`, `cache_hit: true` |
| viewer peak RSS (wasm) | 334 MiB before any encode, **1030 MiB** peak during the loop, 938 MiB idle after |
| encoder subprocess peak RSS (4-bit) | **5190 MiB** (`/usr/bin/time -l`, own process) |
| encoder subprocess peak RSS (8-bit) | **7862 MiB** |

The encoder's peak is ~5.2 GB and the viewer's is ~1.0 GB, and they are **different
processes**: the viewer's own footprint returns to ~0.9 GB when the encode finishes.
That separation is the reason the encoder is a subprocess (§3) rather than an import —
an in-process encoder would put 5.2 GB inside the same address space as the ARDY
checkpoint for the whole session.

The first uncached prompt of a session measured 1367 ms rather than ~974 ms (cold
page cache on the 5.17 GB of encoder weights, which are evicted between sessions on
this host). The encoder repository's own numbers, for the parts this viewer does not
control, are a 1.36 s (4-bit) / 1.83 s (8-bit) cold start and a 401 ms / 361 ms warm
encode — the gap between 401 ms and the 974 ms measured here is the cost of a fresh
subprocess per prompt, and it is reported rather than hidden.

### Encoder fidelity (from the encoder repository, not re-measured here)

The local encoder is accepted against 16 reference vectors in
`llm2vec-mlx/docs/status-deepseek.md`:

| artifact | min cosine vs the bf16 reference | floor | result |
|---|---|---|---|
| 8-bit group 64, no exceptions | **0.9997453** | ≥ 0.997 | pass |
| 4-bit group 32, last-2 layers + embed_tokens at 8-bit | **0.9875765** | ≥ 0.985 | pass |
| 4-bit group 32, **no** exceptions (not used here) | 0.9828 | ≥ 0.985 | **fail** |

That last row is why the quantization is named in full in every sidecar (§3). This
repository does not re-derive those numbers — it reports them from the artifact's own
decision record, and `tests/test_viewer_local_encoder.py` checks the weaker property
that the wiring uses the right artifact (cosine > 0.98) rather than restating the
acceptance measurement.

## 8. Tests

```sh
./.venv/bin/python -m pytest tests/test_viewer_api.py tests/test_viewer_http.py \
    tests/test_viewer_negative_controls.py tests/test_viewer_service.py \
    tests/test_viewer_local_encoder.py -q
```

| file | needs weights? | covers |
|---|---|---|
| `tests/test_viewer_api.py` | no | prompt lookup (hash / text / full sha256 / miss), duration rules, noise selection, payload shape, golden-reference parsing, the shared loader's error messages, the encoder block and the cache-key listing in `/api/health` |
| `tests/test_viewer_http.py` | no | the real `HTTPServer` over a loopback socket: routes, status codes, error names, static whitelist and traversal probes, body limits, `405`s, and that the server is **not** the threading variant |
| `tests/test_viewer_negative_controls.py` | partly | deliberate faults that must be refused **or** must visibly change the clip: a missing embedding, a partial horizon, a missing golden clip, a wrong-window `x_T`, a partial multi-window injection, the wrong prompt, the wrong seed, and `random` noise failing to reproduce the reference |
| `tests/test_viewer_local_encoder.py` | **two of its tests need the encoder** | the quantization naming, the `(prompt_hash, encoder_id)` cache key, reference precedence, free-text encode-and-cache, the per-encoder directory layout, and the named refusals (`local_encoder_unavailable`, `local_encoder_failed`, `encoder_variant_not_cached`, a wrong-shaped vector, a non-finite vector, a vector labelled with another encoder, a mislabelled file in an encoder's directory) |
| `tests/test_viewer_service.py` | yes | the shipped command over HTTP with real weights and a golden comparison, the CLI/viewer frame-identity check, and the 2.0 s / 4.0 s golden residues |

The tests that need the converted checkpoint or the golden dumps **skip with a
reason** when they are absent, because neither is committed (`NOTICE`).
`tests/test_viewer_api.py` and `tests/test_viewer_http.py` need neither: a fake
runner returns correctly-shaped feature blocks, and the HTTP tests fake the FK in
numpy so no test ever touches MLX from a non-MLX thread.

`tests/test_viewer_local_encoder.py` is split the same way, and the split is the
point:

* the cache-key, precedence, layout and refusal tests run against
  `FakeLocalEncoder` (in `tests/conftest.py`), which implements exactly the two
  methods the viewer uses — `status()` and `encode()`. They are exact and always run.
* **only two tests touch the real sibling checkout**
  (`test_the_real_encoder_answers_through_the_file_interface` and
  `test_the_real_encoder_is_close_to_the_reference_vector`), and both `skip` with a
  reason when `~/dev/llm2vec-mlx` or its 4-bit manifest is absent. They are the only
  tests that can prove the CLI flags, the output paths and the sidecar field names
  this adapter relies on are the ones the other repository actually implements. They
  are **not** the acceptance measurement — that lives with the encoder, in its own
  decision record — and the threshold they assert (cosine > 0.98) is deliberately
  looser than the encoder's own 0.985 floor, because their job is to catch this
  repository wiring the wrong artifact, not to re-judge the artifact.

`tests/conftest.py` owns the fixtures used across the files (a real `ViewerApp`
wired to a fake runner, a one-prompt embedding cache, and the encoder-wired variants)
plus the synthetic feature block, the fake runner and `FakeLocalEncoder`. The viewer
module itself is loaded by path under the fixed name `ardy_viewer` — by the fixture
*and* by each test file — so a `monkeypatch.setattr(ardy_viewer, ...)` reaches the
same module object the server under test is using, and so
`pytest.raises(ardy_viewer.ViewerError)` catches the exception the server actually
raised. **Loading it twice makes both of those silently wrong**: the monkeypatch
lands on a copy, and `ViewerError` becomes two different classes, so `pytest.raises`
fails against an exception that looks perfect. Every by-path loader therefore
consults `sys.modules` first and registers the module before executing it, and
`tests/test_viewer_api.py::test_reference_payload_shape` is the check that catches a
regression (it compares a digest reported by the patched function).
