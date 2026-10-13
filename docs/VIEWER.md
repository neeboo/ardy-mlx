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
| `--no-router` | off | turn semantic routing off entirely: no judgment is made and **nothing leaves this machine** (§9) |
| `--router-deadline-ms` | `2500` | how long one plan may wait for a judgment before degrading to the plain vector route, **by name** (§9) |
| `--no-preload` | off | serve before loading weights; the page then reports `model_unavailable` |

Startup prints the URL and the load time, and the process keeps running after the
shell exits under `nohup`. `Ctrl-C` (or `SIGTERM` to the process) shuts it down.

Requirements: `mlx`, `numpy` and the standard library — nothing else, **in this
repository**. The page loads three.js from `viewer/vendor/`, never from a CDN. The
converted weights must exist (`scripts/ardy_convert.py`); without them the server
still starts and `/api/health` says exactly what is missing instead of failingobscurely.

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
  "golden_references": [{"duration_s": 2.0, "path": "artifacts/golden-run", "available": true}],
  "router": {                           // §9. The same six names a plan carries.
    "enabled": true, "available": true,
    "state": "ok",                      // "router_disabled" | "router_unavailable" | "ok"
    "router_id": "jev-latest+typesafe-sdk==0.7.4",
    "route": "compose",                 // the LAST verdict's route, null before any plan
    "probabilities": {"library": 0.01, "compose": 0.99, "generate": 0.0, "non_motion": 0.0},
    "latency_ms": 308.4, "cache_hit": false,
    "deadline_ms": 2500, "model": "jev-latest", "sdk_version": "0.7.4",
    "key_env": "TYPESAFE_API_KEY",       // the VARIABLE's name, never the key
    "calls": 3, "cache_hits": 1, "timeouts": 0, "cache_size": 2,
    "library_entries": 28, "known_intents": ["A person walks forward.", "…"],
    "last": {"state": "ok", "degraded": false, "judgment": {"is_motion": {"type": "noul", "noul": 0.97}, "…": "…"}},
    "privacy": "on: `utterance` and `context` in every judgment are sent to the hosted TypeSafe API at https://api.typesafe.ai (TYPESAFE_API_KEY)"
  },
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
  "router": {                      // §9. What was read (or the name of why nothing was)
    "state": "ok",                 // "ok" | "cached" | "router_timeout" | "router_unavailable"
                                   // | "router_error" | "router_bad_response"
                                   // | "router_disabled" | "router_no_utterance"
    "router_id": "jev-latest+typesafe-sdk==0.7.4",
    "route": "compose",            // "library" | "compose" | "generate" | "non_motion"
                                   // | "vector" (no judgment) | null
    "probabilities": {"library": 0.01, "compose": 0.99, "generate": 0.0, "non_motion": 0.0},
    "latency_ms": 308.4,           // time THIS request spent on the router (0.0 on a cache hit)
    "cache_hit": false,
    "degraded": false,             // true whenever state is not "ok"/"cached"
    "detail": null, "deadline_ms": 2500,
    "model": "jev-latest", "sdk_version": "0.7.4",
    "known_intents_rev": "7b1a29790c964ec1",   // the third part of the judgment cache key
    "judgment": {"is_motion": {"type": "noul", "noul": 0.97}, "…": "…"}
  },
  "decision": {                    // §9. What the viewer DOES about it
    "route": "compose", "action": "compose_knobs",   // "load_library" | "generate" | "hold" | "vector_fallback"
    "starts_new_action": true,     // false means: no clip is drawn and the current one stays
    "conservative": false, "source": "jev",          // "jev" | "vector"
    "reason": "route=compose p=0.990 conf=0.990",    // machine-readable, language-neutral
    "message": "用现有动作组合：速度 0.65×、幅度 0.66×（Jev 读出 slow and relaxed / subtle）。",
    "params": {"speed": 0.65, "amplitude": 0.656, "tempo": {"score": 0.0, "level": "slow and relaxed",
               "probabilities": {"0": 0.98, "1": 0.02, "2": 0.0}},
               "intensity": {"score": 0.14, "level": "subtle", "probabilities": {"…": 0.0}},
               "emotion": "平静", "expression": "neutral", "gaze": "forward",
               "replaces_current": true, "expression_placeholder": true},
    "library_entry": null          // {"prompt_hash": "…", "text": "…"} for action == "load_library"
  },
  "served_by": "127.0.0.1 scripts/ardy_viewer.py"
}
```

A **hold** is a different `kind` and carries no clip at all:

```json
{"kind": "hold", "reason": "route=non_motion p=0.990",
 "router": {"…": "…"}, "decision": {"starts_new_action": false, "action": "hold", "…": "…"},
 "duration_s": 2.0, "seed": 0, "prompt": "A person walks forward.", "held_action": "…",
 "timing": {"encode_ms": null, "generate_ms": null, "total_ms": 4.1, "router_ms": 281.6}}
```

`kind` is the field a consumer switches on: `"plan"` (a clip is present), `"hold"`
(nothing was drawn — §9) or `"golden"` (`/api/reference`).

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
| `no_prompt_given` | 400 | no `prompt` at all. The page's own guard (`§8.2`) makes this unreachable from the UI: a request without a prompt would be a page bug, not a user error |
| `duration_not_horizon_multiple` | 400 | `duration` is not a whole number of horizons |
| `invalid_duration` | 400 | missing, zero, negative, NaN or infinite |
| `unknown_noise_mode` | 400 | `noise` is neither `golden` nor `random` |
| `no_golden_noise_for_duration` | 400 | no recorded reference `x_T` for that duration |
| `invalid_parameter`, `invalid_json`, `invalid_cfg_weight` | 400 | malformed request |
| `no_golden_reference_for_duration` | 404 | no golden clip for that exact duration. The page does not ask when `/api/health` already said so (`§8.2`) |
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
  on the same playhead. It is still only available for the two durations that have a
  reference run on disk (2.0 s and 4.0 s), and the page **decides from
  `/api/health.golden_references` before it requests anything**: for any other
  duration no request is made at all (so no 404 in the console), the toggle goes back
  to unchecked, and the panel says the duration has no golden run — the same words the
  server would have used, without asking it. When the clip on screen was conditioned
  on a *local* vector, the overlay is still loaded — it is the reference
  implementation's clip for that duration and prompt — but the two are **not**
  aligned, and the HUD's `vector` field is what says so. The one 404 left on this
  endpoint is `golden_reference_prompt_mismatch`, a real refusal, still reported by
  name.
* **Prompt guard** — `Generate`, `Enter`, `Go`, `Reload` and the boot request all
  reach `/api/plan` through one guard (`planIfReady`), so a prompt that has not
  resolved (an empty box *and* a dropdown that is not populated yet) sends **nothing**:
  the page clears any error, says one plain sentence in the status line, and leaves the
  last clip on screen. `Reload` waits for `/api/health` — the round trip that fills the
  dropdown — before it decides. See §8.2 for the 400 this came from.
* **Bubble** (over the canvas, bottom left) — what the viewer understood from the
  sentence, where that came from, and what it is doing about it. It appears **the
  instant a trigger fires**, before the request is built, and is replaced by the
  server's decision when the plan returns. Its colour is the server's own router
  state, so a degraded judgment (no semantic parse) is never drawn like a real one.
  See §9.7.
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
   two authorised reference runs. For any other duration the reference toggle is
   unchecked, says why, and **no `/api/reference` request is made at all**: the page
   already knows from `/api/health.golden_references`, so it does not turn a fact it
   was told into a 404 (§8.2).
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
13. **The front end is not tested in a browser.** There is no browser harness, and no
    test synthesises clicks, keys or input events. What is verified is the *decision
    logic* and its *wiring*: the pure guard functions are run under node out of the
    shipped `viewer/app.js` (§8), and `node --check` proves the page parses. Nothing
    in this repository proves that a click reaches that logic, that the DOM elements
    exist, or that anything renders. **That still needs a human at the page**: a hard
    refresh, one click per trigger, and a look at the console. The tests can be green
    while the page is broken in a way no test claims to cover — this is a boundary,
    not a gap to be quietly closed later.
14. **Routing sends the user's sentence off this machine.** With the router enabled,
    `utterance` and `context` (§9.1) are posted to TypeSafe's hosted API. That is a
    real disclosure of what the user typed, it happens on the first plan of a session,
    and no local test can observe it. `--no-router` (or simply not setting a key)
    keeps the viewer fully local, and `/api/health.router.privacy` states which of the
    three conditions is in force. The **model's judgments are not verified for
    correctness here** either: §9.6 records what it answered for four sentences, and
    one of the four did not match the expectation. A judgment is an input to code, not
    a fact.
15. **The router's parameters are carried, not applied.** `speed`, `amplitude`,
    `expression` and `gaze` travel in `decision.params`, and the viewer displays them,
    but nothing in this repository feeds them into the sampler: the clip is still
    conditioned on the prompt vector alone, and `expression`/`gaze` are explicitly
    marked `expression_placeholder: true` because the rig has no face and no eyes.
    Wiring them into continuous motion is the next layer's job (§5).

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
| `tests/test_router_jev.py` | **no** (its one SDK-shape test skips without the extra) | §9: the state and the question set, the route→action table, the parameter mappings, the thresholds, every named degradation, the judgment cache and its key, the timeout, and that the key is in no router output |
| `tests/test_viewer_router.py` | no | §9's viewer layer: a plan carries `router`/`decision`, `non_motion` returns a hold and draws nothing, a degraded verdict still plans and says it did not parse, `library` conditions on the entry, the judgment overlaps the encoder, and the key is in no response and in no file under `viewer/` |

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

### 8.1 The page's own logic, run under node

The page has no browser test harness, so the part of it that decides **whether to
send a request** would otherwise be untestable. It lives instead in a
marker-delimited region of pure functions inside `viewer/app.js`
(`// --- pure-prompt-guard: start … end`), and that region is loaded out of the
**shipped file** — not copied — and executed in a bare `node:vm` context by
`tests/viewer_js_guard.test.mjs`:

```sh
node --check viewer/app.js                      # the page parses (ESM, no bundler)
node tests/viewer_js_guard.test.mjs             # 12 cases; also: node --test <file>
./.venv/bin/python -m pytest -q                 # the same run, via
                                                # test_viewer_http.py::test_the_pages_prompt_guard_runs_under_node
```

The 12 cases cover: empty text box **and** empty dropdown → nothing resolves;
whitespace is not a prompt (it falls through to the dropdown); text wins over the
selection and is used trimmed; a selected entry is planned by its `prompt_hash`; the
guard region is pure (it is re-evaluated with `document`, `window`, `fetch`,
`state`, `els` and `THREE` replaced by traps that throw); the reference decisions for
a duration with a golden run, one without, one whose npz is missing, and malformed
input; and two **wiring** checks on the real source — `generateClip` keeps its
`shouldPlan` check and builds its body through `planBody`, every trigger calls
`planIfReady`, `Reload` awaits `/api/health`, and `loadReference` decides before it
requests.

Two of those cases are **negative controls**, because a check that cannot fail is not
a check: one asserts that a body built from an unresolved prompt is literally
`{"seed":0,"duration":2,"noise":"golden"}` (no `prompt` key — the 400 above), and the
other re-runs the wiring checks against a deliberately mutated copy of the page and
requires them to fail. `node` is a build-time tool, not a Python dependency: the
pytest bridge skips with a reason when it is not on `PATH`.

### 8.2 A 400 `no_prompt_given` that came from the page

A browser session showed this in the console:

```
app.js:275  POST http://127.0.0.1:8850/api/plan 400 (Bad Request)
  getJson @ app.js:275
  generateClip @ app.js:463
```

The server named it `no_prompt_given`, and the engine and the encoder were both
fine — with a prompt (and `noise=random`, because the reference `x_T` is only
recorded for 2.0 s and 4.0 s) the same `duration=8` request answers 200: 160 frames,
4 windows, 753 ms. The fault was in the page, and the mechanism is worth writing
down, because the request that fails does not *look* empty: **it looks small**.

* `POST /api/plan` reads `params.get("prompt")`, and `_embedding_for_request` refuses
  `None`/`""` by name (`scripts/ardy_viewer.py`; the `plan` route and
  `no_prompt_given`);
* `JSON.stringify` **drops `undefined` values**. A body built from a prompt that never
  resolved is therefore not "a request with an empty prompt" — it is a body with **no
  `prompt` key at all**: `{"seed":0,"duration":2,"noise":"golden"}`;
* the controls that could replan did **not** share a guard: the only check lived
  inside `generateClip`, and each trigger read the controls itself. The `Reload`
  handler at `app.js:725` called `generateClip()` *without* awaiting `loadHealth()` —
  the round trip that **fills the cached-embedding dropdown** — so on the first
  interaction with nothing to plan, the page read a control that had not been
  populated yet, and then reported the *server's* error name in a red box
  (`no_prompt_given`) for what was a page-side "not yet".

What can and cannot be concluded from this tree: the exact body that produces the
server's 400 is the one in the second bullet, and that body is what an unresolved
prompt becomes once it is `JSON.stringify`d — the test asserts it literally. In the
committed `bc00b26` source the empty case already returned `null` before the body was
built, so the plain empty case showed the red box rather than sending; a request that
really lost the field needs a path that built the body from a prompt it never
resolved. That is the class of bug this change removes, and it is no longer reachable:
the body builder now refuses an unresolved prompt, **and** every trigger is checked
before it gets there.

What the page does now (the same rules are stated in §4):

* `resolveRequest()`, `shouldPlan()`, `planBody()` and `referenceDecision()` are pure
  functions in `viewer/app.js`; the DOM call sites only adapt control values into
  their arguments;
* every trigger — `Generate`, `Enter`, `Go`, `Reload`, and the boot request — goes
  through `planIfReady()`. When nothing resolves it builds **no request**, clears any
  error box, and says one plain sentence in the status line (`Nothing to generate yet
  (<trigger>). Type a prompt, or pick a cached embedding from the list.`) instead of
  raising a red box for something that is not a server failure;
* `Reload` awaits `/api/health` first, so what the guard reads is what the user can
  see;
* `generateClip()` keeps its own `shouldPlan()` check as the last place that can
  notice an unresolved prompt;
* `/api/reference` is decided from `/api/health.golden_references` *before* it is
  requested, so a duration without a golden run no longer produces a 404: the toggle
  is unchecked and the panel names the durations that do exist. The remaining 404 on
  that endpoint is `golden_reference_prompt_mismatch`, which is a real refusal and is
  still reported by name.

**Neither of those two verifications is a browser** (§6 boundary 13). They prove the
decision logic and the wiring. A click, a hard refresh and a look at the console are
still what prove the bundled page behaves.

---

## 9. Semantic routing: Jev supplies judgments, code decides

A user says one sentence. The viewer decides what to do. **The model never decides
anything by itself** — it answers a closed set of typed questions and
`src/ardy_mlx/router/jev.py` turns those answers into one of four actions through a
pure, testable mapping. There is no path from model text to engine behaviour.

### 9.1 What leaves the device, and how to stop it

When the router runs, **the user's sentence and the context around it are sent to
TypeSafe's hosted API** (`https://api.typesafe.ai` by default). That is what
`state` is:

```json
{"utterance": "慢慢走过去",
 "context": {"current_action": "A person walks forward.",
             "known_intents": ["A person walks forward.", "…"],
             "library_entries": [{"prompt_hash": "e9cf8141cac79f35", "text": "A person walks forward."}, "…"]}}
```

* The key is read **only** from the server's environment: `TYPESAFE_API_KEY`, or
  `JEV_API_KEY` as a fallback **mapped onto the same slot** (`resolve_api_key()` is
  the single place a key can enter the module). Blank values are not keys.
* The key is never in a request payload, never logged, never in any response, and
  never in a file under `viewer/`. `/api/health` reports the *variable's name*
  (`key_env`), which is what an operator needs and all they need.
* `--no-router` turns the whole thing off: no judgment, no client, nothing sent.
  `/api/health` then says `router_disabled` and the page says so in its bubble.
* With no key set, the viewer runs exactly as it did before routing existed and
  reports `router_unavailable` — not a silent no-op.

### 9.2 The question set

Eight questions, asked in **one** request, all against the same state. One question,
one judgment; every closed set keeps a way out.

| id | primitive | asks | exit |
|---|---|---|---|
| `is_motion` | Noul | does this **ask for or name** a physical action | — (the false case is "about how someone feels") |
| `known` | Noul | is this something this engine can already do | — |
| `route` | Choice | which of four behaviours | `non_motion` is an option, not an absence |
| `tempo` | Score | how fast | 3 levels, `slow and relaxed` / `normal pace` / `hurried` |
| `intensity` | Score | how large and forceful | 3 levels, `subtle` / `moderate` / `vigorous` |
| `emotion` | Choice | the speaker's mood | 平静 / 高兴 / 烦躁 / 低落 |
| `interrupt` | Noul | replace what is happening now, or wait | — |
| `library_pick` | Choice | **which** library entry | `none` |

`library_pick` is asked **speculatively**, in the same request. Without it
`route = "library"` would name a behaviour ("load the library entry") with no way to
say which entry; asking it in the same call costs tokens and no wall clock, because
all questions run in parallel. Its choices are labelled `L1…Ln` because a prompt hash
is not a label a model can reason about — the hash travels in the criteria
description and the code maps the label back.

### 9.3 The mapping table, and the thresholds

`route → action` is total over the four routes and lives in `ROUTE_ACTIONS`:

| `route` | `action` | what happens | `starts_new_action` |
|---|---|---|---|
| `library` | `load_library` | the clip is conditioned on the picked entry's **own cached vector**, not on the sentence that pointed at it | yes |
| `compose` | `compose_knobs` | the sentence's own vector, plus `speed`/`amplitude` from the Scores | yes |
| `generate` | `generate` | the sentence's own vector, neutral knobs | yes |
| `non_motion` | `hold` | **no clip is drawn**; the one on screen stays; the bubble answers | **no** |
| *(no judgment)* | `vector_fallback` | the sentence's own vector, neutral knobs, and the bubble says no semantic parse happened | yes |

Parameters are computed, not guessed:

```
speed      = 1 + (tempo - 1) * 0.35          → 0.65× … 1.35×   (neutral at "normal pace")
amplitude  = 0.6 + intensity/2 * 0.8         → 0.60× … 1.40×   (neutral at "moderate")
expression, gaze = EMOTION_PLACEHOLDERS[emotion]             ← **placeholders**: the rig has no face
                                                               and no eyes, and the payload says so
replaces_current = interrupt >= 0.5
```

Thresholds are three different numbers because the three costs are different
(`Thresholds` in the module):

| threshold | default | what crossing it does |
|---|---|---|
| `route_confidence` | 0.5 | the reading is a guess: the **conservative branch** runs |
| `route_top_probability` | 0.5 | the probability mass is split: same branch |
| `is_motion` | 0.5 | **no new action is started at all** — moving unasked is the worst available outcome |
| `known` | 0.5 | does not change the action; makes the bubble admit the engine is improvising |

The **conservative branch** keeps the nearest route but clamps the parameters to the
gentlest setting (`speed ≤ 1.0`, `amplitude ≤ 1.0`), sets `conservative: true`, and
prefixes the bubble with `我拿不准，先用最接近的：`. A low-confidence reading is never
presented as a confident one.

### 9.4 Asynchronous, with a deadline and a name for missing it

`JevRouter.submit()` starts the call on its own thread and returns immediately. The
request then resolves the embedding — a ~1 s encoder subprocess on a cache miss — and
**only then** collects the judgment, so the two overlap:

```
submit(judgment) ────────────────┐          Jev is thinking (warm p50 ≈ 380 ms)
_embedding_for_request(prompt) ──┤          the encoder is working (~1 s)
handle.resolve(deadline) ────────┘
```

If the deadline passes, the verdict is `router_timeout`, the plan proceeds down the
plain vector route, and the page says `这次没做语义解析（router_timeout：超过 2500 ms）`
— **a degraded answer is named, never passed off as a judgment.** The worker keeps
running and still caches its answer, so the *next* identical sentence is a cache hit
rather than a second charge.

Every degraded state is a value of `router.state`, and they are the API's own error
vocabulary:

| `router.state` | when |
|---|---|
| `ok` | a judgment was made |
| `cached` | a judgment was made earlier and reused |
| `router_timeout` | the deadline passed; the call may still land in the cache |
| `router_unavailable` | no key in the server's environment, or the SDK/network is unreachable |
| `router_error` | the SDK raised something nameable (rate limit, internal error, …) |
| `router_bad_response` | an answer arrived but could not be read as the questions asked |
| `router_disabled` | `--no-router` |
| `router_no_utterance` | the prompt is a hash with no cached text to judge |

Classification is by the SDK's **exception name**, not by words in the message: a
`TypeSafeBadRequestError` whose text happens to contain "timeout" is `router_error`.

### 9.5 The judgment cache

Key: `(prompt_hash, router_id, known_intents_rev)`, where

* `prompt_hash` — the sentence, by the repository's existing convention;
* `router_id` — `"<model>+typesafe-sdk==<version>"`, because a judgment is only
  reusable against both;
* `known_intents_rev` — a hash of `(known_intents, library_entries)`.

The third part is not decoration: `route` and `library_pick` were both chosen
*against the library*, so a library that has changed makes the cached answer invalid
by construction rather than by remembering to clear anything. The observed
consequence is in §9.6: the first plan for a new sentence is a miss (the encoder adds
it to the library), the second shares a key with the third, and the third is a hit.

A cache hit reports `state: "cached"`, `cache_hit: true` and `latency_ms: 0.0` — the
time *this* request spent at the model. The original call's own cost stays in
`judgment.model_latency_ms`, where it is a fact about that call.

### 9.6 Measured, with a real key

`jev-latest` via `typesafe-sdk==0.7.4`, deadline 8000 ms, on the repository's own
27-entry embedding cache (copied to a throwaway directory — no measurement writes
into `artifacts/`). Four sentences, each twice, the library restored before every
call:

| utterance | expected | run 1 | run 2 | `tempo` / `intensity` | `emotion` |
|---|---|---|---|---|---|
| `往前走` | `library` | **library** 0.96 | **library** 0.96 | 1.00 normal / 0.87 moderate | 平静 0.99 |
| `慢慢走过去` | `compose` | **compose** 0.99 | **compose** 0.99 | 0.00 slow / 0.13 subtle | 平静 1.00 |
| `跳过去挥挥手` | `generate` | compose 0.71 | compose 0.70 | 1.14 normal / 0.95 moderate | 高兴 0.96 |
| `今天好累` | `non_motion` | **non_motion** 0.99 | **non_motion** 0.99 | 0.07 slow / 0.08 subtle | 低落 0.99 |
| `A person walks forward.` | `library` | **library** | **library** | 1.00 normal / 0.89 moderate | 平静 1.00 |

The last row is the dropdown's own vocabulary — a third-person *description* rather
than an imperative — and it is in the table because the first version of the
`is_motion` question got it **wrong**: it asked whether the speaker was *requesting*
motion, so every cached library prompt read as a non-request and the viewer answered
`hold` instead of drawing. That was found by driving the running server
(`POST /api/plan` with the cached hash `e9cf8141cac79f35`), not by any test; the
question now asks whether the sentence **asks for or names** an action, and the false
case is about the speaker's own state. Both runs agree on all five sentences. Three of the four matched the expectation.
`跳过去挥挥手` was read as **`compose`**, consistently, and that reading is
defensible rather than wrong: the route definitions reserve `generate` for an action
the library does not contain *and* that cannot be composed, and this library already
contains **"A person jumps over a puddle."** and **"A person waves with both hands."**
— so "jump over and wave" is exactly a composition of existing motions. A user who
wants `generate` for it needs a library that lacks those two entries, or a route
description that weights novelty more heavily. It is recorded here rather than
tuned away.

Actions and bubbles from the same run:

```
往前走        action=load_library  embedding='A person walks forward.'
              照库里那条来：A person walks forward.（Jev 读出这句最接近库里已有的一条）。
慢慢走过去     action=compose_knobs speed=0.65 amplitude=0.656
              用现有动作组合：速度 0.65×、幅度 0.66×（Jev 读出 slow and relaxed / subtle）。
跳过去挥挥手    action=compose_knobs speed=1.07 amplitude=0.99 expression=smile
              用现有动作组合：速度 1.07×、幅度 0.99×（Jev 读出 normal pace / moderate）。
今天好累      action=hold starts_new_action=False kind=hold
              这句不是动作要求（Jev 读 non_motion 的概率 0.99），保持当前动作。
A person walks forward.  action=load_library  embedding='A person walks forward.'
              照库里那条来：A person walks forward.（Jev 读出这句最接近库里已有的一条）。
```

Latency, ten warm calls in one process: **min 284.7 ms, p50 440.6 ms, max 1646.7 ms**
(an earlier eight-call run on a quieter host measured min 281.6 / p50 379.3 / max 560.0 ms;
the spread between the two runs is this shared host's load, not the router).
The judgment cache: call 1 `ok` (miss), call 2 `ok` (`known_intents_rev` changed —
the encoder had added the sentence to the library), call 3 `cached`, `latency_ms 0.0`.

### 9.7 The bubble

The bubble is the page's sentence about its own understanding. It appears **the instant
a trigger fires** — before the request is built — and is replaced by the server's
decision when the plan returns. Its `data-state` is set from the server's own router
state, so a degraded judgment is drawn differently from a real one
(`degraded` → amber, `hold` → gold, `routed` → mint, `thinking` → blue). The meta line
carries `state=… · route=… p=… · <router_id> · <latency> · source=…`, which is where
"what it chose, why, and where that came from" is legible without opening a console.

Its copy is **Chinese**, unlike the rest of the page: the sentences it explains are
usually Chinese, and the bubble is product copy for the person speaking rather than a
developer diagnostic. It is one table (`MESSAGES`) plus one prefix
(`CONSERVATIVE_PREFIX`) in `src/ardy_mlx/router/jev.py` — changing the language is
changing those, and nothing else.
