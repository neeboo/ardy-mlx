# Golden reference run 2 — record

The **second and final** authorised reference inference run on `pancat-linux-ci`
(ruling 4 of `docs/decisions/0003-alignment-baseline.md`), used as the
multi-window alignment oracle for the MLX port. **One run, as authorised.**
Written 2026-10-13 by the A side (DeepSeek).

## 1. What was run

Identical to run 1 (`artifacts/golden-run/RUN-RECORD.md`) in every respect
**except `--duration`**, which is `4.0` instead of `2.0`:

* 4.0 s × 20 fps = **80 frames** = `2 × gen_horizon_len` = **two autoregressive
  windows** — the smallest clip that exercises window 1 at all.

| item | value |
|---|---|
| reference repo | `nv-tlabs/ardy` @ `693f74d13b3d04a0a22ce127ee79c929dd89756b` (`~/services/ardy-engine/ardy`) |
| checkpoint | `ARDY-Core-RP-20FPS-Horizon40` (the only one on disk) |
| entry point | `scripts/generate.py` |
| prompt | `A person walks forward.` |
| duration | **4.0 s** → 80 frames @ 20 fps = **two generation horizons** |
| `--num_samples` | 1 |
| `--seed` | 0 |
| `--diffusion_steps` | default (unset) → `num_base_steps` = **10** |
| `--history_frames` | default (unset) → **160** |
| `--no-postprocess` | **set** (ruling 2 of `docs/decisions/0002`) |
| `--cfg_weight` | default `[2.0, 2.0]` (`cfg_type: separated`, 3 passes) |
| `--checkpoints_dir` | `~/services/ardy-engine/checkpoints` |
| `--output` | `~/ardy-golden-2026-10-13-4s/out/golden_core_h40_4s_seed0` |
| new directory on that machine | `~/ardy-golden-2026-10-13-4s/` (**run 1's directory untouched**) |

Exact command (also in `run.log`, and reproducible from `run.sh`):

```sh
cd ~/services/ardy-engine/ardy
HF_HOME=~/services/ardy-engine/hf-cache \
CHECKPOINTS_DIR=~/services/ardy-engine/checkpoints \
TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENABLE_PARALLEL_LOADING=YES \
PYTHONPATH=~/ardy-golden-2026-10-13-4s/hook \
ARDY_GOLDEN_DUMP=~/ardy-golden-2026-10-13-4s/raw \
ARDY_MERGED_BASE=~/services/ardy-engine/llm2vec-base-merged \
ARDY_SUPERVISED_ADAPTER=~/services/ardy-engine/source-models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
OMP_NUM_THREADS=16 \
/usr/bin/time -v ~/services/ardy-engine/venv/bin/python scripts/generate.py \
  "A person walks forward." \
  --model ARDY-Core-RP-20FPS-Horizon40 --duration 4.0 --num_samples 1 --seed 0 \
  --no-postprocess --checkpoints_dir ~/services/ardy-engine/checkpoints \
  --output ~/ardy-golden-2026-10-13-4s/out/golden_core_h40_4s_seed0
```

The only recorded deviation from the reference defaults is the same one as run 1:
the 8B LLM2Vec text encoder runs on **CPU, bfloat16** (placement only, no
precision reduction), because the box has 8 GB VRAM / 14 GB RAM against a ~16 GB
encoder. See run 1's record §2 for the full justification.

## 2. Resource usage

| metric | run 2 (`--duration 4.0`) | run 1 (`--duration 2.0`) |
|---|---|---|
| wall clock | **38.26 s** | 40.87 s |
| user / system CPU | 70.59 s / 43.68 s (298 %) | 65.68 s / 34.81 s (245 %) |
| **peak RSS** | **13,255,268 KB ≈ 13.26 GB** | 13,337,108 KB ≈ 13.34 GB |
| major page faults | 101,657 | 122,395 |
| **peak VRAM** | **2,308 MiB** total; the pre-existing `gmgn-motion-service` held 1,097 MiB throughout → this run added ≈ **1,211 MiB** | 2,308 MiB total (same) |
| swap used | 2,241 MB → max 3,393 MB sampled; **2,255 MB after the run** | 2,035 → ≈2,248 MB |
| denoising loop | 2 × 10 steps | 10 steps |
| exit status | **0** | 0 |

The two runs cost essentially the same: the text encoder (8B, CPU bf16) dominates,
and doubling the clip only adds a second 10-step loop (the whole motion model is
~108 ms per 40-frame window on this Mac, so ~0.2 s of the 38 s).

GPU sampler: `run.sh` polls `nvidia-smi` every 2 s (`gpu_samples.txt`). The
pre-existing `gmgn-motion-service` on port 2337 was left untouched.
`mem_samples.txt` records `free -m` every 2 s; its per-process `VmRSS` line is
**not** usable (it caught the monitor subshell, not `generate.py`) — the peak RSS
above is from `/usr/bin/time -v`, which is authoritative.

## 3. Instrumentation (`sitecustomize.py`, v2) — and the fix for run 1's data loss

Run 1's `atexit` flush died with
`TypeError: Object of type ndarray is not JSON serializable` (the
`generation_token_mask` arrays were stored inside a JSON payload). Because
*everything* was written at exit, the crash took `randn_*.npy`,
`diffusion_buffers.npz` and `modules/` with it, and the second authorised run was
already spent.

**The v2 hook makes that class of failure impossible, in two independent ways,
and adds the per-window records the multi-window task needs.**

Fix 1 — *the flush is no longer load-bearing*:

* `_jsonable()` recursively sanitises every record (`ndarray` → inline list for
  ≤ 64 elements, otherwise a shape/dtype/min/max summary; numpy scalars,
  `torch.Size`, tensors, nested dicts/lists handled). Validated on the exact run-1
  crash shape before the run started, in a pre-flight that seeded `STATE` with an
  ndarray record and let the exit hook run in a scratch directory: flush completed
  with `errors=0`.
* every artefact is written **incrementally, at the moment it is observed**
  (`_save()` for `.npy`, `_jsonl()` for one flushed JSON line per event,
  `_write_json()` via temp-file + `os.replace`). The exit hook now only writes the
  optional aggregate `trace.json`, and reports its own failure instead of raising.
  Result on the real run: `trace.json` → `errors: []`, and every dump present.

Fix 2 — *the module hooks no longer retain tensors*: each `modules/*.npy` is
written from the forward hook and dropped immediately, instead of accumulating in
`STATE` until exit (which also removes run 1's never-realised RSS contribution).

New per-window records (this is what makes per-window alignment possible at all):

| path | content |
|---|---|
| `randn_{w}.npy` | the reference's own `x_T` for window `w` — **recovered: this is run 1's missing file** |
| `window{w}_history_in.npy` / `_history_out.npy` | the accumulated hybrid sequence entering / leaving the window |
| `window{w}_x0.npy` | **the exact `x` the window starts from** (= `denoise_step{10w}_x_in.npy`), i.e. history + fresh noise |
| `window{w}_first_heading_in.npy` | the `first_heading_angle` **fed into** window `w`; for `w ≥ 1` this *is* window `w-1`'s refreshed value |
| `window{w}_global_transl_in.npy` | the accumulated translation entering the window |
| `recenter{w}_center_pos.npy`, `recenter{w}_new_root_motion.npy`, `recenter{w}_history_recentered.npy` | the per-step recentring, captured inside `Ardy._recenter_history` |
| `masks_{w}.npz`, `token_masks_{w}.npz` | the three frame masks and the three **token** masks per window |
| `final_hybrid_translated.npy` | the translated hybrid sequence handed to the decoder |
| `windows.jsonl`, `recenters.jsonl`, `steps.jsonl`, `fsq.jsonl`, `modules.jsonl`, … | one flushed JSON line per event, each tagged with its window phase |
| `modules/*.npy` (506 files) | per-module inputs/outputs — **recovered: run 1 lost all of these** |

The full run-1 dump set is also present and unchanged in shape
(`text_feat`, `denoise_step*`, `denoiser_raw_*`, `cfg_combined_*`, `fsq_*`,
`probe_*`), now with 20 denoising steps instead of 10.

## 4. Artifacts

Copied back to this repo under `artifacts/golden-run-4s/` and **not committed**
(see §5):

| path | content |
|---|---|
| `out/golden_core_h40_4s_seed0.npz` | the reference's own output: `local_rot_mats`, `global_rot_mats`, `posed_joints`, `root_positions`, `smooth_root_pos`, `foot_contacts`, `global_root_heading`, `fps`, `text` (all `(80, …)`) |
| `raw/text_feat.npy`, `raw/text_pad_mask.npy` | the text conditioning the model consumed, `(1,1,4096)` / `(1,1)` |
| `raw/randn_{0,1}.npy` | the two initial-noise draws, `(1,10,148)` each — the **first** time these are on disk |
| `raw/motion_normalized.npy` | the model's returned normalised features, `(1,80,330)` |
| `raw/denoise_step{k}_{x_in,x_out}.npy` | all 20 DDIM steps, `(1,20,148)` (window 0's `x` has 20 tokens: 10 generation + 10 masked future) |
| `raw/denoiser_raw_{k}.npy`, `raw/cfg_combined_{k}.npy` | raw three-pass output `(3,20,148)` and the CFG composition `(1,20,148)`, all 20 steps |
| `raw/window{w}_*.npy`, `raw/recenter{w}_*.npy`, `raw/{masks,token_masks}_{w}.npz` | the per-window records listed in §3 |
| `raw/final_hybrid_translated.npy`, `raw/final_motion_decoded_autoregressive.npy` | the translated hybrid `(1,20,148)` and the chunked-decode result `(1,80,330)` (identical to `motion_normalized.npy`) |
| `raw/modules/*.npy` (506) + `raw/modules.jsonl` | per-module I/O — **recovered from run 1's loss** |
| `raw/*.jsonl`, `raw/hook.log`, `raw/trace.json` | the per-event trace; `trace.json` flush completed with `errors: []` |
| `run.log`, `gpu_samples.txt`, `mem_samples.txt`, `run.sh`, `sitecustomize.py` | the run record and the (fixed) instrumentation |

Size on disk: `raw/` ≈ 64 MB, `out/` ≈ 188 KB.

Cross-checks that the dumps are self-consistent (all verified locally):

* `denoise_step0_x_in[:, :10] == randn_0` and `denoise_step10_x_in[:, :10] == window1_history_in`;
* `recenter0_history_recentered == window1_history_in` (window 1's history *is*
  window 0's recentred output);
* `final_motion_decoded_autoregressive == motion_normalized` (the chunked decode is
  what `__call__` returns);
* `fsq_detokenize_0_in` is `(1,10,128)` and `fsq_detokenize_1_in` is `(1,20,128)` —
  the two chunks of the autoregressive decode;
* `window0` token masks: history 0 / generation 10 / **future 0** (even though the
  frame-level future mask is 40 frames); `window1`: history 10 / generation 10 / future 0.

## 5. Why these files are not committed

Same policy as run 1: `NOTICE` states that we do not redistribute NVIDIA weights
or golden outputs derived from them, so the `.npz` and every `raw/*.npy` stay on
disk uncommitted. `artifacts/golden-run-4s/.gitignore` ignores `raw/` and `out/`
exactly as `artifacts/golden-run/.gitignore` does, without touching the repo-root
`.gitignore`. What *is* committed is the instrumentation, this record, and the
alignment report.

## 6. Machine hygiene

* **One run**, launched once and never retried; no other GPU work; no installs, no
  `sudo`, no system or configuration changes; no credentials printed or stored.
* The reference repo's `git status --porcelain` was empty **before and after** the
  run (`GIT_STATUS_BEFORE=0` / `GIT_STATUS_AFTER=0`).
* **No processes left behind**: after the run `pgrep -af 'generate.py|nvidia-smi'`
  matched nothing, and the GPU returned to its 1,097 MiB baseline.
* One new directory on that machine: `~/ardy-golden-2026-10-13-4s/` (hook + raw
  dumps + logs + the gold `.npz`). The run-1 directory was left untouched.
* A pre-flight scratch directory (`~/ardy-golden-2026-10-13-4s/preflight/`) was
  created, used to validate the fixed exit hook, and then deleted (verified absent).
* Residual: swap used 2,241 MB → 2,255 MB (page-cache/swap residue from loading the
  8B encoder; peak sampled 3,393 MB). No configuration was changed.
