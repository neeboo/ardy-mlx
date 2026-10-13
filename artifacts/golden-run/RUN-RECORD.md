# Golden reference run — record

The single authorised reference inference run on `pancat-linux-ci`, used as the
alignment oracle for the MLX port. **One run, as authorised.** Written
2026-10-13 by the A side (DeepSeek).

## 1. What was run

| item | value |
|---|---|
| reference repo | `nv-tlabs/ardy` @ `693f74d13b3d04a0a22ce127ee79c929dd89756b` (`~/services/ardy-engine/ardy`) |
| checkpoint | `ARDY-Core-RP-20FPS-Horizon40` (the only one on disk) |
| entry point | `scripts/generate.py` |
| prompt | `A person walks forward.` |
| duration | **2.0 s** → 40 frames @ 20 fps = **exactly one generation horizon** |
| `--num_samples` | 1 |
| `--seed` | 0 |
| `--diffusion_steps` | default (unset) → `num_base_steps` = **10** |
| `--history_frames` | default (unset) → **160** |
| `--no-postprocess` | **set** (ruling 2 of `docs/decisions/0002`) |
| `--cfg_weight` | default `[2.0, 2.0]` (`cfg_type: separated`, 3 passes) |
| `--checkpoints_dir` | `~/services/ardy-engine/checkpoints` |
| `--output` | `~/ardy-golden-2026-10-13/out/golden_core_h40_2s_seed0` |

Exact command (also in `run.log`):

```sh
cd ~/services/ardy-engine/ardy
HF_HOME=~/services/ardy-engine/hf-cache \
CHECKPOINTS_DIR=~/services/ardy-engine/checkpoints \
TEXT_ENCODER_MODE=local TEXT_ENCODER_DEVICE=cpu \
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_ENABLE_PARALLEL_LOADING=YES \
PYTHONPATH=~/ardy-golden-2026-10-13/hook \
ARDY_GOLDEN_DUMP=~/ardy-golden-2026-10-13/raw \
ARDY_MERGED_BASE=~/services/ardy-engine/llm2vec-base-merged \
ARDY_SUPERVISED_ADAPTER=~/services/ardy-engine/source-models/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp-supervised \
OMP_NUM_THREADS=16 \
/usr/bin/time -v ~/services/ardy-engine/venv/bin/python scripts/generate.py \
  "A person walks forward." \
  --model ARDY-Core-RP-20FPS-Horizon40 --duration 2.0 --num_samples 1 --seed 0 \
  --no-postprocess --checkpoints_dir ~/services/ardy-engine/checkpoints \
  --output ~/ardy-golden-2026-10-13/out/golden_core_h40_2s_seed0
```

## 2. The minimum viable configuration (and why)

The text encoder is the binding constraint: LLM2Vec + Llama-3-8B bf16 is ~16 GB
against **8 GB VRAM and 14 GB RAM**. Measured facts on that box:

* the local production recipe (`~/services/gmgn-motion-service/.../run_ardy.sh`)
  already pins the encoder with `TEXT_ENCODER_DEVICE=cpu`, and the box has a
  purpose-built `swap-gmgn-ardy.img` (8 GB) — so **CPU bf16 with swap** is the
  established local configuration, not something invented here;
* `load_text_encoder()` ends with `.to(device=<cuda>, dtype=bfloat16)`
  (`load_model.py:150-152`), which would move 16 GB onto an 8 GB card, so the
  encoder had to be kept on CPU explicitly;
* the default preset points `base_model_name_or_path` at
  `McGill-NLP/LLM2Vec-Meta-Llama-3-8B-Instruct-mntp`, which is **not** in
  `HF_HOME`. `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` were set so a missing
  local path would fail loudly instead of silently downloading ~15 GB. The local
  merged base (`llm2vec-base-merged`, 13.98 GiB) and the local supervised adapter
  were used instead, exactly as `run_ardy_server.py` does.

**Recorded deviation:** text encoder on **CPU, bfloat16** (the reference default
dtype), i.e. no precision reduction — placement only. This is the only change
relative to the reference default of "cuda if available".

Instrumentation (`sitecustomize.py`, in this directory) was injected via
`PYTHONPATH`; **no file inside the reference repo was modified** (`git status`
clean afterwards). It records the text embedding, `torch.randn` (the initial
noise), every denoising step, the FSQ/denoiser module I/O, and runs one extra
read-only tokenizer `encode -> decode` probe on the gold clip's own features
(a forward pass of the frozen 35 M-parameter tokenizer — **not** a second
generation).

## 3. Resource usage

| metric | value |
|---|---|
| wall clock | **40.87 s** |
| user / system CPU | 65.68 s / 34.81 s (245 % CPU) |
| **peak RSS** | **13,337,108 KB ≈ 13.34 GB** |
| swap used during the run | ≈ 2.2 GB (of 12 GB) |
| major page faults | 122,395 |
| **peak VRAM** | **2,308 MiB** total; the pre-existing `gmgn-motion-service` held 1,097 MiB throughout, so this run added ≈ **1,211 MiB**. GPU utilisation 0–3 % (`gpu_samples.txt`) |
| denoising loop | 10 steps, ≈ 1.2 s |
| exit status | 0 |

GPU sampler: `run.sh` polled `nvidia-smi` every 2 s (`gpu_samples.txt`). The
pre-existing `gmgn-motion-service` on port 2337 was left untouched.

## 4. Artifacts

Copied back to this repo (all under `artifacts/golden-run/`, and **not
committed** — see §6):

| path | content |
|---|---|
| `out/golden_core_h40_2s_seed0.npz` | the reference's own output: `local_rot_mats`, `global_rot_mats`, `posed_joints`, `root_positions`, `smooth_root_pos`, `foot_contacts`, `global_root_heading`, `fps`, `text` |
| `raw/text_feat.npy` | **the text embedding the model consumed**, `(1, 1, 4096)` float32 |
| `raw/text_pad_mask.npy` | `(1, 1)` bool |
| `raw/motion_normalized.npy` | the model's returned normalised features, `(1, 40, 330)` — the `frames` target after un-normalisation |
| `raw/denoise_step{k}_x_in.npy` | `x_t` at the start of DDIM step `k`, `(1, 10, 148)`; **step 0 is the reference's `x_T`** (the window has no history, so `x == x_T`) |
| `raw/denoise_step{k}_x_out.npy` | `x_{t-1}` after step `k` |
| `raw/denoiser_raw_{k}.npy` | raw two-stage denoiser output for the 3 CFG passes, `(3, 10, 148)` |
| `raw/cfg_combined_{k}.npy` | after `uncond + 2.0*(text-uncond) + 2.0*(cstr-uncond)`, `(1, 10, 148)` |
| `raw/fsq_tokenize_0_{in,out}.npy` | tokenizer encode I/O: `(1, 40, 330)` → `(1, 10, 128)` |
| `raw/fsq_detokenize_{0,1}_{in,out_root,out_body}.npy` | tokenizer decode I/O, `(1,10,128)` → `(1,40,4)` + `(1,40,325)` |
| `raw/fsq_requantize_0_{in,out}.npy` | the stats-wrapped `requantize` round trip |
| `raw/probe_*.npy` | the extra encode→decode probe on the gold clip (input / hybrid / token mask / round-trip features) |
| `raw/{call_meta,text_encode_meta}.json` | run parameters and dtypes |
| `raw/hook.log` | per-call trace, incl. the `t` value of every denoising step |
| `run.log`, `gpu_samples.txt`, `run.sh`, `sitecustomize.py` | the run record and the instrumentation |

Note `raw/trace.json` is a **stale, empty** file from the pre-flight check: the
final `json.dump` in the hook failed on a non-serialisable numpy array (see §5),
so the flush aborted **after** the per-step writes but **before** writing
`randn_*.npy`, `diffusion_buffers.npz` and `modules/`. `trace.json` and
`raw/modules/` from this run must therefore be ignored.

## 5. What is missing, and how it was recovered

The hook's `atexit` flush crashed with
`TypeError: Object of type ndarray is not JSON serializable` (the
`generation_token_mask` arrays were stored inside the JSON payload). Because the
run happened once, this was **not** re-run. Consequences:

* `randn_0.npy` was not written — but `x_T` is **exactly** `denoise_step0_x_in.npy`,
  because the window has no history: `x = zeros(B, 10, 148); x[:, 0:10] = x_T`.
  Verified by construction, and the full-clip reproduction (level 3) confirms it.
* `diffusion_buffers.npz` was not written — the schedule is hardcoded pure maths
  (`diffusion.py:12-92`) and is reproduced from the source instead.
* `modules/*.npy` (per-module internals such as the patchified encoder input and
  the per-stage projections) were not written. The public-API I/O captured during
  the run (`fsq_*`, `denoiser_raw_*`, `cfg_combined_*`) was sufficient for all
  three alignment levels; deeper per-module tensors are the one thing a re-run
  would add.

All three alignment levels are nevertheless fully covered, and the reproduction is
verified end to end (§4 of `docs/status-deepseek.md`).

## 6. Why these files are not committed

`NOTICE` states the policy: *we do not redistribute NVIDIA weights or golden
outputs derived from them*. The `.npz` and every `raw/*.npy` are outputs derived
from NVIDIA weights, so they stay on disk uncommitted. The repo-root `.gitignore`
ignores `out/` but **not** `raw/`, so `artifacts/golden-run/.gitignore` closes that
gap (`raw/`, `out/`) without touching the root `.gitignore` — otherwise a single
`git add -A artifacts/` would have staged weight-derived outputs. The converted
weights live under `checkpoints/`, which is gitignored. What *is* committed is the
instrumentation, the run record, and the tensor manifest (names and shapes only,
no values).

## 7. Machine hygiene

* No installs, no `sudo`, no system/config changes, no other GPU work.
* No credentials printed or stored.
* **No processes left behind** (`generate.py`, the sampler loop and
  `nvidia-smi` were all reaped; GPU returned to its 1,097 MiB baseline).
* The reference repo git tree is unmodified.
* Only one new directory was created on that machine:
  `~/ardy-golden-2026-10-13/` (hook + raw dumps + logs + the gold `.npz`).
  It can be deleted at any time; nothing else depends on it.
* Residual: `free` showed swap used going from 2,035 MB to ≈2,248 MB (page-cache
  and swap residue from loading the 8B encoder). No configuration was changed.
