# Status · B side (sampling / autoregressive loop / generate CLI)

- Scope of this file: B's own files only — `src/ardy_mlx/sampling/**`,
  `scripts/ardy_generate.py`, `tests/test_sampling_*.py`, `tests/test_generate_*.py`.
- Contract: `docs/motion-contract.md` (A's, frozen). Every number below cites it.
- Environment: `./.venv/bin/python` (decision 0001). MLX 0.32.3, numpy 2.5.4, pytest 9.1.1.

## 1. What exists now

| File | Contents |
|---|---|
| `src/ardy_mlx/sampling/diffusion.py` | cosine β schedule (S4.1), `space_timesteps`, `resolve_num_steps` (S4.3), deterministic DDIM (S4.2) |
| `src/ardy_mlx/sampling/cfg.py` | `separated` three-pass layout and composition (S4.4) |
| `src/ardy_mlx/sampling/window.py` | window planning, right-aligned history crop, recentring, accumulated translation, `(cos,sin) → θ` (S5) |
| `src/ardy_mlx/sampling/autoregressive.py` | the chunk loop, model-agnostic via a `step_fn` callable (S5.3–S5.5) |
| `src/ardy_mlx/sampling/demo.py` | **placeholder** backend: random-weight, shape-faithful stand-in for the two stages |
| `src/ardy_mlx/sampling/pipeline.py` | embedding loading, output assembly, `.npz` + sidecar (S7); `resolve_history_frames` (S5.2) and `resolve_noise_windows` (S11 N16) |
| `scripts/ardy_generate.py` | CLI; `--backend placeholder` (local loop) and `--backend mlx` (A's model); exports the skeleton FK for `posed_joints` |

Implementation notes that are load-bearing:

- The cosine schedule is built in python floats and stored **float32**: `float64` is
  unsupported on the MLX GPU backend, and the released weights are all `F32` anyway.
- The DDIM schedule is indexed by a **scalar** `t`. The reference writes
  `alphas_cumprod_prev[t, None, None]` on a 1-D table, which does not index a batch
  axis; indexing a python int here keeps the semantics identical instead of
  accidentally broadcasting.
- The loop keeps two buffers: `trajectory` (the generated frames, recentred every
  step) and `context` (the crop that is handed to the model, which spills back into
  the zero seed block until enough frames exist). History is never noised and never
  re-denoised; only the generation tokens are (contract S5.3).
- The initial context is a **zero** block of `default_history_frames` and is *not*
  part of the output; the clip is `num_autoregressive_steps × horizon` frames, which
  is the reference's own frame count (`ardy_model.py:569-570`). A's `ArdyBatch` uses
  a zero-length initial history instead, so for the same `--duration` the two
  backends produce the same frame count (verified: 240 frames for 12 s, both).

## 2. Test output

```
./.venv/bin/python -m pytest -q
214 passed, 1 xfailed in 21.36s
```

Breakdown (all green): schedule 16, cfg 8, window 24, loop 23, pipeline 28, CLI 15,
plus A's existing io/model tests. (`test_sampling_*` / `test_generate_*` are B's;
`test_io_*` / `test_model_*` are A's.) Round 5 added 15 tests and removed or
weakened none; the single `xfail` is unchanged (see §4.4).

Negative controls, i.e. tests that exist to prove the others can fail:

| Control | What it proves |
|---|---|
| `test_negative_control_dropping_the_cosine_offset_is_detected` | dropping `+0.008` changes `alpha_bar` and `beta_0` by >1e-4, so the pointwise test is falsifiable |
| `test_negative_control_skipping_the_shift_would_break_alphas_cumprod_prev` | the DDIM anchor is a *shifted* table with a leading 1.0 |
| `test_negative_control_ddim_without_eps_recovers_a_different_value` | the ε term is load-bearing |
| `test_swapping_text_and_constraint_passes_changes_the_result` | 7.0 vs 11.0 — the pass order is pinned numerically, not just structurally |
| `test_negative_control_using_the_gui_minimum_would_change_the_window` | 160-frame batch history vs 4-frame GUI history give different windows |
| `test_negative_control_default_history_would_be_wrong_if_clamped` | `default_history_frames(20,400,4) == -200` is faithful; a `max(0, …)` clamp would be a deviation |
| `test_shifting_the_ground_plane_ignores_the_y_component` | a y translation must be dropped (S2.3) |
| `test_negative_control_heading_not_refreshed_after_crop` | `first_heading_angle` is the crop start's heading, not `[:, 0]` |
| `test_negative_control_noised_history_would_break_the_clean_chain` | corrupting one history window by +0.5 makes the body no longer byte-identical to the clean suffix |
| `test_negative_control_missing_accumulation_would_break_world_recovery` | dropping `global_transl` changes the last frame by >1.0 |
| `test_negative_control_left_aligned_crop_is_detectable` | the crop keeps the **last** `crop_history_length` frames |

## 3. End-to-end results

Placeholder backend (random weights; structure only):

| Run | Result |
|---|---|
| `--backend placeholder --mode fast`, 12 s Core, 240 frames | `frames (1, 240, 330)`, 6 chunks, **0.05 s** |
| `--backend placeholder --mode dense` (contract width 1024), 12 s | `frames (1, 240, 330)`, 6 chunks, **0.05 s** |
| dense loop directly, `history_frames=8` | 80 frames, 2 chunks, 0.034 s |

Real weights (`--backend mlx`, A's checkpoint + `ArdyBatch`):

| Run | Result |
|---|---|
| 12 s Core, 6 windows, seed 0 | `frames (1, 240, 330)`, offsets `[0,40,…,240]`, **1.60 s** |
| 2 s Core, 1 window, reference `x_T` injected | `frames (1, 40, 330)`, **0.37 s** |

**Golden agreement (the one that matters).** With the reference's own `x_T` injected
(`--noise-file artifacts/golden-run/raw/denoise_step0_x_in.npy`), the CLI's
`--backend mlx` output matches the golden clip un-normalised from
`artifacts/golden-run/raw/motion_normalized.npy`:

```
whole row (T=40, D=330)   max_abs = 1.87e-05
  root_pos                 max_abs = 5.14e-06   ref_scale 9.60e-01
  global_root_heading      max_abs = 2.99e-06   ref_scale 1.00e+00
  local_joints_positions   max_abs = 1.27e-06   ref_scale 1.68e+00
  global_rot_data          max_abs = 2.50e-06   ref_scale 1.01e+00
  velocities               max_abs = 1.87e-05   ref_scale 1.59e+00
  foot_contacts            max_abs = 2.44e-06   ref_scale 1.01e+00
```

That is the same order as A's own L3 residue (≤3.9e-05) and well inside the 1e-3
tolerance in `scripts/ardy_align.py`. This is locked by
`tests/test_generate_cli.py::test_cli_mlx_backend_reproduces_the_golden_clip`
(skips when the golden artifacts or the checkpoint are absent).

**4 s / 80 frames / two windows (round 5, no flag needed).** After the round-5
fixes, the *whole* shipped workload is reproduced without `--history-frames` and
with a per-window `--noise-file (2,1,10,148)`:

| Quantity | max_abs | reference \|max\| |
|---|---|---|
| `frames (1,80,330)` vs un-normalised `motion_normalized.npy` | **3.5429992e-05** | 1.682 |
| npz `posed_joints` vs the golden npz's `posed_joints` | **4.8130751e-06** | 1.682 |
| 2 s: `frames (1,40,330)` (single-window noise file, no flag) | **1.8691724e-05** | 1.679 |
| 2 s: npz `posed_joints` vs the golden npz | **5.1781535e-06** | 1.679 |

Both 4 s figures are identical whether `--history-frames` is omitted or passed as
160, and the 2 s figure is bit-identical to round 4 (no regression). Locked by
`test_cli_defaults_and_multi_window_noise_reproduce_the_4s_golden`.

> Caveat, measured and explained: reconstructing the clip from the golden **npz**
> instead (i.e. from `posed_joints` + `global_rot_mats`, the `inverse()` outputs)
> gives 4e-3..2.0e-1 per-block residues. That is reconstruction loss in `inverse()`,
> not a defect in the CLI: the CLI's 6D rotation data, run through the contract's
> Gram–Schmidt rule, rebuilds the golden `global_rot_mats` to **2.2e-06**. Compare
> against `motion_normalized.npy`, never against the inverse-derivable columns.

## 4. Deviations, gaps and open items

### 4.1 Contract ambiguities that B had to resolve (please rule)

1. **`num_autoregressive_steps` − `init_history_len`.** The contract's formula is
   `ceil((num_frames − init_history_len) / horizon)` (`ardy_model.py:569-570`).
   A's batch path passes a **zero-length** initial history, so `init_history_len`
   is the *sliding crop* (160), not a seed block: 240 requested frames → 6 windows
   of 40. B's local loop instead seeds a 160-frame context, so its clip is
   `seed + steps × horizon`. Both produce 240 frames for 12 s, but the meanings of
   `num_frames` differ by the seed block. **B treats `num_frames` as generated
   frames** and only the placeholder loop adds a seed. Ruling wanted if the contract
   ever has to define one number for both.
2. **Step count.** No contract text fixes how `--duration` maps to a step count;
   B uses A's model-side arithmetic. Round 5 removed the second, wrong copy of it
   (`plan_steps_of` subtracted the crop and so under-counted the RNG keys; the
   window count now comes from `compute_feature_layout`, which for the MLX backend
   is exactly `ceil(num_frames / horizon)` = the number of `noise_fn` calls).
3. **`--history-frames` when unset.** Resolved in round 5: unset means the
   **preset's** `default_history_frames` (`resolve_history_frames`), never `None`.
   Passing the preset to the model is what stops the silent loss of cropping, the
   heading refresh and the chunked decode (§6.2). B's placeholder loop still
   rejects a clip shorter than the seed context with an explanatory error rather
   than silently shrinking the seed.

### 4.2 Places where the contract is silent and B did **not** invent a rule

- ~~`posed_joints` (S1.5/S7) needs forward kinematics over the skeleton offsets …
  Measured effect vs the golden npz: `max_abs 0.132`.~~ **Closed in round 5**:
  A's `ardy_mlx.model.skeleton.posed_joints_from_features` is now on the shipped
  path and the sidecar says `skeleton FK (CoreSkeleton27) on global_rot_data +
  root_pos`. See §6.1 for the before/after numbers; the old placeholder function
  survives only as a diagnostic in `sampling/pipeline.py`.
- Root-velocity / contact second-order terms beyond the block layout (S1.4):
  not needed for the golden agreement above.
- `space_timesteps(10, 1) == [0, 9]` and `space_timesteps(10, 2) == [0, 9]`:
  `--diffusion-steps` is a *request*, and the deduped chain can be longer than
  requested. Faithful to `diffusion.py:50-58`; flagged because it is surprising.
- RNG: MLX has no `manual_seed`. The seed becomes an MLX key split once per window,
  and `--noise-file` injects an external `x_T`. Bit-exact agreement with CUDA's
  Philox stream is **not** attempted (A's `model/ardy.py` docstring says the same)
  and remains unverified by construction.

### 4.3 Contract observations worth recording (not B's to change)

- **6D storage is the first two ROWS**: `concat(matrix[...,0], matrix[...,1])`.
  Decisive check: rebuilding the golden `global_rot_mats` from the CLI's 6D data
  with the contract's Gram–Schmidt gives **2.2e-06** (rows) versus **2.6e-02**
  (columns). The contract (S1.4, citing `geometry.py:16-18`) is correct; this entry
  exists so nobody "fixes" it to columns.
- `foot_contacts` in the *generated* feature rows is a continuous regression output
  (not 0/1); the sidecar's `foot_contacts` member is the `> 0.5` binarisation and
  matches the golden boolean array exactly (141/160 true in the 2 s clip).

### 4.4 Waiting on A

Nothing is waiting on A any more, and A is not waiting on B for the three items it
handed over in round 4 — all three are closed in §6.

1. ~~`posed_joints` needs the skeleton offsets to run FK on A's side.~~ Done: A
   exported `posed_joints_from_features` / `core_skeleton().fk`, B wired it.
2. The local sampler loop is **not** the path the golden validates; A's `ArdyBatch`
   is. B's loop exists for the placeholder backend and to pin the window arithmetic
   in isolation. `0004` ruling 5 and the `0005` lead approval already settled this:
   the batch path is standard and B's loop is kept as an independent implementation.
3. **`0005` §4 option (b) is deliberately not implemented.** It was optional ("not
   required"), and the `0005` lead ruling says B's loop is not a golden-judgeable
   path. The `xfail(strict=False)` in `tests/test_model_stepfn.py` therefore stays:
   B's window 0 is "160 zero-seed frames + 40 generated" (50 tokens) where the
   reference is "0 history + 40 generated + 40 masked future" (20 tokens), measured
   **5.466** against golden (golden \|max\| 1.682). Changing that geometry would
   rewrite the placeholder loop's semantics for no release benefit. If it is ever
   wanted, §4 of `0005` already spells out the minimal change.

## 5. How to reproduce

```sh
# placeholder (no weights needed)
./.venv/bin/python scripts/ardy_generate.py \
  --embedding <emb.npy> --duration 12 --seed 0 --mode fast \
  --no-postprocess --out out

# real weights
./.venv/bin/python scripts/ardy_generate.py \
  --embedding <emb.npz> --duration 2 --seed 0 --backend mlx \
  --noise-file artifacts/golden-run/raw/denoise_step0_x_in.npy \
  --no-postprocess --out out

# real weights, the 4 s / 2-window golden (no --history-frames needed any more):
# the noise file is the reference's per-window stack, shape (2, 1, 10, 148)
./.venv/bin/python scripts/ardy_generate.py \
  --embedding <emb.npz> --duration 4 --seed 0 --backend mlx \
  --noise-file <stack of randn_0.npy and randn_1.npy> \
  --no-postprocess --out out
```

## 6. Round 5 (closing) · A's three hand-offs

Closed on top of `3598b3d` with `docs/motion-contract.md` §11 (N14–N16) as the
spec. **No A-side file was touched**; the "before" numbers below come from running
`git archive HEAD` in a scratch tree, i.e. the exact code of `3598b3d`, against the
same golden data and the same invocation.

### 6.1 `posed_joints` is the skeleton FK (task 1)

`scripts/ardy_generate.py::_common_outputs` now calls
`posed_joints_from_features(frames)` and the sidecar records
`extra.posed_joints_source = "skeleton FK (CoreSkeleton27) on global_rot_data +
root_pos"` (one constant, `sampling/pipeline.POSED_JOINTS_SOURCE`, used by both
backends). The old wording and the placeholder call are gone from the CLI.

| 2.0 s / 40 frames | before (`3598b3d`) | after | golden \|max\| |
|---|---|---|---|
| npz `posed_joints` vs golden npz | **1.3222279e-01** | **5.1781535e-06** | 1.679 |
| `frames (1,40,330)` vs golden | 1.8691724e-05 | 1.8691724e-05 (unchanged) | 1.679 |

| 4.0 s / 80 frames | before (`3598b3d`) | after | golden \|max\| |
|---|---|---|---|
| npz `posed_joints` vs golden npz | **2.3628676e-01** | **4.8130751e-06** | 1.682 |

"Before" is what `3598b3d`'s CLI actually wrote into the npz (the placeholder), and
"after" is the FK — the `frames` are *identical* on both sides for the 2 s case, so
the table isolates the joint change. The 4 s `frames` residue is a different story
(§6.2 / §6.3) and is measured separately: its "before" value depends on which of the
two bugs is left in, which is why it is not quoted here.

That is the `0.132 / 0.236` from `0004` ruling 4 down to `5.18e-06 / 4.81e-06`,
identical to A's own FK acceptance values (`5.178e-06 / 4.813e-06`) to every digit
printed here, and five orders of magnitude below the `1.68` signal.

### 6.2 `--history-frames` defaults to the preset value (task 2)

`sampling/pipeline.resolve_history_frames(model, None)` returns
`PRESETS[model].default_history_frames`, and `main()` resolves once and forwards the
number to both backends and to `crop_history_length`. The four presets give four
different values — `160 / 192 / 196 / 240` for Core-H40 / Core-H8 / G1-H52 / G1-H8 —
so nothing is hardcoded to 160 (locked by a test).

| 4.0 s, reference `x_T` injected in-process, `--history-frames` **omitted** | `frames` max_abs |
|---|---|
| before (`3598b3d`; `None` reached the model as `crop_history_length=None`) | **4.422788e-02** |
| after | **3.542999e-05** (= the golden value) |

This is A's `4.42e-02` (44× the 1e-3 tolerance) reproduced exactly, then removed.
Note the sidecar in the "before" run still claimed `init_history_frames: 160` while
the model got `None` — the flag's default and the recorded value had drifted apart.

### 6.3 `--noise-file` is validated, never silently bypassed (task 3)

The old check `injected.shape[1:] == shape[1:]` compared `(1,10,148)` against
`(10,148)`, so a per-window stack never matched and **every** window fell back to
MLX's RNG with no message. `sampling/pipeline.resolve_noise_windows` now accepts
exactly two layouts — `window_shape` (one window, single-window runs only) and
`(n_windows,) + window_shape` — slices per step, and **raises** otherwise. A file
with too few windows also raises; a file with extra windows is accepted with a
printed note (still injected, not a fallback).

| 4.0 s, `--noise-file` = reference stack `(2,1,10,148)` | `frames` max_abs | warning printed |
|---|---|---|
| before (`3598b3d`) | **4.4224728e+00** | none |
| after | **3.5429992e-05** | n/a (injected) |

Negative evidence, all `exit=2` with a single named `error:` line (no traceback,
and no weight is loaded before the check):

```
error: --noise-file holds one window of shape (1, 10, 148) but this run has 2
windows; refusing to inject only window 0 (the rest would silently use MLX's RNG).
Pass a stacked array of shape (2,) + (1, 10, 148).

error: --noise-file has shape (1, 10, 147), expected (1, 10, 148) (one window) or
(1,) + (1, 10, 148) (one entry per autoregressive window).
```

Task 2 and task 3 share one end-to-end lock,
`tests/test_generate_cli.py::test_cli_defaults_and_multi_window_noise_reproduce_the_4s_golden`
(no `--history-frames`, stacked `--noise-file`, asserts `frames < 1e-3` **and**
`posed_joints < 1e-4` against the golden npz). The single-window 2 s path is the
pre-existing `test_cli_mlx_backend_reproduces_the_golden_clip`, unchanged and still
`1.8691724e-05`.

### 6.4 Two honest side effects

* The placeholder loop now receives an explicit `history_frames=160` instead of
  `None`, so from step 1 its context is the cropped 160 frames rather than the
  unbounded seed+trajectory concatenation. Frame count, offsets, npz shapes, the
  heading/contact domain checks and determinism are unchanged (the suite pins
  them); the *values* were always random-weight noise.
* The MLX RNG keys now come from `ceil(num_frames / horizon)` — the real number of
  `noise_fn` calls — instead of the old `plan_steps_of`, which subtracted the crop.
  Only runs **without** `--noise-file` are affected: 12 s now splits 6 keys instead
  of saturating on the second. Nothing is pinned to the MLX stream (it is not the
  CUDA stream; §4.2), and every golden run injects `x_T`.

### 6.5 Still unverified / open (kept as residual unknowns)

1. **`--constraints` is untested**: neither reference run had a non-zero
   `motion_mask` / `observed_motion` / `future_token_mask`, so the constrained path
   is a source-level port only. Unchanged by this round.
2. **B's independent loop cannot be judged against the golden** (`ArdyAutoregressive`,
   **5.466** vs golden \|max\| 1.682). The `xfail(strict=False)` stays by design; §4.4
   item 3 explains why `0005` §4 option (b) was not implemented.
3. **RNG is not aligned with CUDA's Philox** and is not claimed to be; *every*
   golden agreement above injects the reference `x_T`.
4. No golden for real history truncation (needs >200 frames, i.e. `--duration ≥ 11`;
   `0005` ruling 3 declined to buy a third reference run).
5. `--duration → frames` rounding, the MotionBricks/Unity conversion (§2.4) and the
   8B text encoder (v1 eats a precomputed embedding) are unchanged and still open.
6. MLX-side device/precision differences remain a float32-on-Metal vs float32-on-CUDA
   matter, not a port claim.
