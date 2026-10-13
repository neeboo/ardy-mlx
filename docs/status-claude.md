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
| `src/ardy_mlx/sampling/pipeline.py` | embedding loading, output assembly, `.npz` + sidecar (S7) |
| `scripts/ardy_generate.py` | CLI; `--backend placeholder` (local loop) and `--backend mlx` (A's model) |

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
./.venv/bin/python -m pytest tests/ -q
155 passed in 3.08s
```

Breakdown (all green): schedule 16, cfg 8, window 24, loop 23, pipeline 16, CLI 12,
plus A's existing io tests. (`test_sampling_schedule` etc. are B's; `test_io_*` are A's.)

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
2. **`plan_steps_of` / step count.** No contract text fixes how `--duration` maps to
   a step count; B uses A's model-side arithmetic.
3. **`--history-frames` seed length.** Uncovered: the contract fixes the default
   (160) and the patch-multiple rule, but not what `init_history_len` is when the
   clip is shorter than the default. B's placeholder CLI rejects that combination
   with an explanatory error rather than silently shrinking the seed.

### 4.2 Places where the contract is silent and B did **not** invent a rule

- `posed_joints` (S1.5/S7) needs forward kinematics over the skeleton offsets. The
  skeleton definition is on A's side, so B writes the root-local joint columns with
  the root inserted and records `extra.posed_joints_source` in the sidecar. The
  model backend's `posed_joints` is therefore **not** the contract's world joints.
  Measured effect vs the golden npz: `max_abs 0.132`.
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

Nothing blocks the golden path any more. Two integration items remain:

1. `posed_joints` needs the skeleton offsets to run FK on A's side (or a B-side
   skeleton definition). Until then the artifact carries the placeholder columns.
2. The local sampler loop is **not** the path the golden validates; A's `ArdyBatch`
   is. B's loop exists for the placeholder backend and to pin the window arithmetic
   in isolation. If A's loop is the intended single implementation, say so and B's
   loop can be dropped — right now they agree on frame counts and both are tested.

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
```
