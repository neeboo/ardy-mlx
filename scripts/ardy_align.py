#!/usr/bin/env python3
"""Align the MLX port against the two reference golden runs, and measure it.

Four levels, in increasing scope, all driven from the *same* deterministic
inputs the reference used (the pre-computed text embedding and the reference's own
initial noise ``x_T`` -- one draw per autoregressive window):

======  ===================================================================  =============
level   what is compared                                                  tolerance
======  ===================================================================  =============
``L1``  tokenizer ``encode -> decode`` round-trip on the clip               ``1e-3``
``L2``  denoiser single step, all 10 DDIM steps of one window               ``1e-3``
``L3``  the whole 40-frame clip (``frames (T, 330)``), 1 window             ``1e-3``
``L4``  the whole 80-frame clip (``frames (T, 330)``), **2 windows**        ``1e-3``
======  ===================================================================  =============

``L4`` needs the second authorised run (``--duration 4.0``, 80 frames =
``2 x gen_horizon_len``) and is the only level that exercises the multi-window
paths at all:

* **history cropping** -- ``history_start_frame = max(0, history_end_frame -
  crop_history_length)`` and the resulting window token count;
* **per-step recentring + accumulated translation** -- the centre moves from
  frame 39 to frame 79 and ``global_transl`` accumulates over the two steps;
* **``first_heading_angle`` refresh** (``ardy_model.py:665-674``) -- the value
  fed into window 1 comes from the *recentred* root motion and is non-zero on
  this clip (``0.0 -> 0.00261``), so it is genuinely observable here and nowhere
  in the single-window run;
* **chunked autoregressive decode** (``latent_utils.py:103-140``) -- each
  horizon is detokenised separately, unlike the whole-sequence decode that a
  one-window clip cannot distinguish from it.

**Why 1e-3.** The golden run is float32 on CUDA (cuBLAS/cuDNN kernels whose
reduction orders differ from MLX's), while this port is float32 on Metal/CPU.
Neither side is exact, so the honest question is not "is it bit-identical" but
"is the residue at the level of reordered float32 summation, or is it a real
bug". Empirically the residue here is 1e-7..5e-5, i.e. 4-8 orders of magnitude
below the signal (features are O(1)); a structural mistake -- the kind this
harness exists to catch -- produces O(1e-1) errors. ``--negative-control`` and
``--controls`` demonstrate exactly that by deliberately corrupting one input (or
flipping one :class:`~ardy_mlx.model.ardy.Ablation` flag) and asserting the
harness fails, so the tolerance is falsifiable rather than decorative.

Every comparison also reports the *reference* scale, because a max_abs error is
meaningless without it.

Usage
-----
    ./.venv/bin/python scripts/ardy_align.py                      # align + benchmark
    ./.venv/bin/python scripts/ardy_align.py --levels L4
    ./.venv/bin/python scripts/ardy_align.py --controls            # multi-window negative controls
    ./.venv/bin/python scripts/ardy_align.py --negative-control    # single-window negative control
    ./.venv/bin/python scripts/ardy_align.py --bench-only
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import mlx.core as mx  # noqa: E402

from ardy_mlx.model.ardy import (  # noqa: E402
    Ablation,
    ArdyBatch,
    autoregressive_chunk_slices,
    convert_frame_mask_to_token_mask,
    get_three_mask_from_len,
)
from ardy_mlx.model.denoiser import AutoLatentTwostageDenoiser  # noqa: E402
from ardy_mlx.model.diffusion import Diffusion  # noqa: E402
from ardy_mlx.model.fsq import FSQVAETransformer  # noqa: E402
from ardy_mlx.model.motion_rep import MotionRep  # noqa: E402
from ardy_mlx.model.nn import WeightStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"

#: Run 1: the single-window (2.0 s) authorised run.
DEFAULT_RAW = ROOT / "artifacts" / "golden-run" / "raw"
DEFAULT_NPZ = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"

#: Run 2: the two-window (4.0 s) authorised run -- the multi-window oracle.
DEFAULT_RAW4 = ROOT / "artifacts" / "golden-run-4s" / "raw"
DEFAULT_NPZ4 = ROOT / "artifacts" / "golden-run-4s" / "out" / "golden_core_h40_4s_seed0.npz"

TOLERANCES = {"L1": 1e-3, "L2": 1e-3, "L3": 1e-3, "L4": 1e-3}

#: The reference's prompt embedding has length 1.
NUM_STEPS = 10
CFG_WEIGHT = (2.0, 2.0)
CROP_HISTORY = 160
GEN_HORIZON = 40
NUM_FRAMES_PER_TOKEN = 4


def _stats(got: np.ndarray, exp: np.ndarray) -> Dict[str, float]:
    d = np.abs(got.astype(np.float64) - exp.astype(np.float64))
    denom = np.maximum(np.abs(exp.astype(np.float64)), 1e-12)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "max_rel": float((d / denom).max()),
        "reference_abs_max": float(np.abs(exp).max()),
        "reference_abs_mean": float(np.abs(exp).mean()),
        "shape": list(got.shape),
    }


def _denorm(obj):
    """Inverse of the hook's ``_jsonable`` for the small arrays it inlines."""
    if isinstance(obj, dict):
        if "__ndarray__" in obj:
            return np.asarray(obj["__ndarray__"], dtype=np.float32)
        if obj.get("__ndarray_summary__"):
            return None
        return {k: _denorm(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_denorm(v) for v in obj]
    return obj


def read_jsonl(path: Path) -> List[dict]:
    if not path.is_file():
        return []
    out = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if line:
            out.append(_denorm(json.loads(line)))
    return out


def window_frame_masks(
    history_start_frame: int,
    history_end_frame: int,
    total_frames: int,
    generation_len: int = GEN_HORIZON,
    f: int = NUM_FRAMES_PER_TOKEN,
    batch: int = 1,
):
    """The masks one reference window uses (ardy_model.py:440-470 + latent_utils.py:192-220).

    Returned as ``(frame_masks, token_masks)``; this is the pure arithmetic the
    golden dumps let us check directly.
    """
    history_token_end = (history_end_frame - history_start_frame) // f
    history_len = mx.full((batch,), float(history_token_end * f), dtype=mx.float32)
    generation_len_mx = mx.full((batch,), float(generation_len), dtype=mx.float32)
    future_len = mx.full(
        (batch,), float(total_frames - generation_len - history_end_frame), dtype=mx.float32
    )
    total = total_frames - history_start_frame
    frame_masks = get_three_mask_from_len(history_len, generation_len_mx, future_len, total)
    token_masks = convert_frame_mask_to_token_mask(*frame_masks, motion_mask=None, num_frames_per_token=f)
    return frame_masks, token_masks


class Harness:
    """Loads the converted checkpoint and the golden dumps; runs the comparisons."""

    def __init__(self, ckpt: Path, raw: Path, npz: Path, device: str = "gpu", label: str = "run1"):
        self.ckpt = Path(ckpt)
        self.raw = Path(raw)
        self.npz = Path(npz)
        self.device = device
        self.label = label
        self.load_seconds: Dict[str, float] = {}
        mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)

    # ------------------------------------------------------------------ load
    def load(self) -> None:
        t0 = time.perf_counter()
        self.motion_rep = MotionRep(self.ckpt / "stats" / "motion")
        self.tok = FSQVAETransformer(
            WeightStore.load(self.ckpt / "mlx" / "tokenizer.mlx.safetensors"), self.motion_rep
        ).load_stats(self.ckpt / "stats")
        t1 = time.perf_counter()
        self.den = AutoLatentTwostageDenoiser(
            WeightStore.load(self.ckpt / "mlx" / "denoiser.mlx.safetensors"), self.motion_rep
        )
        t2 = time.perf_counter()
        self.pipe = ArdyBatch(self.tok, self.den, self.motion_rep)
        self.load_seconds = {"tokenizer": t1 - t0, "denoiser": t2 - t1, "total": t2 - t0}

    # -------------------------------------------------------------- golden IO
    def golden(self, name: str) -> np.ndarray:
        return np.load(self.raw / name)

    # ------------------------------------------------------------------- L1
    def level1(self) -> Dict[str, Dict[str, float]]:
        feats = mx.array(self.golden("probe_input_features.npy"))
        pmask = mx.array(self.golden("probe_pad_mask.npy").astype(bool))
        rt = self.tok.roundtrip(feats, pmask)
        out = {
            "latent": _stats(np.asarray(rt["latent"]), self.golden("fsq_tokenize_0_out.npy")),
            "decode_root": _stats(np.asarray(rt["root"]), self.golden("fsq_detokenize_2_out_root.npy")
                                  if (self.raw / "fsq_detokenize_2_out_root.npy").is_file()
                                  else self.golden("fsq_detokenize_1_out_root.npy")),
            "decode_body": _stats(np.asarray(rt["body"]), self.golden("fsq_detokenize_2_out_body.npy")
                                  if (self.raw / "fsq_detokenize_2_out_body.npy").is_file()
                                  else self.golden("fsq_detokenize_1_out_body.npy")),
        }
        # the generation-path detokenize (constraint-free root condition)
        mn = mx.array(self.golden("motion_normalized.npy"))
        cond = self.motion_rep.global_root_to_local_root(mn[:, :, :5], normalized=True)
        o0 = self.tok.detokenize(mx.array(self.golden("fsq_detokenize_0_in.npy")), cond)
        out["detok0_root"] = _stats(np.asarray(o0["root"]), self.golden("fsq_detokenize_0_out_root.npy"))
        out["detok0_body"] = _stats(np.asarray(o0["body"]), self.golden("fsq_detokenize_0_out_body.npy"))
        out["requantize"] = _stats(
            np.asarray(self.tok.requantize(mx.array(self.golden("fsq_requantize_0_in.npy")))),
            self.golden("fsq_requantize_0_out.npy"),
        )
        return out

    # ------------------------------------------------------------------- L2
    def level2(self, corrupt_cond: bool = False) -> Dict[str, Dict[str, float]]:
        text = mx.array(self.golden("text_feat.npy"))
        tmask = mx.array(self.golden("text_pad_mask.npy").astype(bool))
        diff = Diffusion(NUM_STEPS)
        results: Dict[str, Dict[str, float]] = {}
        worst = {"raw": 0.0, "cfg": 0.0, "ddim": 0.0}
        per_step = []
        for step in range(NUM_STEPS):
            x = self.golden(f"denoise_step{step}_x_in.npy")
            nt = x.shape[1]
            frames = nt * self.tok.num_frames_per_token
            t = NUM_STEPS - 1 - step
            diff.calc_diffusion_vars(diff.space_timesteps(NUM_STEPS)[0])
            raw = self.den.forward_batch3(
                mx.array(x),
                history_len=mx.array([0.0]),
                generation_len=mx.array([float(frames)]),
                future_len=mx.array([0.0]),
                history_mask=mx.zeros((1, frames)),
                generation_mask=mx.ones((1, frames)),
                future_mask=mx.zeros((1, frames)),
                history_token_mask=mx.zeros((1, nt), dtype=mx.bool_),
                generation_token_mask=mx.ones((1, nt), dtype=mx.bool_),
                future_token_mask=mx.zeros((1, nt), dtype=mx.bool_),
                text_feat=text,
                text_feat_pad_mask=tmask,
                timesteps=mx.array([float(t)]),
                first_heading_angle=mx.array([0.0]),
                motion_mask=None,
                observed_motion=None,
            )
            got_raw = np.asarray(raw)
            exp_raw = self.golden(f"denoiser_raw_{step}.npy")
            if corrupt_cond:
                got_raw = got_raw + 0.05
            o_t, o_c, o_u = mx.split(raw, 3, axis=0)
            cfg = o_u + CFG_WEIGHT[0] * (o_t - o_u) + CFG_WEIGHT[1] * (o_c - o_u)
            got_cfg = np.asarray(cfg)
            exp_cfg = self.golden(f"cfg_combined_{step}.npy")
            x_next = np.asarray(self.pipe.sampler(mx.array(x), cfg, t))
            exp_next = self.golden(f"denoise_step{step}_x_out.npy")
            per_step.append(
                {
                    "step": step,
                    "t": t,
                    "raw_max_abs": float(np.abs(got_raw - exp_raw).max()),
                    "cfg_max_abs": float(np.abs(got_cfg - exp_cfg).max()),
                    "ddim_max_abs": float(np.abs(x_next - exp_next).max()),
                }
            )
            worst["raw"] = max(worst["raw"], float(np.abs(got_raw - exp_raw).max()))
            worst["cfg"] = max(worst["cfg"], float(np.abs(got_cfg - exp_cfg).max()))
            worst["ddim"] = max(worst["ddim"], float(np.abs(x_next - exp_next).max()))
        results["per_step"] = per_step  # type: ignore[assignment]
        results["summary"] = {k: {"max_abs": v} for k, v in worst.items()}  # type: ignore[assignment]
        return results

    # ------------------------------------------------------------------- L3
    def level3(self, corrupt: bool = False) -> Dict[str, Dict[str, float]]:
        text = mx.array(self.golden("text_feat.npy"))
        tmask = mx.array(self.golden("text_pad_mask.npy").astype(bool))
        num_frames = int(self.golden("motion_normalized.npy").shape[1])
        x_t = self.golden("denoise_step0_x_in.npy")
        if corrupt:
            x_t = x_t + 0.25  # wrong initial noise -> must fail
        out = self.pipe.generate(
            text,
            tmask,
            num_frames=num_frames,
            num_steps=NUM_STEPS,
            cfg_weight=CFG_WEIGHT,
            first_heading_angle=mx.array([0.0]),
            crop_history_length=CROP_HISTORY,
            noise_fn=lambda i, shape: mx.array(x_t),
        )
        gold_norm = self.golden("motion_normalized.npy")
        mine_norm = np.asarray(out["motion_normalized"])
        gold_frames = np.asarray(self.motion_rep.unnormalize(mx.array(gold_norm)))
        mine_frames = np.asarray(out["frames"])
        res = {
            "motion_normalized": _stats(mine_norm, gold_norm),
            "frames": _stats(mine_frames, gold_frames),
        }
        if self.npz.is_file():
            z = np.load(self.npz)
            res["root_positions_vs_npz"] = _stats(mine_frames[0, :, :3], z["root_positions"])
            res["heading_vs_npz"] = _stats(mine_frames[0, :, 3:5], z["global_root_heading"])
        return res

    # ------------------------------------------------------------------- L4
    def level4(
        self,
        ablation: Ablation = Ablation(),
        crop_history_length: Optional[int] = CROP_HISTORY,
        noise_offset: float = 0.0,
        measure_only: bool = False,
    ) -> Dict:
        """Multi-window alignment: 80 frames, two autoregressive windows.

        Uses the **same embedding and the same injected noise** as the reference
        (one ``x_T`` draw per window, taken from the reference's own dumps), and
        reports per-window numbers rather than a single worst case.

        ``measure_only`` skips the per-window / per-step golden comparisons and
        returns only the end-to-end ``frames`` residue. The negative controls need
        it because a control that deliberately changes the window geometry (a
        truncated crop) produces a different token count, so an elementwise
        comparison against the reference's window would compare apples to oranges.
        """
        text = mx.array(self.golden("text_feat.npy"))
        tmask = mx.array(self.golden("text_pad_mask.npy").astype(bool))
        gold_norm = self.golden("motion_normalized.npy")
        num_frames = int(gold_norm.shape[1])
        n_windows = len(
            [r for r in read_jsonl(self.raw / "windows.jsonl") if "history_start_frame" in r]
        ) or (num_frames // GEN_HORIZON)
        x_T = [self.golden(f"randn_{w}.npy") + noise_offset for w in range(n_windows)]

        trace: Dict = {}
        out = self.pipe.generate(
            text,
            tmask,
            num_frames=num_frames,
            num_steps=NUM_STEPS,
            cfg_weight=CFG_WEIGHT,
            first_heading_angle=mx.array([0.0]),
            crop_history_length=crop_history_length,
            noise_fn=lambda w, shape: mx.array(x_T[w]),
            ablation=ablation,
            trace=trace,
        )

        ref_windows = [r for r in read_jsonl(self.raw / "windows.jsonl") if "history_start_frame" in r]
        ref_recenters = read_jsonl(self.raw / "recenters.jsonl")
        ref_meta = {r["window"]: r for r in ref_windows}

        report: Dict = {"num_windows": n_windows, "num_frames": num_frames}

        # ---- per-window geometry + mask assertions -------------------------
        per_window = []
        for w in range(n_windows):
            mine = trace["windows"][w]
            ref = ref_meta.get(w, {})
            ref_x0 = self.golden(f"denoise_step{w * NUM_STEPS}_x_in.npy")
            geom = {
                "window": w,
                "history_start_frame": mine["history_start_frame"],
                "history_start_frame_ref": ref.get("history_start_frame"),
                "history_end_frame": mine["history_end_frame"],
                "history_end_frame_ref": ref.get("history_end_frame"),
                "history_token_end": mine["history_token_end"],
                "history_token_end_ref": ref.get("history_token_end"),
                "total_tokens": mine["total_tokens"],
                "total_tokens_ref": int(ref_x0.shape[1]),
                "future_len": mine["future_len"],
                "future_len_ref": ref.get("future_len"),
                "token_future_mask_sum": mine["token_future_mask_sum"],
                "frame_future_frames": mine["frame_future_frames"],
                "geometry_matches_reference": (
                    mine["history_start_frame"] == ref.get("history_start_frame")
                    and mine["history_end_frame"] == ref.get("history_end_frame")
                    and mine["history_token_end"] == ref.get("history_token_end")
                    and mine["total_tokens"] == int(ref_x0.shape[1])
                    and mine["future_len"] == ref.get("future_len")
                ),
            }
            # token masks, elementwise, against the reference's own dump
            tm_ref_path = self.raw / f"token_masks_{w}.npz"
            if tm_ref_path.is_file():
                tm = np.load(tm_ref_path)
                _, (h_tok, g_tok, f_tok) = window_frame_masks(
                    mine["history_start_frame"], mine["history_end_frame"], num_frames
                )
                geom["token_history_mask_equal"] = bool(
                    np.array_equal(np.asarray(h_tok).astype(bool), tm["history"].astype(bool))
                )
                geom["token_generation_mask_equal"] = bool(
                    np.array_equal(np.asarray(g_tok).astype(bool), tm["generation"].astype(bool))
                )
                geom["token_future_mask_equal"] = bool(
                    np.array_equal(np.asarray(f_tok).astype(bool), tm["future"].astype(bool))
                )
                geom["token_future_mask_sum_ref"] = float(tm["future"].sum())
            # the window's starting latent, i.e. history + fresh noise.
            # A control that deliberately changes the window geometry produces a
            # different token count; that is itself the signal, not a crash.
            if mine["x0"].shape == ref_x0.shape:
                geom["x0"] = _stats(mine["x0"], ref_x0)
            else:
                geom["x0"] = {
                    "max_abs": float("inf"),
                    "mean_abs": float("inf"),
                    "max_rel": float("inf"),
                    "reference_abs_max": float(np.abs(ref_x0).max()),
                    "reference_abs_mean": float(np.abs(ref_x0).mean()),
                    "shape": list(mine["x0"].shape),
                    "shape_mismatch": list(ref_x0.shape),
                }
            # first_heading_angle fed into this window
            fh_ref_path = self.raw / f"window{w}_first_heading_in.npy"
            if fh_ref_path.is_file():
                geom["first_heading_angle_in"] = _stats(
                    mine["first_heading_angle_in"], np.load(fh_ref_path)
                )
            gt_ref_path = self.raw / f"window{w}_global_transl_in.npy"
            if gt_ref_path.is_file():
                geom["global_transl_in"] = _stats(mine["global_transl_in"], np.load(gt_ref_path))
            # recentring
            if w < len(ref_recenters):
                rec = ref_recenters[w]
                geom["center_frame_index"] = mine.get("center_frame_index")
                geom["center_frame_index_ref"] = int(np.asarray(rec["center_frame_index"]).reshape(-1)[0])
                geom["center_pos"] = _stats(mine["center_pos"], np.asarray(rec["center_pos"]))
                rr_path = self.raw / f"recenter{w}_history_recentered.npy"
                if rr_path.is_file():
                    geom["recentered_hybrid"] = _stats(mine["history_after_recenter"], np.load(rr_path))
            # the refreshed heading is what the next window receives
            nxt = self.raw / f"window{w + 1}_first_heading_in.npy"
            if nxt.is_file():
                geom["first_heading_angle_after"] = _stats(
                    mine["first_heading_angle_after"], np.load(nxt)
                )
            ng = self.raw / f"window{w + 1}_global_transl_in.npy"
            if ng.is_file():
                geom["global_transl_after"] = _stats(mine["global_transl_after"], np.load(ng))
            per_window.append(geom)

        report["per_window"] = per_window

        # ---- per-step denoiser alignment, all 20 steps ----------------------
        per_step = []
        worst = {"raw": 0.0, "cfg": 0.0, "ddim": 0.0}
        for step in range(n_windows * NUM_STEPS):
            w = step // NUM_STEPS
            x = self.golden(f"denoise_step{step}_x_in.npy")
            nt = x.shape[1]
            frames = nt * NUM_FRAMES_PER_TOKEN
            fh = self.golden(f"window{w}_first_heading_in.npy")
            # Masks come from the **reference's** window geometry, not the port's, so
            # this loop isolates the denoiser + mask arithmetic from whatever the
            # window planner did. Geometry itself is asserted in ``per_window``.
            (hm, gm, fm), (ht, gt, ft) = window_frame_masks(
                ref_meta[w]["history_start_frame"],
                ref_meta[w]["history_end_frame"],
                num_frames,
                batch=x.shape[0],
            )
            # the mask total must equal this step's token count
            assert hm.shape[1] == frames, (hm.shape, frames)
            t = NUM_STEPS - 1 - (step % NUM_STEPS)
            self.pipe.diffusion.calc_diffusion_vars(self.pipe.diffusion.space_timesteps(NUM_STEPS)[0])
            raw = self.den.forward_batch3(
                mx.array(x),
                history_len=mx.full((x.shape[0],), float(np.asarray(hm).sum()), dtype=mx.float32),
                generation_len=mx.full((x.shape[0],), float(GEN_HORIZON), dtype=mx.float32),
                future_len=mx.full((x.shape[0],), float(ref_meta[w]["future_len"]), dtype=mx.float32),
                history_mask=hm,
                generation_mask=gm,
                future_mask=fm,
                history_token_mask=ht,
                generation_token_mask=gt,
                future_token_mask=ft,
                text_feat=text,
                text_feat_pad_mask=tmask,
                timesteps=mx.full((x.shape[0],), float(t)),
                first_heading_angle=mx.array(fh),
                motion_mask=None,
                observed_motion=None,
            )
            got_raw = np.asarray(raw)
            exp_raw = self.golden(f"denoiser_raw_{step}.npy")
            o_t, o_c, o_u = mx.split(raw, 3, axis=0)
            cfg = o_u + CFG_WEIGHT[0] * (o_t - o_u) + CFG_WEIGHT[1] * (o_c - o_u)
            # DDIM updates **generation tokens only** (ardy_model.py:302-313); the
            # history tokens of window >= 1 are carried through untouched, so the
            # comparison must be restricted to the generation span.
            hte = ref_meta[w]["history_token_end"]
            gte = hte + GEN_HORIZON // NUM_FRAMES_PER_TOKEN
            x_next = np.asarray(self.pipe.sampler(mx.array(x[:, hte:gte]), cfg[:, hte:gte], t))
            exp_next = self.golden(f"denoise_step{step}_x_out.npy")[:, hte:gte]
            history_untouched = bool(
                hte == 0
                or np.array_equal(
                    np.asarray(self.golden(f"denoise_step{step}_x_out.npy"))[:, :hte],
                    x[:, :hte],
                )
            )
            row = {
                "step": step,
                "window": w,
                "t": t,
                "tokens": nt,
                "raw_max_abs": float(np.abs(got_raw - exp_raw).max()),
                "cfg_max_abs": float(np.abs(np.asarray(cfg) - self.golden(f"cfg_combined_{step}.npy")).max()),
                "ddim_max_abs": float(np.abs(x_next - exp_next).max()),
                "reference_abs_max": float(np.abs(exp_raw).max()),
                "history_tokens_untouched_in_reference": history_untouched,
            }
            per_step.append(row)
            for k, v in (("raw", "raw_max_abs"), ("cfg", "cfg_max_abs"), ("ddim", "ddim_max_abs")):
                worst[k] = max(worst[k], row[v])
        report["per_step"] = per_step
        report["per_step_summary"] = {k: {"max_abs": v} for k, v in worst.items()}

        # ---- final clip: whole + per-window chunks --------------------------
        gold_frames = np.asarray(self.motion_rep.unnormalize(mx.array(gold_norm)))
        mine_frames = np.asarray(out["frames"])
        report["final"] = {
            "motion_normalized": _stats(np.asarray(out["motion_normalized"]), gold_norm),
            "frames": _stats(mine_frames, gold_frames),
        }
        chunks = []
        for w in range(n_windows):
            a, b = w * GEN_HORIZON, (w + 1) * GEN_HORIZON
            chunks.append(
                {
                    "window": w,
                    "frames": _stats(mine_frames[:, a:b], gold_frames[:, a:b]),
                }
            )
        report["chunks"] = chunks
        if self.npz.is_file():
            z = np.load(self.npz)
            report["final"]["root_positions_vs_npz"] = _stats(mine_frames[0, :, :3], z["root_positions"])
            report["final"]["heading_vs_npz"] = _stats(mine_frames[0, :, 3:5], z["global_root_heading"])
            for w in range(n_windows):
                a, b = w * GEN_HORIZON, (w + 1) * GEN_HORIZON
                chunks[w]["root_positions_vs_npz"] = _stats(
                    mine_frames[0, a:b, :3], z["root_positions"][a:b]
                )
                chunks[w]["heading_vs_npz"] = _stats(
                    mine_frames[0, a:b, 3:5], z["global_root_heading"][a:b]
                )

        # ---- accumulated translation / final hybrid -------------------------
        fh_path = self.raw / "final_hybrid_translated.npy"
        if fh_path.is_file():
            report["final"]["hybrid_translated"] = _stats(
                trace["final"]["hybrid_translated"], np.load(fh_path)
            )
        report["final"]["global_transl"] = np.asarray(trace["final"]["global_transl"]).tolist()
        report["final"]["decode_chunks"] = list(trace["final"]["decode_chunks"])

        # ---- chunked vs whole decode ---------------------------------------
        # The reference decodes chunk-by-chunk (latent_utils.py:103-140). Because
        # the Stage-A decoder is causal, decoding the whole sequence at once must
        # give the same frames up to float32 reduction order; measuring that is
        # what tells us the chunk boundary is not silently dropping context.
        root_motion, latent = self.den.split_hybrid(mx.array(trace["final"]["hybrid_translated"]))
        chunked = np.asarray(
            self.pipe.explicit_from_hybrid_autoregressive(latent, root_motion, crop_history_length)
        )
        whole = np.asarray(self.pipe.explicit_from_hybrid_whole(latent, root_motion))
        report["decode_chunked_vs_whole"] = {
            "comparison": _stats(chunked, whole),
            "chunk_plan": [list(c) for c in autoregressive_chunk_slices(
                latent.shape[1], GEN_HORIZON // NUM_FRAMES_PER_TOKEN,
                None if crop_history_length is None else crop_history_length // NUM_FRAMES_PER_TOKEN)],
        }

        # ---- overall worst residue ------------------------------------------
        report["summary"] = {
            "tolerance": TOLERANCES["L4"],
            "x0_worst": max(g["x0"]["max_abs"] for g in per_window),
            "step_worst": worst,
            "frames_max_abs": report["final"]["frames"]["max_abs"],
            "crop_history_length": crop_history_length,
            "ablation": ablation.__dict__,
            "noise_offset": noise_offset,
            "measure_only": measure_only,
        }
        return report

    # ------------------------------------------------- multi-window controls
    def negative_controls_multi(self) -> List[Dict]:
        """Deliberately break one thing at a time and record whether it shows.

        The signal is the end-to-end ``frames (T, 330)`` residue against the golden
        clip -- measured in ``measure_only`` mode, because a control that changes
        the window geometry does not have a like-for-like window to compare.

        ``expect_detected=False`` rows are controls that *cannot* fire on this
        golden clip, and the reason is part of the result:

        * the "late centre" control clamps back to the reference centre, because
          ``history_end_frame + generation_len - 1`` is already the last frame of
          the sequence (ardy_model.py:655-656);
        * ``crop_history_length = 164`` (one patch more than 160) changes nothing,
          because an 80-frame clip never reaches the 160-frame crop budget -- the
          truncation branch is simply not taken. Any "off by one frame in the crop"
          negative control is therefore vacuous at ``--duration 4.0``; the
          truncating control below uses ``crop = 4`` instead.
        """
        cases = [
            ("x_T += 0.25 (wrong initial noise)", dict(noise_offset=0.25), True),
            ("recentre centre one token early (-4 frames)",
             dict(ablation=Ablation(center_index_offset=-4)), True),
            ("recentre centre two tokens early (-8 frames)",
             dict(ablation=Ablation(center_index_offset=-8)), True),
            ("recentre centre one token late (+4 frames; clamps to the reference centre)",
             dict(ablation=Ablation(center_index_offset=+4)), False),
            ("stale first_heading_angle (refresh disabled)",
             dict(ablation=Ablation(refresh_first_heading=False)), True),
            ("future_token_mask derived from future_mask",
             dict(ablation=Ablation(future_token_mask_from_frames=True)), True),
            ("crop_history_length = 4 (truncates the history to one token)",
             dict(crop_history_length=4), True),
            ("crop_history_length = 164 (one patch more; no truncation at 80 frames)",
             dict(crop_history_length=164), False),
            ("crop_history_length = None (no cropping at all)",
             dict(crop_history_length=None), True),
            ("whole-sequence decode instead of the chunked decode",
             dict(ablation=Ablation(autoregressive_decode=False)), True),
        ]
        rows = [
            {
                "control": "clean baseline (identical to the port's default behaviour)",
                "worst_max_abs": self.level4(measure_only=True)["summary"]["frames_max_abs"],
                "tolerance": TOLERANCES["L4"],
                "detected": False,
                "expect_detected": False,
                "matches_expectation": True,
                "kwargs": {},
            }
        ]
        for name, kwargs, expect in cases:
            worst = self.level4(measure_only=True, **kwargs)["summary"]["frames_max_abs"]
            detected = bool(worst > TOLERANCES["L4"])
            rows.append(
                {
                    "control": name,
                    "worst_max_abs": worst,
                    "tolerance": TOLERANCES["L4"],
                    "detected": detected,
                    "expect_detected": bool(expect),
                    "matches_expectation": bool(detected == expect),
                    "kwargs": {k: (v.__dict__ if isinstance(v, Ablation) else v) for k, v in kwargs.items()},
                }
            )
        return rows


def _pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def benchmark(ckpt: Path, device: str, repeats: int = 20, raw: Path = DEFAULT_RAW) -> Dict:
    """Time the three hot paths on one device, with cold start and peak memory."""
    mx.set_default_device(mx.gpu if device == "gpu" else mx.cpu)
    mx.reset_peak_memory()
    result: Dict = {"device": device, "repeats": repeats}
    result["loadavg"] = list(os.getloadavg())
    result["machine"] = f"{platform.machine()} / {platform.platform()}"

    t0 = time.perf_counter()
    motion_rep = MotionRep(ckpt / "stats" / "motion")
    tok = FSQVAETransformer(
        WeightStore.load(ckpt / "mlx" / "tokenizer.mlx.safetensors"), motion_rep
    ).load_stats(ckpt / "stats")
    t1 = time.perf_counter()
    den = AutoLatentTwostageDenoiser(
        WeightStore.load(ckpt / "mlx" / "denoiser.mlx.safetensors"), motion_rep
    )
    pipe = ArdyBatch(tok, den, motion_rep)
    t2 = time.perf_counter()
    result["cold_start_s"] = {"tokenizer_load": t1 - t0, "denoiser_load": t2 - t1, "total": t2 - t0}

    feats = mx.array(np.load(raw / "probe_input_features.npy"))
    pmask = mx.array(np.load(raw / "probe_pad_mask.npy").astype(bool))
    text = mx.array(np.load(raw / "text_feat.npy"))
    tmask = mx.array(np.load(raw / "text_pad_mask.npy").astype(bool))
    latent = tok.tokenize(feats, pmask)
    local_root = motion_rep.global_root_to_local_root(motion_rep.extract_root(feats), normalized=True)
    mx.eval(latent, local_root)  # materialise: MLX is lazy, and detokenize re-reads these
    x = mx.array(np.load(raw / "denoise_step0_x_in.npy"))
    mx.eval(x)
    nt = x.shape[1]
    frames = nt * tok.num_frames_per_token
    diff = Diffusion(NUM_STEPS)
    diff.calc_diffusion_vars(diff.space_timesteps(NUM_STEPS)[0])
    step_kwargs = dict(
        history_len=mx.array([0.0]), generation_len=mx.array([float(frames)]),
        future_len=mx.array([0.0]), history_mask=mx.zeros((1, frames)),
        generation_mask=mx.ones((1, frames)), future_mask=mx.zeros((1, frames)),
        history_token_mask=mx.zeros((1, nt), dtype=mx.bool_),
        generation_token_mask=mx.ones((1, nt), dtype=mx.bool_),
        future_token_mask=mx.zeros((1, nt), dtype=mx.bool_),
        text_feat=text, text_feat_pad_mask=tmask,
        timesteps=mx.array([9.0]), first_heading_angle=mx.array([0.0]),
        motion_mask=None, observed_motion=None,
    )
    x_t = np.load(raw / "denoise_step0_x_in.npy")

    def timed(fn, n):
        """Time ``fn`` with MLX's lazy graph forced inside the timed region."""
        def run():
            out = fn()
            if isinstance(out, dict):
                arrays = [v for v in out.values() if isinstance(v, mx.array)]
            elif isinstance(out, (list, tuple)):
                arrays = [v for v in out if isinstance(v, mx.array)]
            else:
                arrays = [out]
            mx.eval(*arrays)

        run()  # warm-up: pays first-call graph construction and kernel compilation
        times = []
        for _ in range(n):
            start = time.perf_counter()
            run()
            times.append(time.perf_counter() - start)
        return times

    enc = timed(lambda: tok.tokenize(feats, pmask), repeats)
    dec = timed(lambda: tok.detokenize(latent, local_root, pmask), repeats)
    one_step = timed(lambda: den.forward_batch3(x, **step_kwargs), repeats)
    clip = timed(
        lambda: pipe.generate(
            text, tmask, num_frames=frames, num_steps=NUM_STEPS, cfg_weight=CFG_WEIGHT,
            first_heading_angle=mx.array([0.0]), crop_history_length=CROP_HISTORY,
            noise_fn=lambda i, shape: mx.array(x_t),
        ),
        max(3, repeats // 4),
    )

    def summarize(times):
        return {
            "p50_s": statistics.median(times),
            "p90_s": _pct(times, 0.9),
            "min_s": min(times),
            "max_s": max(times),
            "n": len(times),
        }

    result["timings"] = {
        "tokenizer_encode": summarize(enc),
        "tokenizer_decode": summarize(dec),
        "denoiser_step_batch3": summarize(one_step),
        "full_clip_40f_10steps": summarize(clip),
    }
    result["peak_memory_bytes"] = int(mx.get_peak_memory())
    result["active_memory_bytes"] = int(mx.get_active_memory())
    return result


def _print_level4(res: Dict, tol: float) -> List[str]:
    fails: List[str] = []
    print(f"  window  history[frames]  tokens  future  x0_max_abs  ref|x0|max  geo_ok")
    for g in res["per_window"]:
        ok = "OK " if g["x0"]["max_abs"] <= tol else "FAIL"
        if g["x0"]["max_abs"] > tol or not g.get("geometry_matches_reference", False):
            fails.append(f"window{g['window']}")
        print(
            f"  {ok} w{g['window']}  [{g['history_start_frame']},{g['history_end_frame']})"
            f"        {g['total_tokens']:>3}    {g['future_len']:>3}   "
            f"{g['x0']['max_abs']:.3e}   {g['x0']['reference_abs_max']:.3f}    "
            f"{g.get('geometry_matches_reference')}"
        )
        for key in ("first_heading_angle_in", "first_heading_angle_after", "global_transl_in",
                    "global_transl_after", "center_pos", "recentered_hybrid"):
            st = g.get(key)
            if isinstance(st, dict) and "max_abs" in st:
                flag = "OK " if st["max_abs"] <= tol else "FAIL"
                if st["max_abs"] > tol:
                    fails.append(f"w{g['window']}:{key}")
                print(f"       {flag} {key:24s} max_abs={st['max_abs']:.3e} "
                      f"ref|max|={st['reference_abs_max']:.3f}")
        for key in ("token_history_mask_equal", "token_generation_mask_equal", "token_future_mask_equal"):
            if key in g and not g[key]:
                fails.append(f"w{g['window']}:{key}")
                print(f"       FAIL {key}")
    s = res["per_step_summary"]
    print(f"  per-step worst over {len(res['per_step'])} steps: raw={s['raw']['max_abs']:.3e} "
          f"cfg={s['cfg']['max_abs']:.3e} ddim={s['ddim']['max_abs']:.3e}")
    for k in ("raw", "cfg", "ddim"):
        if s[k]["max_abs"] > tol:
            fails.append(f"per_step:{k}")
    for row in res["per_step"]:
        if row["cfg_max_abs"] > tol or row["ddim_max_abs"] > tol or row["raw_max_abs"] > tol:
            print(f"       FAIL step {row['step']} (w{row['window']}) raw={row['raw_max_abs']:.3e} "
                  f"cfg={row['cfg_max_abs']:.3e} ddim={row['ddim_max_abs']:.3e}")
    for name, st in res["final"].items():
        if not isinstance(st, dict) or "max_abs" not in st:
            continue
        flag = "OK " if st["max_abs"] <= tol else "FAIL"
        if st["max_abs"] > tol:
            fails.append(f"final:{name}")
        print(f"  {flag} {name:26s} max_abs={st['max_abs']:.3e} ref|max|={st['reference_abs_max']:.3f}")
    for c in res["chunks"]:
        for key in ("frames", "root_positions_vs_npz", "heading_vs_npz"):
            st = c.get(key)
            if not isinstance(st, dict):
                continue
            flag = "OK " if st["max_abs"] <= tol else "FAIL"
            if st["max_abs"] > tol:
                fails.append(f"chunk{c['window']}:{key}")
            print(f"  {flag} chunk{c['window']} {key:24s} max_abs={st['max_abs']:.3e} "
                  f"ref|max|={st['reference_abs_max']:.3f}")
    return fails


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Align the MLX ARDY port against the golden reference runs.")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--raw", default=str(DEFAULT_RAW), help="run-1 dumps (2.0 s, one window)")
    ap.add_argument("--npz", default=str(DEFAULT_NPZ), help="run-1 npz output")
    ap.add_argument("--raw4", default=str(DEFAULT_RAW4), help="run-2 dumps (4.0 s, two windows)")
    ap.add_argument("--npz4", default=str(DEFAULT_NPZ4), help="run-2 npz output")
    ap.add_argument("--levels", nargs="*", default=["L1", "L2", "L3", "L4"])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--negative-control", action="store_true",
                    help="deliberately corrupt an input and assert the harness fails (single window)")
    ap.add_argument("--controls", action="store_true",
                    help="run every multi-window negative control and assert each one is detected")
    ap.add_argument("--json", default=None, help="write the full report to this path")
    args = ap.parse_args(argv)

    report: Dict = {"environment": {"default_device": str(mx.default_device())}}
    rc = 0

    if not args.bench_only:
        report["levels"] = {}
        single = [lv for lv in args.levels if lv in ("L1", "L2", "L3")]
        h = None
        if single:
            h = Harness(Path(args.ckpt), Path(args.raw), Path(args.npz), device=args.device, label="run1")
            h.load()
            report["mlx_load_seconds"] = h.load_seconds
        for level in args.levels:
            tol = TOLERANCES[level]
            if level in ("L1", "L2", "L3"):
                assert h is not None
                res = {"L1": h.level1, "L2": h.level2, "L3": h.level3}[level]()
            elif level == "L4":
                if not (Path(args.raw4) / "randn_0.npy").is_file():
                    print(f"  L4 skipped: no multi-window golden dumps in {args.raw4}", file=sys.stderr)
                    continue
                h4 = Harness(Path(args.ckpt), Path(args.raw4), Path(args.npz4),
                             device=args.device, label="run2")
                h4.load()
                res = h4.level4()
            else:
                print(f"unknown level {level}", file=sys.stderr)
                return 2
            report["levels"][level] = {"tolerance": tol, "comparisons": res}
            print(f"--- {level} (tolerance {tol:g}) ---")
            if level == "L2":
                print(f"  worst over {NUM_STEPS} steps: raw={res['summary']['raw']['max_abs']:.3e} "
                      f"cfg={res['summary']['cfg']['max_abs']:.3e} "
                      f"ddim={res['summary']['ddim']['max_abs']:.3e}")
                fails = [k for k, v in res["summary"].items() if v["max_abs"] > tol]
            elif level == "L4":
                fails = _print_level4(res, tol)
            else:
                fails = []
                for name, st in res.items():
                    flag = "OK " if st["max_abs"] <= tol else "FAIL"
                    print(f"  {flag} {name:22s} max_abs={st['max_abs']:.3e} mean={st['mean_abs']:.3e} "
                          f"ref|max|={st['reference_abs_max']:.3f}")
                    if st["max_abs"] > tol:
                        fails.append(name)
            if fails:
                rc = 1
                print(f"  => {level} FAILED on: {', '.join(fails)}")

    if args.negative_control:
        h = Harness(Path(args.ckpt), Path(args.raw), Path(args.npz), device=args.device, label="run1")
        h.load()
        bad = h.level3(corrupt=True)
        worst = max(v["max_abs"] for v in bad.values())
        ok = worst > TOLERANCES["L3"]
        report["negative_control"] = {"corruption": "x_T += 0.25", "max_abs": worst,
                                      "detected": bool(ok)}
        print(f"--- negative control (x_T += 0.25): max_abs={worst:.3e} detected={ok} ---")
        if not ok:
            rc = 1

    if args.controls:
        h4 = Harness(Path(args.ckpt), Path(args.raw4), Path(args.npz4),
                     device=args.device, label="run2")
        h4.load()
        rows = h4.negative_controls_multi()
        report["negative_controls_multi"] = rows
        print(f"--- multi-window negative controls (tolerance {TOLERANCES['L4']:g}) ---")
        for row in rows:
            flag = "detected" if row["detected"] else "not detected"
            mark = "ok " if row["matches_expectation"] else "BAD"
            print(f"  {mark} {flag:13s} frames_max_abs={row['worst_max_abs']:.3e}  {row['control']}")
            if not row["matches_expectation"]:
                rc = 1

    if not args.no_bench:
        report["benchmarks"] = []
        for device in ("cpu", "gpu"):
            if device == "gpu" and not mx.metal.is_available():
                continue
            print(f"--- benchmark on {device} ---")
            try:
                bench = benchmark(Path(args.ckpt), device, repeats=args.repeats)
            except Exception as exc:  # pragma: no cover - device dependent
                print(f"  benchmark on {device} failed: {type(exc).__name__}: {exc}", file=sys.stderr)
                report["benchmarks"].append({"device": device, "error": f"{type(exc).__name__}: {exc}"})
                continue
            report["benchmarks"].append(bench)
            print(f"  cold start: {bench['cold_start_s']['total']:.3f}s   peak mem: "
                  f"{bench['peak_memory_bytes'] / 1e6:.1f} MB   loadavg: {bench['loadavg']}")
            for name, st in bench["timings"].items():
                print(f"  {name:24s} p50={st['p50_s'] * 1e3:8.2f} ms  p90={st['p90_s'] * 1e3:8.2f} ms  n={st['n']}")

    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(report, indent=2, default=str) + "\n")
        print(f"report -> {args.json}")

    print("RESULT: ALIGNED" if rc == 0 else "RESULT: MISALIGNED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
