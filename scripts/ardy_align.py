#!/usr/bin/env python3
"""Align the MLX port against the reference golden run, and measure it.

Three levels, in increasing scope, all driven from the *same* deterministic
inputs the reference used (the pre-computed text embedding and the reference's own
initial noise ``x_T``):

======  =========================================================  =============
level   what is compared                                        tolerance
======  =========================================================  =============
``L1``  tokenizer ``encode -> decode`` round-trip on the clip    ``1e-3``
``L2``  denoiser single step, all 10 DDIM steps of the window    ``1e-3``
``L3``  the whole 40-frame clip (``frames (T, 330)``)            ``1e-3``
======  =========================================================  =============

**Why 1e-3.** The golden run is float32 on CUDA (cuBLAS/cuDNN kernels whose
reduction orders differ from MLX's), while this port is float32 on Metal/CPU.
Neither side is exact, so the honest question is not "is it bit-identical" but
"is the residue at the level of reordered float32 summation, or is it a real
bug". Empirically the residue here is 1e-7..5e-5, i.e. 4-8 orders of magnitude
below the signal (features are O(1)); a structural mistake -- the kind this
harness exists to catch -- produces O(1e-1) errors. ``--negative-control``
demonstrates exactly that by deliberately corrupting one input and asserting the
harness fails, so the tolerance is falsifiable rather than decorative.

Every comparison also reports the *reference* scale, because a max_abs error is
meaningless without it.

Usage
-----
    ./.venv/bin/python scripts/ardy_align.py                      # align + benchmark
    ./.venv/bin/python scripts/ardy_align.py --levels L1 L2
    ./.venv/bin/python scripts/ardy_align.py --bench-only
    ./.venv/bin/python scripts/ardy_align.py --negative-control
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

from ardy_mlx.model.ardy import ArdyBatch  # noqa: E402
from ardy_mlx.model.denoiser import AutoLatentTwostageDenoiser  # noqa: E402
from ardy_mlx.model.diffusion import Diffusion  # noqa: E402
from ardy_mlx.model.fsq import FSQVAETransformer  # noqa: E402
from ardy_mlx.model.motion_rep import MotionRep  # noqa: E402
from ardy_mlx.model.nn import WeightStore  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"
DEFAULT_RAW = ROOT / "artifacts" / "golden-run" / "raw"
DEFAULT_NPZ = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"

TOLERANCES = {"L1": 1e-3, "L2": 1e-3, "L3": 1e-3}

#: The reference's prompt embedding has length 1; the clip is 40 frames
#: (2.0 s at 20 fps), i.e. exactly one generation horizon.
NUM_FRAMES = 40
NUM_STEPS = 10
CFG_WEIGHT = (2.0, 2.0)
CROP_HISTORY = 160


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


class Harness:
    """Loads the converted checkpoint and the golden dumps; runs the comparisons."""

    def __init__(self, ckpt: Path, raw: Path, npz: Path, device: str = "gpu"):
        self.ckpt = Path(ckpt)
        self.raw = Path(raw)
        self.npz = Path(npz)
        self.device = device
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
            "decode_root": _stats(np.asarray(rt["root"]), self.golden("fsq_detokenize_1_out_root.npy")),
            "decode_body": _stats(np.asarray(rt["body"]), self.golden("fsq_detokenize_1_out_body.npy")),
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
        x_t = self.golden("denoise_step0_x_in.npy")
        if corrupt:
            x_t = x_t + 0.25  # wrong initial noise -> must fail
        out = self.pipe.generate(
            text,
            tmask,
            num_frames=NUM_FRAMES,
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


def _pct(values: List[float], q: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def benchmark(ckpt: Path, device: str, repeats: int = 20) -> Dict:
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

    raw = DEFAULT_RAW
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

    # tokenizer encode
    enc = timed(lambda: tok.tokenize(feats, pmask), repeats)
    dec = timed(lambda: tok.detokenize(latent, local_root, pmask), repeats)
    one_step = timed(lambda: den.forward_batch3(x, **step_kwargs), repeats)
    clip = timed(
        lambda: pipe.generate(
            text, tmask, num_frames=NUM_FRAMES, num_steps=NUM_STEPS, cfg_weight=CFG_WEIGHT,
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


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Align the MLX ARDY port against the golden reference run.")
    ap.add_argument("--ckpt", default=str(DEFAULT_CKPT))
    ap.add_argument("--raw", default=str(DEFAULT_RAW))
    ap.add_argument("--npz", default=str(DEFAULT_NPZ))
    ap.add_argument("--levels", nargs="*", default=["L1", "L2", "L3"])
    ap.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    ap.add_argument("--bench-only", action="store_true")
    ap.add_argument("--no-bench", action="store_true")
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--negative-control", action="store_true",
                    help="deliberately corrupt an input and assert the harness fails")
    ap.add_argument("--json", default=None, help="write the full report to this path")
    args = ap.parse_args(argv)

    report: Dict = {"environment": {"default_device": str(mx.default_device())}}
    rc = 0

    if not args.bench_only:
        h = Harness(Path(args.ckpt), Path(args.raw), Path(args.npz), device=args.device)
        h.load()
        report["mlx_load_seconds"] = h.load_seconds
        report["levels"] = {}
        for level in args.levels:
            tol = TOLERANCES[level]
            if level == "L1":
                res = h.level1()
            elif level == "L2":
                res = h.level2()
            elif level == "L3":
                res = h.level3()
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
        h = Harness(Path(args.ckpt), Path(args.raw), Path(args.npz), device=args.device)
        h.load()
        bad = h.level3(corrupt=True)
        worst = max(v["max_abs"] for v in bad.values())
        ok = worst > TOLERANCES["L3"]
        report["negative_control"] = {"corruption": "x_T += 0.25", "max_abs": worst,
                                      "detected": bool(ok)}
        print(f"--- negative control (x_T += 0.25): max_abs={worst:.3e} detected={ok} ---")
        if not ok:
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
        Path(args.json).write_text(json.dumps(report, indent=2) + "\n")
        print(f"report -> {args.json}")

    print("RESULT: ALIGNED" if rc == 0 else "RESULT: MISALIGNED")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
