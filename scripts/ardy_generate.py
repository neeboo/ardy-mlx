#!/usr/bin/env python
"""ardy-mlx generation CLI (B side) -- contract ``docs/motion-contract.md`` S7.

Batch path only, mirroring the reference ``scripts/generate.py`` argument surface:
precomputed text embedding in, one whole motion clip out.

    ./.venv/bin/python scripts/ardy_generate.py \
        --embedding path/to/embedding.npy \
        --model ARDY-Core-RP-20FPS-Horizon40 \
        --duration 2.0 --seed 0 --out out

Defaults follow the contract: ``--diffusion-steps`` falls back to
``num_base_steps`` (10), ``--cfg-weight`` defaults to ``[2.0, 2.0]``, and
``--history-frames`` defaults to the **batch** history (the *maximum*), 160 frames
for Core/20fps/H40 -- not the GUI slider's minimum of one patch (S5.2).

Post-processing: contract S6.4 and decision 0002 ruling 2 pin the v1 policy at
"**both sides skip post-processing**". The reference Core path *does* run
``MotionCorrection`` by default, but that C++ extension is not built here, so:

* ``--no-postprocess`` -> builds fine, ``postprocess: false``;
* without it, on Core -> the script exits with an explanatory error rather than
  silently producing output whose sidecar claims a different policy;
* G1 -> always skipped, exactly like the reference (``generate.py:265-275``).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ardy_mlx.io import (  # noqa: E402
    DEFAULT_CFG_WEIGHT,
    NUM_BASE_STEPS,
    NUM_FRAMES_PER_TOKEN,
    PRESETS,
)
import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from ardy_mlx.sampling.autoregressive import (  # noqa: E402
    ArdyAutoregressiveConfig,
    ArdySampler,
)
from ardy_mlx.sampling.demo import make_demo_denoise_fn, sanitize_placeholder_output  # noqa: E402
from ardy_mlx.sampling.diffusion import resolve_num_steps  # noqa: E402
from ardy_mlx.sampling.pipeline import (  # noqa: E402
    compute_feature_layout,
    duration_to_frames,
    load_text_embedding,
    save_outputs,
    synthesize_posed_joints,
)

BACKENDS = ("placeholder", "mlx")
#: "g1" not in model name -> post-processing applies (generate.py:265-275).
POSTPROCESS_DISABLED_MODELS = ("g1",)


def _postprocess_applies(model: str) -> bool:
    return not any(token in model.lower() for token in POSTPROCESS_DISABLED_MODELS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ardy_generate.py",
        description="Generate one motion clip with ardy-mlx (contract S7 output).",
    )
    parser.add_argument(
        "--embedding",
        required=True,
        help="precomputed text embedding (.npy [L,4096] or [B,L,4096]; .npz with 'embedding')",
    )
    parser.add_argument(
        "--embedding-array",
        default="embedding",
        help="array name inside an .npz embedding file (default: embedding)",
    )
    parser.add_argument(
        "--embedding-mask-array",
        default=None,
        help="optional mask array name inside an .npz embedding file",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="raw prompt text; defaults to the embedding file name. Only used for the output file name and the npz 'text' member.",
    )
    parser.add_argument(
        "--model",
        default="ARDY-Core-RP-20FPS-Horizon40",
        choices=sorted(PRESETS),
        help="checkpoint preset (default: %(default)s)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="clip duration in seconds (default: %(default)s)",
    )
    parser.add_argument("--seed", type=int, default=None, help="RNG seed (default: none)")
    parser.add_argument(
        "--diffusion-steps",
        type=int,
        default=None,
        help=f"denoising steps, 1..num_base_steps; default num_base_steps = {NUM_BASE_STEPS}",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        nargs="+",
        default=None,
        metavar="W",
        help=(
            "classifier-free guidance weight(s): one value = text only (constraint 0.0), "
            f"two values = [text, constraint]; default {list(DEFAULT_CFG_WEIGHT)}"
        ),
    )
    parser.add_argument(
        "--history-frames",
        type=int,
        default=None,
        help=(
            "history crop length; must be a multiple of "
            f"{NUM_FRAMES_PER_TOKEN}; default = batch max (160 for Core/20fps/H40)"
        ),
    )
    parser.add_argument(
        "--no-postprocess",
        action="store_true",
        help="do not apply motion post-processing (v1 default; ignored for G1)",
    )
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="placeholder",
        help=(
            "placeholder = random-weight stand-in, shape-correct but meaningless "
            "(default); mlx = A's converted weights, not wired yet"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=("fast", "dense"),
        default="fast",
        help="placeholder backend width: fast (64) or dense (1024, contract width)",
    )
    parser.add_argument(
        "--out",
        required=True,
        help="output directory; artifacts go to <out>/<model>/<prompt-hash>-<seed>.{npz,json}",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="batch size (default 1)")
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="checkpoint root for --backend mlx (default: checkpoints/<model> in this repo)",
    )
    parser.add_argument(
        "--noise-file",
        default=None,
        help=(
            "inject the initial noise from a .npy file (one window per leading axis, or a "
            "single window for a one-window run); used to reproduce the reference x_T"
        ),
    )
    parser.add_argument(
        "--device",
        choices=("gpu", "cpu"),
        default="gpu",
        help="MLX device for --backend mlx (default: gpu)",
    )
    return parser


def _resolve_cfg_weight(values) -> list:
    if values is None:
        return list(DEFAULT_CFG_WEIGHT)
    if len(values) == 1:
        # generate.py:176-181 -- a single value is the text weight, constraint 0.0.
        return [float(values[0]), 0.0]
    if len(values) == 2:
        return [float(values[0]), float(values[1])]
    raise SystemExit(
        f"--cfg-weight takes 1 or 2 values (text, constraint), got {len(values)}"
    )


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    cfg_weight = _resolve_cfg_weight(args.cfg_weight)
    num_steps = resolve_num_steps(args.diffusion_steps, NUM_BASE_STEPS)

    layout, init_history_len, plan_steps = compute_feature_layout(
        args.model, duration_to_frames(args.duration, PRESETS[args.model].fps), args.history_frames
    )
    total_frames = duration_to_frames(args.duration, layout.fps)

    postprocess_applies = _postprocess_applies(args.model)
    if postprocess_applies and not args.no_postprocess:
        print(
            "error: Core models run MotionCorrection by default, but the C++ extension is not "
            "built here and contract S6.4 / decision 0002 ruling 2 pin v1 at 'both sides skip it'.\n"
            "       re-run with --no-postprocess (the sidecar will record postprocess: false).",
            file=sys.stderr,
        )
        return 2

    started = time.time()
    embedding = load_text_embedding(
        args.embedding,
        array=args.embedding_array,
        mask_array=args.embedding_mask_array,
    )
    if embedding.batch_size != args.batch_size:
        print(
            f"error: embedding batch {embedding.batch_size} != --batch-size {args.batch_size}",
            file=sys.stderr,
        )
        return 2

    prompt = args.prompt if args.prompt is not None else Path(args.embedding).stem

    print(
        f"[ardy_generate] model={args.model} fps={layout.fps} horizon={layout.horizon} "
        f"D={layout.motion_rep_dim} J={layout.nbjoints}"
    )
    print(
        f"[ardy_generate] duration={args.duration}s -> requested {total_frames} frames; "
        f"init_history={init_history_len} (batch max); steps={plan_steps}; "
        f"diffusion_steps={num_steps}; cfg_weight={cfg_weight}; backend={args.backend}"
    )

    if args.backend == "mlx":
        result = _run_mlx_backend(args, layout, embedding, total_frames, num_steps, tuple(cfg_weight))
    else:
        # The local loop seeds a context block *before* generating, so its clip is
        # seed + chunks; the requested frame count has to cover the seed too.
        if total_frames < init_history_len:
            print(
                f"error: --duration {args.duration}s is {total_frames} frames, shorter than the "
                f"{init_history_len}-frame seed context the placeholder loop uses. Increase "
                f"--duration or lower --history-frames (the reference batch path has no seed "
                f"context; use --backend mlx for short clips).",
                file=sys.stderr,
            )
            return 2
        result = _run_placeholder_backend(
            args, layout, embedding, total_frames, init_history_len, num_steps, tuple(cfg_weight)
        )

    npz_path, json_path = save_outputs(
        out_dir=Path(args.out) / args.model,
        model=args.model,
        layout=layout,
        frames=result["frames"],
        posed_joints=result["posed_joints"],
        root_positions=result["root_positions"],
        foot_contacts=result["foot_contacts"],
        offsets=result["offsets"],
        prompt=prompt,
        seed=args.seed,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        postprocess=False,
        duration_s=args.duration,
        embedding_sha256=embedding.sha256,
        embedding_source=embedding.source_path,
        backend=args.backend,
        extra_sidecar=result["extra"],
    )

    elapsed = time.time() - started
    print(
        f"[ardy_generate] frames={tuple(result['frames'].shape)} "
        f"offsets={list(result['offsets'])} ({len(result['offsets']) - 1} chunks)"
    )
    print(f"[ardy_generate] wrote {npz_path}")
    print(f"[ardy_generate] wrote {json_path}")
    print(f"[ardy_generate] elapsed {elapsed:.2f}s")
    return 0


def _common_outputs(frames, layout):
    """Root positions, contacts and joints derived from the contract feature rows."""
    return {
        "frames": frames,
        "root_positions": frames[..., layout.slices["root_pos"]],
        "foot_contacts": frames[..., layout.slices["foot_contacts"]] > 0.5,
        "posed_joints": synthesize_posed_joints(frames, layout),
    }


def _run_placeholder_backend(args, layout, embedding, total_frames, init_history_len, num_steps, cfg_weight):
    config = ArdyAutoregressiveConfig(
        layout=layout,
        num_frames=total_frames,
        batch_size=args.batch_size,
        history_frames=args.history_frames,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        seed=args.seed,
        text_embeddings=embedding.embeddings,
        text_mask=embedding.mask,
    )
    sampler = ArdySampler(config, make_demo_denoise_fn(layout, mode=args.mode, seed=args.seed or 0))
    sampled = sampler.run()
    frames = sanitize_placeholder_output(sampled.translated(), layout)
    return {
        **_common_outputs(frames, layout),
        "offsets": sampled.offsets,
        "extra": {
            "mode": args.mode,
            "init_history_frames": init_history_len,
            "num_autoregressive_steps": len(sampled.offsets) - 1,
            "global_transl": [float(v) for v in sampled.global_transl[0]],
            "first_heading_angle": float(sampled.first_heading_angle[0]),
            "posed_joints_source": (
                "local_joints_positions (root-local) + root_pos; real FK not implemented"
            ),
        },
    }


def _run_mlx_backend(args, layout, embedding, total_frames, num_steps, cfg_weight):
    """Real weights: A's ``src/ardy_mlx/model/**`` assembled as ``ArdyBatch``.

    The window loop, recentring, requantisation and detokenisation all live in
    A's ``model/ardy.py`` (it is the side ``scripts/ardy_align.py`` aligns), so
    this path delegates to it rather than reimplementing the loop -- the local
    sampler is what the placeholder backend exercises and what the golden
    comparison validates.

    ``x_T`` is drawn here with MLX's RNG: CUDA's Philox stream is not
    reproducible in MLX (A's module docstring), so absolute equality with the
    reference needs the reference's own ``x_T`` (the align harness injects it).
    """
    from ardy_mlx.model.ardy import ArdyBatch
    from ardy_mlx.model.denoiser import AutoLatentTwostageDenoiser
    from ardy_mlx.model.fsq import FSQVAETransformer
    from ardy_mlx.model.motion_rep import MotionRep
    from ardy_mlx.model.nn import WeightStore

    ckpt = Path(args.checkpoint_dir) if args.checkpoint_dir else (REPO_ROOT / "checkpoints" / args.model)
    if not ckpt.is_dir():
        raise SystemExit(f"error: checkpoint directory not found: {ckpt}")
    tokenizer_weights = ckpt / "mlx" / "tokenizer.mlx.safetensors"
    denoiser_weights = ckpt / "mlx" / "denoiser.mlx.safetensors"
    for path in (tokenizer_weights, denoiser_weights):
        if not path.exists():
            raise SystemExit(
                f"error: {path} is missing; run scripts/ardy_convert.py first "
                "(see docs/status-deepseek.md)"
            )

    mx.set_default_device(mx.gpu if args.device == "gpu" else mx.cpu)
    motion_rep = MotionRep(ckpt / "stats" / "motion")
    tokenizer = FSQVAETransformer(WeightStore.load(tokenizer_weights), motion_rep).load_stats(
        ckpt / "stats"
    )
    denoiser = AutoLatentTwostageDenoiser(WeightStore.load(denoiser_weights), motion_rep)
    batch_pipe = ArdyBatch(tokenizer, denoiser, motion_rep)

    base_key = mx.random.key(int(args.seed)) if args.seed is not None else mx.random.key(0)
    windows = max(plan_steps_of(args, layout), 1)
    step_keys = list(mx.random.split(base_key, windows))

    injected = None
    if args.noise_file is not None:
        injected = np.load(args.noise_file, allow_pickle=False).astype(np.float32)
        print(f"[ardy_generate] noise injected from {args.noise_file} shape={injected.shape}")

    def noise_fn(step, shape):
        # ``--noise-file`` reproduces the reference x_T. A file may hold one window
        # per leading axis, or a single window for a one-window run (the aligned
        # case); later windows then fall back to MLX's RNG.
        if injected is not None:
            if injected.shape == shape:
                if windows > 1:
                    print(
                        f"[ardy_generate] warning: --noise-file holds one window but this run "
                        f"has {windows}; steps >= 1 use the MLX RNG"
                    )
                return mx.array(injected)
            if step < injected.shape[0] and injected.shape[1:] == shape[1:]:
                return mx.array(injected[step])
        return mx.random.normal(shape, key=step_keys[min(step, len(step_keys) - 1)]).astype(
            mx.float32
        )

    out = batch_pipe.generate(
        embedding.embeddings,
        embedding.mask,
        num_frames=total_frames,
        num_steps=num_steps,
        cfg_weight=cfg_weight,
        first_heading_angle=mx.zeros((embedding.batch_size,)),
        crop_history_length=args.history_frames,
        noise_fn=noise_fn,
    )
    frames = out["frames"].astype(mx.float32)
    return {
        **_common_outputs(frames, layout),
        "offsets": list(out["offsets"]),
        "extra": {
            "mode": "dense",
            "init_history_frames": args.history_frames
            if args.history_frames is not None
            else layout.default_history_frames,
            "num_autoregressive_steps": max(len(out["offsets"]) - 1, 0),
            "posed_joints_source": (
                "local_joints_positions (root-local) + root_pos; real FK not implemented"
            ),
            "noise": "mlx.random (CUDA Philox x_T is not reproducible; align injects the reference x_T)",
            "checkpoint_dir": str(ckpt),
        },
    }


def plan_steps_of(args, layout):
    """How many autoregressive windows this run will take (for the RNG keys)."""
    _layout, init_history_len, _ = compute_feature_layout(
        args.model,
        duration_to_frames(args.duration, layout.fps),
        args.history_frames,
    )
    total_frames = duration_to_frames(args.duration, layout.fps)
    return int(-(-(total_frames - init_history_len) // layout.horizon))


if __name__ == "__main__":
    raise SystemExit(main())
