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
``--history-frames`` unset resolves to the model preset's own
``default_history_frames`` -- the **batch** history (the *maximum*), 160 frames
for Core/20fps/H40, not the GUI slider's minimum of one patch (S5.2). It is
resolved to a number rather than left ``None``: A's batch path reads ``None`` as
``crop_history_length=None``, which silently drops cropping, the heading refresh
and the chunked decode (contract S11 N16).

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

from ardy_mlx.model.skeleton import posed_joints_from_features  # noqa: E402
from ardy_mlx.sampling.autoregressive import (  # noqa: E402
    ArdyAutoregressiveConfig,
    ArdySampler,
)
from ardy_mlx.sampling.demo import make_demo_denoise_fn, sanitize_placeholder_output  # noqa: E402
from ardy_mlx.sampling.diffusion import resolve_num_steps  # noqa: E402
from ardy_mlx.sampling.pipeline import (  # noqa: E402
    POSED_JOINTS_SOURCE,
    compute_feature_layout,
    duration_to_frames,
    load_text_embedding,
    resolve_history_frames,
    resolve_noise_windows,
    save_outputs,
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
            f"{NUM_FRAMES_PER_TOKEN}; default = the model preset's own "
            "default_history_frames (160 for Core/20fps/H40, 196 for G1/25fps/H52)"
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
            "(default); mlx = A's converted weights via ArdyBatch (the standard path)"
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
            "inject the initial noise from a .npy file; either one window's x_T "
            "(only for a one-window run) or a stack with one entry per window, "
            "shape (n_windows,) + (batch, gen_tokens, dim_token). A mismatch fails "
            "loudly rather than falling back to the MLX RNG"
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

    # ``--history-frames`` unset must resolve to the preset's batch default
    # (contract S5.2). Forwarding the raw ``None`` reaches A's batch path as
    # ``crop_history_length=None``, which silently disables cropping, the
    # ``first_heading_angle`` refresh and the chunked decode (contract S11 N16).
    history_frames = resolve_history_frames(args.model, args.history_frames)

    layout, init_history_len, plan_steps = compute_feature_layout(
        args.model, duration_to_frames(args.duration, PRESETS[args.model].fps), history_frames
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

    # ``--noise-file`` is validated at the CLI boundary, before any weight loads.
    # A wrong x_T still produces plausible-looking motion, so a shape mismatch, a
    # short file or one window's noise for a multi-window run must fail loudly:
    # silently falling back to MLX's RNG put the whole 80-frame clip 4.42 away
    # from golden with no message at all (contract S11 N16).
    windows = max(plan_steps, 1)
    window_shape = (
        embedding.batch_size,
        layout.horizon // NUM_FRAMES_PER_TOKEN,
        layout.dim_token,
    )
    noise_windows = None
    if args.noise_file is not None and args.backend == "mlx":
        raw_noise = np.load(args.noise_file, allow_pickle=False).astype(np.float32)
        print(f"[ardy_generate] noise injected from {args.noise_file} shape={raw_noise.shape}")
        try:
            noise_windows = resolve_noise_windows(
                raw_noise, window_shape, windows, source="--noise-file"
            )
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2

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
        result = _run_mlx_backend(
            args,
            layout,
            embedding,
            total_frames,
            num_steps,
            tuple(cfg_weight),
            history_frames,
            windows,
            noise_windows,
        )
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
            args,
            layout,
            embedding,
            total_frames,
            init_history_len,
            num_steps,
            tuple(cfg_weight),
            history_frames,
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
    """Root positions, contacts and joints derived from the contract feature rows.

    ``posed_joints`` is the real ``CoreSkeleton27`` forward kinematics over the
    ``global_rot_data`` + ``root_pos`` blocks (contract S11 N14), not the
    pre-FK placeholder: the placeholder was ``0.132`` / ``0.236`` away from the
    golden npz, this is ``~5e-06``, the same order as the clip's own residue.
    """
    return {
        "frames": frames,
        "root_positions": frames[..., layout.slices["root_pos"]],
        "foot_contacts": frames[..., layout.slices["foot_contacts"]] > 0.5,
        "posed_joints": posed_joints_from_features(frames),
    }


def _run_placeholder_backend(
    args, layout, embedding, total_frames, init_history_len, num_steps, cfg_weight, history_frames
):
    config = ArdyAutoregressiveConfig(
        layout=layout,
        num_frames=total_frames,
        batch_size=args.batch_size,
        history_frames=history_frames,
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
            "posed_joints_source": POSED_JOINTS_SOURCE,
        },
    }


def _run_mlx_backend(
    args, layout, embedding, total_frames, num_steps, cfg_weight, history_frames, windows, noise_windows
):
    """Real weights: A's ``src/ardy_mlx/model/**`` assembled as ``ArdyBatch``.

    The window loop, recentring, requantisation and detokenisation all live in
    A's ``model/ardy.py`` (it is the side ``scripts/ardy_align.py`` aligns), so
    this path delegates to it rather than reimplementing the loop -- the local
    sampler is what the placeholder backend exercises and what the golden
    comparison validates.

    ``x_T`` is drawn here with MLX's RNG: CUDA's Philox stream is not
    reproducible in MLX (A's module docstring), so absolute equality with the
    reference needs the reference's own ``x_T`` (the align harness injects it, or
    ``--noise-file`` supplies one entry per window, validated in :func:`main`).
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
    step_keys = list(mx.random.split(base_key, windows))
    window_shape = (
        embedding.batch_size,
        layout.horizon // NUM_FRAMES_PER_TOKEN,
        layout.dim_token,
    )

    def noise_fn(step, shape):
        # ``--noise-file`` reproduces the reference x_T: one entry per window
        # (single-window files are only accepted for single-window runs). A shape
        # mismatch or a short file already raised in ``main``; it must never fall
        # back to MLX's RNG, because that error is O(1) and silent (contract S11 N16).
        if noise_windows is not None:
            if tuple(shape) != window_shape:
                raise ValueError(
                    f"model asked for noise shape {tuple(shape)} but this run was validated "
                    f"for {window_shape}; refusing to inject a mismatched x_T"
                )
            return mx.array(noise_windows[min(step, len(noise_windows) - 1)])
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
        crop_history_length=history_frames,
        noise_fn=noise_fn,
    )
    frames = out["frames"].astype(mx.float32)
    return {
        **_common_outputs(frames, layout),
        "offsets": list(out["offsets"]),
        "extra": {
            "mode": "dense",
            "init_history_frames": history_frames,
            "num_autoregressive_steps": max(len(out["offsets"]) - 1, 0),
            "posed_joints_source": POSED_JOINTS_SOURCE,
            "noise": "mlx.random (CUDA Philox x_T is not reproducible; align injects the reference x_T)",
            "checkpoint_dir": str(ckpt),
        },
    }


if __name__ == "__main__":
    raise SystemExit(main())
