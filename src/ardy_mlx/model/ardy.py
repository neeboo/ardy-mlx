"""The batch generation path: ``Ardy`` -- window loop, recentring, detokenising.

Reference: ``ardy/model/ardy_model.py`` (the ``Ardy`` class) plus
``ardy/model/latent_utils.py``. This is the assembly that turns the two stages
into whole motion, and it is the thing ``scripts/ardy_align.py`` compares against
the golden clips.

Why this file exists separately from the stages: the recentring/translation
arithmetic in ``Ardy.__call__`` is easy to get subtly wrong and impossible to
test through either stage alone. For a 2.0 s Core clip (40 frames at 20 fps,
``gen_horizon_len == 40``) there is exactly **one** autoregressive window, so the
whole clip is: 10 DDIM steps -> recentre -> requantise -> translate -> detokenise.
For 4.0 s (80 frames) there are **two** windows, and the second one is the first
place where the multi-window code paths are actually observable:

* window 1 is built from 10 tokens of **clean history** plus 10 fresh ones, while
  window 0 is 10 generation tokens followed by 10 *masked future* tokens
  (``future_len = 80 - 40 - 0 = 40`` frames);
* the recentring translation accumulates over two steps;
* ``first_heading_angle`` is refreshed after window 0 and fed into window 1
  (measured on the golden clip: ``0.0 -> 0.00261``);
* the final decode runs **chunk by chunk**
  (``latent_utils.py:103-140``) rather than once over the whole sequence.

The initial noise is **injected**, not sampled here: ``noise_fn(step, shape)``
must supply it. That keeps this module deterministic and lets the alignment
harness feed the reference's own ``x_T`` -- CUDA's Philox RNG is not reproducible
in MLX, so re-deriving the noise would be a separate (and much harder) project.

The last thing in this file is :func:`make_step_fn`, the **seam for the sampling
side (B)**: it adapts this batch path to the ``DenoiseRequest`` /
``DenoiseResult`` step callable that ``ardy_mlx.sampling.ArdySampler`` drives.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import mlx.core as mx
import numpy as np

from .denoiser import AutoLatentTwostageDenoiser, AutoLatentClassifierFreeGuidedModelSeparated
from .diffusion import DDIMSampler, Diffusion
from .fsq import FSQVAETransformer
from .motion_rep import MotionRep

__all__ = [
    "Ablation",
    "ArdyBatch",
    "assemble_window_latent",
    "autoregressive_chunk_slices",
    "convert_frame_mask_to_token_mask",
    "get_three_mask_from_len",
    "hybrid_from_explicit_frames",
    "make_step_fn",
    "timestep_indices",
]


# --------------------------------------------------------------------------- #
# pure helpers (no weights, no state -- these are the unit-testable pieces)
# --------------------------------------------------------------------------- #
def get_three_mask_from_len(history_len, generation_len, future_len, total_len):
    """``get_three_mask_from_len`` (ardy_model.py:15-26).

    Returns float ``[B, total_len]`` masks (the reference multiplies by them).
    """
    batch = history_len.shape[0]
    idx = mx.arange(total_len, dtype=mx.float32)
    history_end = history_len[:, None]
    generation_end = (history_len + generation_len)[:, None]
    history_mask = (idx[None, :] < history_end).astype(mx.float32)
    generation_mask = ((idx[None, :] >= history_end) & (idx[None, :] < generation_end)).astype(mx.float32)
    future_mask = (idx[None, :] >= generation_end).astype(mx.float32)
    return history_mask, generation_mask, future_mask


def timestep_indices(num_steps: int) -> List[int]:
    """``indices = list(range(num_denoising_steps))[::-1]`` (ardy_model.py:619).

    The loop variable indexes the *subsampled* schedule; ``diffusion.map_tensor``
    indexes the diffusion buffers. For ``num_steps == num_base_steps`` (10) the two
    happen to coincide, which is exactly why conflating them is easy to miss.
    """
    if num_steps < 1:
        raise ValueError(f"num_steps must be >= 1, got {num_steps}")
    return list(range(num_steps))[::-1]


def convert_frame_mask_to_token_mask(
    history_mask: mx.array,
    generation_mask: mx.array,
    future_mask: mx.array,
    motion_mask: Optional[mx.array] = None,
    num_frames_per_token: int = 4,
):
    """``HybridMotionConverter.convert_frame_mask_to_token_mask`` (latent_utils.py:192-220).

    **The trap this function exists to pin** (``latent_utils.py:207``): when
    ``motion_mask is None`` -- which is the whole no-constraint batch path --
    ``future_token_mask`` is ``zeros_like(history_token_mask)``, *not*
    ``future_mask`` reduced to tokens. The three frame masks and the three token
    masks therefore disagree whenever ``future_len > 0``: window 0 of an 80-frame
    clip has ``future_mask.sum() == 40`` frames but ``future_token_mask.sum() == 0``
    tokens. The token mask feeds ``*_stage_pad_mask`` (attention key padding) in
    the denoiser, so getting it wrong silently changes window 0's output.
    """
    bs, num_frames = history_mask.shape[0], history_mask.shape[1]
    nt = num_frames // num_frames_per_token
    history_token_mask = history_mask.reshape(bs, nt, num_frames_per_token).all(axis=-1)
    generation_token_mask = generation_mask.reshape(bs, nt, num_frames_per_token).all(axis=-1)
    if motion_mask is None:
        future_token_mask = mx.zeros_like(history_token_mask)
    else:
        future_token_mask = future_mask.reshape(bs, nt, num_frames_per_token).all(axis=-1)
        has_observation = motion_mask.any(axis=-1)  # [B, T]
        has_observation = has_observation.reshape(bs, nt, num_frames_per_token).any(axis=-1)
        future_token_mask = future_token_mask & has_observation
    return history_token_mask, generation_token_mask, future_token_mask


def autoregressive_chunk_slices(
    token_len: int, gen_tokens: int, crop_tokens: Optional[int]
) -> List[Tuple[int, int, int]]:
    """Chunk plan of ``get_explicit_motion_from_hybrid_autoregressive`` (latent_utils.py:121-137).

    Returns ``(history_token_start, generation_token_start, generation_token_end)``
    per generation horizon. ``crop_tokens`` is ``crop_history_length //
    num_frames_per_token``; with the Core default (160 frames = 40 tokens) a
    2-window clip never truncates, but the arithmetic is the same one that does.
    """
    if gen_tokens < 1:
        raise ValueError(f"gen_tokens must be >= 1, got {gen_tokens}")
    out = []
    for token_idx in range(0, token_len, gen_tokens):
        generation_token_end = min(token_idx + gen_tokens, token_len)
        history_token_start = (
            max(0, token_idx - crop_tokens) if crop_tokens is not None else 0
        )
        out.append((history_token_start, token_idx, generation_token_end))
    return out


def assemble_window_latent(
    history_hybrid: Optional[mx.array],
    x_t: mx.array,
    history_token_end: int,
    generation_token_end: int,
    total_tokens: int,
) -> mx.array:
    """The ``x`` a window starts from (ardy_model.py:481-490).

    ``zeros(total_tokens)`` with the *clean* history written into the first
    ``history_token_end`` positions and the fresh noise into the generation span.
    The region after the generation span (when ``total_frames`` exceeds what this
    window needs) stays zero: those are the masked *future* tokens.
    """
    bs = x_t.shape[0]
    d = x_t.shape[-1]
    x = mx.zeros((bs, total_tokens, d))
    if history_token_end > 0:
        if history_hybrid is None:
            raise ValueError("history_token_end > 0 but no history_hybrid was supplied")
        x = x.at[:, :history_token_end].add(history_hybrid[:, -history_token_end:])
    return x.at[:, history_token_end:generation_token_end].add(x_t)


def hybrid_from_explicit_frames(
    batch: "ArdyBatch", frames: mx.array, motion_pad_mask: Optional[mx.array] = None
) -> mx.array:
    """``HybridMotionConverter.get_hybrid_motion_from_explicit`` (latent_utils.py:159-188).

    ``frames`` are **normalised** 330-dim explicit features; the result is the
    148-dim hybrid token = ``[root block flattened over 4 frames (20) | body
    latent (128)]``. Masked-out tokens have their latent zeroed
    (``latent_utils.py:171-174``), which the reference does and a naive
    ``tokenize`` would not.
    """
    f = batch.num_frames_per_token
    mr = batch.motion_rep
    b, num_frames, _ = frames.shape
    if num_frames % f != 0:
        raise ValueError(f"frame count {num_frames} must be a multiple of {f}")
    nt = num_frames // f
    if motion_pad_mask is None:
        motion_pad_mask = mx.ones((b, num_frames), dtype=mx.bool_)
    latent = batch.tokenizer.tokenize(frames, motion_pad_mask)
    token_mask = motion_pad_mask.reshape(b, nt, f).all(axis=-1)
    latent = latent * token_mask[:, :, None]
    root = mr.extract_root(frames).reshape(b, nt, mr.root_block_dim * f)
    return mx.concatenate([root, latent], axis=-1)


# --------------------------------------------------------------------------- #
# deliberate defects, for negative controls
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Ablation:
    """Knobs that *break* the reference behaviour on purpose (negative controls).

    Every default reproduces ``nv-tlabs/ardy`` @ ``693f74d`` exactly. The
    alignment harness flips one field at a time and asserts the comparison then
    fails -- a tolerance that cannot fail is not a tolerance.
    """

    #: Shift the recentring centre by this many frames (one token = 4 frames).
    center_index_offset: int = 0
    #: The reference refreshes ``first_heading_angle`` after cropping
    #: (ardy_model.py:665-674). ``False`` keeps the stale value.
    refresh_first_heading: bool = True
    #: The pre-fix bug: derive ``future_token_mask`` from ``future_mask`` even
    #: when there are no constraints (latent_utils.py:207 says otherwise).
    future_token_mask_from_frames: bool = False
    #: The reference decodes chunk-by-chunk when ``crop_history_length`` is set
    #: (ardy_model.py:684-691). ``False`` decodes the whole sequence at once.
    autoregressive_decode: bool = True


# --------------------------------------------------------------------------- #
# the assembled batch path
# --------------------------------------------------------------------------- #
class ArdyBatch:
    """Assembled Core batch path (``gen_horizon_len=40``, ``num_frames_per_token=4``)."""

    def __init__(
        self,
        tokenizer: FSQVAETransformer,
        denoiser: AutoLatentTwostageDenoiser,
        motion_rep: MotionRep,
        gen_horizon_len: int = 40,
        num_base_steps: int = 10,
    ):
        self.tokenizer = tokenizer
        self.denoiser = denoiser
        self.motion_rep = motion_rep
        self.gen_horizon_len = gen_horizon_len
        self.num_frames_per_token = tokenizer.num_frames_per_token
        self.diffusion = Diffusion(num_base_steps)
        self.sampler = DDIMSampler(self.diffusion)
        self.cfg = AutoLatentClassifierFreeGuidedModelSeparated(denoiser)

    # ------------------------------------------------------------ window loop
    def _generate_window(
        self,
        x_t: mx.array,
        history_sequence: Optional[mx.array],
        text_feat: mx.array,
        text_pad_mask: mx.array,
        first_heading_angle: mx.array,
        num_steps: int,
        cfg_weight: tuple,
        total_frames: int,
        history_start_frame: int,
        history_end_frame: int,
        ablation: Ablation = Ablation(),
        trace: Optional[Dict] = None,
        window_index: int = 0,
    ) -> mx.array:
        """``Ardy._generate_window`` (ardy_model.py:415-538) for the no-constraint case."""
        f = self.num_frames_per_token
        bs = text_feat.shape[0]
        num_generation_tokens = self.gen_horizon_len // f

        history_token_end = (history_end_frame - history_start_frame) // f
        generation_token_end = history_token_end + num_generation_tokens
        history_len = mx.full((bs,), history_token_end * f, dtype=mx.float32)
        generation_len = mx.full((bs,), self.gen_horizon_len, dtype=mx.float32)
        future_len = mx.full((bs,), total_frames - self.gen_horizon_len - history_end_frame, dtype=mx.float32)

        total_frames_window = total_frames - history_start_frame
        history_mask, generation_mask, future_mask = get_three_mask_from_len(
            history_len, generation_len, future_len, total_frames_window
        )
        if ablation.future_token_mask_from_frames:
            history_token_mask = history_mask.reshape(bs, -1, f).all(axis=-1)
            generation_token_mask = generation_mask.reshape(bs, -1, f).all(axis=-1)
            future_token_mask = future_mask.reshape(bs, -1, f).all(axis=-1)
        else:
            history_token_mask, generation_token_mask, future_token_mask = convert_frame_mask_to_token_mask(
                history_mask, generation_mask, future_mask, motion_mask=None, num_frames_per_token=f
            )

        total_tokens = total_frames_window // f
        x = assemble_window_latent(
            history_sequence, x_t, history_token_end, generation_token_end, total_tokens
        )
        if trace is not None:
            trace.setdefault("steps", [])
            trace.setdefault("windows", [])
            while len(trace["windows"]) <= window_index:
                trace["windows"].append({"index": len(trace["windows"])})
            rec = trace["windows"][window_index]
            rec.update(
                {
                    "index": window_index,
                    "history_start_frame": int(history_start_frame),
                    "history_end_frame": int(history_end_frame),
                    "history_token_end": int(history_token_end),
                    "total_tokens": int(total_tokens),
                    "future_len": int(future_len[0]),
                    "frame_future_frames": int(np.asarray(future_mask).sum()),
                    "token_history_mask_sum": int(np.asarray(history_token_mask).sum()),
                    "token_generation_mask_sum": int(np.asarray(generation_token_mask).sum()),
                    "token_future_mask_sum": int(np.asarray(future_token_mask).sum()),
                    "first_heading_angle_in": np.asarray(first_heading_angle).copy(),
                    "history_in": None if history_sequence is None else np.asarray(history_sequence).copy(),
                    "x0": np.asarray(x).copy(),
                }
            )

        indices = timestep_indices(num_steps)
        for i in indices:
            use_timesteps = self.diffusion.space_timesteps(num_steps)[0]
            self.diffusion.calc_diffusion_vars(use_timesteps)
            t_map = int(self.diffusion.map_tensor[i])
            timesteps = mx.full((bs,), float(t_map))
            clean = self.cfg(
                mx.array([cfg_weight[0]], dtype=mx.float32),
                mx.array([cfg_weight[1]], dtype=mx.float32),
                x,
                history_len=history_len,
                generation_len=generation_len,
                future_len=future_len,
                history_mask=history_mask,
                generation_mask=generation_mask,
                future_mask=future_mask,
                history_token_mask=history_token_mask,
                generation_token_mask=generation_token_mask,
                future_token_mask=future_token_mask,
                text_feat=text_feat,
                text_feat_pad_mask=text_pad_mask,
                timesteps=timesteps,
                first_heading_angle=first_heading_angle,
                motion_mask=None,
                observed_motion=None,
            )
            # DDIM update on the generation tokens only
            gen_t = x[:, history_token_end:generation_token_end]
            gen_clean = clean[:, history_token_end:generation_token_end]
            # ardy_model.py:302-313 passes the *loop* index to the sampler and the
            # *mapped* timestep to the denoiser; they coincide only because
            # num_steps == num_base_steps here (diffusion.py:50-58).
            gen_next = self.sampler(gen_t, gen_clean, i)
            x = mx.concatenate(
                [x[:, :history_token_end], gen_next, x[:, generation_token_end:]], axis=1
            )
            if trace is not None:
                trace["steps"].append(
                    {
                        "window": window_index,
                        "index": len(trace["steps"]),
                        "t": int(t_map),
                        "loop_index": int(i),
                        "x_in": np.asarray(x).copy(),
                    }
                )

        if history_sequence is None:
            return x[:, :generation_token_end]
        return mx.concatenate([history_sequence, x[:, history_token_end:generation_token_end]], axis=1)

    # ------------------------------------------------------------- decoding
    def explicit_from_hybrid_autoregressive(
        self,
        latent: mx.array,
        root_motion: mx.array,
        crop_history_length: Optional[int],
        motion_pad_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """``get_explicit_motion_from_hybrid_autoregressive`` (latent_utils.py:103-140).

        Each generation horizon is detokenised **with only its own (cropped)
        history** and then stitched, unlike
        :meth:`explicit_from_hybrid_whole`. Because the Stage-A decoder is causal
        the two agree up to float32 reduction order -- ``scripts/ardy_align.py``
        measures exactly that.
        """
        f = self.num_frames_per_token
        gen_tokens = self.gen_horizon_len // f
        token_len = latent.shape[1]
        crop_tokens = None if crop_history_length is None else crop_history_length // f
        pieces = []
        for h_start, g_start, g_end in autoregressive_chunk_slices(token_len, gen_tokens, crop_tokens):
            h_fs, g_fs, g_fe = h_start * f, g_start * f, g_end * f
            sub_root = root_motion[:, h_fs:g_fe]
            sub_latent = latent[:, h_start:g_end]
            sub_mask = None if motion_pad_mask is None else motion_pad_mask[:, h_fs:g_fe]
            lengths = None if sub_mask is None else mx.sum(sub_mask, axis=-1)
            local_root = self.motion_rep.global_root_to_local_root(
                sub_root, normalized=True, lengths=lengths
            )
            body = self.tokenizer.detokenize(sub_latent, local_root, sub_mask)["body"]
            explicit = self.motion_rep.concat_root_body(sub_root, body)
            pieces.append(explicit[:, g_fs - h_fs : g_fe - h_fs])
        return mx.concatenate(pieces, axis=1)

    def explicit_from_hybrid_whole(
        self,
        latent: mx.array,
        root_motion: mx.array,
        motion_pad_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """``get_explicit_motion_from_hybrid`` (latent_utils.py:78-102)."""
        lengths = None if motion_pad_mask is None else mx.sum(motion_pad_mask, axis=-1)
        local_root = self.motion_rep.global_root_to_local_root(
            root_motion, normalized=True, lengths=lengths
        )
        body = self.tokenizer.detokenize(latent, local_root, motion_pad_mask)["body"]
        return self.motion_rep.concat_root_body(root_motion, body)

    # --------------------------------------------------------------- generate
    def generate(
        self,
        text_feat: mx.array,
        text_pad_mask: mx.array,
        num_frames: int,
        num_steps: int = 10,
        cfg_weight: tuple = (2.0, 2.0),
        first_heading_angle: Optional[mx.array] = None,
        crop_history_length: Optional[int] = 160,
        noise_fn: Optional[Callable[[int, tuple], mx.array]] = None,
        init_history_sequence: Optional[mx.array] = None,
        ablation: Ablation = Ablation(),
        trace: Optional[Dict] = None,
    ) -> dict:
        """``Ardy.__call__`` (ardy_model.py:538-695) for the no-constraint batch path.

        ``trace`` (optional) is filled with the per-window and per-step arrays the
        alignment harness needs; it never changes the computation.
        """
        f = self.num_frames_per_token
        bs = text_feat.shape[0]
        if first_heading_angle is None:
            first_heading_angle = mx.zeros((bs,))

        init_history_len = 0 if init_history_sequence is None else init_history_sequence.shape[1]
        num_auto_steps = int(-(-(num_frames - init_history_len) // self.gen_horizon_len))
        num_gen_frames = num_auto_steps * self.gen_horizon_len + init_history_len

        history_sequence = init_history_sequence
        global_transl = mx.zeros((bs, 3))
        offsets = [0]
        if trace is not None:
            trace["steps"] = []
            trace["windows"] = []

        for auto_step in range(num_auto_steps):
            history_end_frame = auto_step * self.gen_horizon_len + init_history_len
            history_start_frame = (
                0 if crop_history_length is None else max(0, history_end_frame - crop_history_length)
            )
            nt = self.gen_horizon_len // f
            if noise_fn is None:
                raise ValueError("noise_fn must supply x_T (MLX cannot reproduce CUDA's RNG)")
            x_t = noise_fn(auto_step, (bs, nt, self.denoiser.nframe_root_dim + self.denoiser.latent_embedding_dim))
            if trace is not None:
                while len(trace["windows"]) <= auto_step:
                    trace["windows"].append({"index": len(trace["windows"])})
                trace["windows"][auto_step].update(
                    {
                        "index": auto_step,
                        "history_start_frame": int(history_start_frame),
                        "history_end_frame": int(history_end_frame),
                        "global_transl_in": np.asarray(global_transl).copy(),
                        "x_T": np.asarray(x_t).copy(),
                    }
                )

            history_sequence = self._generate_window(
                x_t, history_sequence, text_feat, text_pad_mask, first_heading_angle,
                num_steps, cfg_weight, num_gen_frames, history_start_frame, history_end_frame,
                ablation=ablation, trace=trace, window_index=auto_step,
            )
            offsets.append(history_sequence.shape[1] * f)

            # recenter on the last generated frame, then requantise the body latent
            root_motion, latent = self.denoiser.split_hybrid(history_sequence)
            # The reference centre is the last frame of this chunk
            # (``history_end_frame + generation_len - 1``, ardy_model.py:655-656);
            # it is already the end of the sequence, so a *positive* ablation
            # offset can only clamp back to it.
            center_frame_index = history_end_frame + self.gen_horizon_len - 1
            center_frame_index = center_frame_index + ablation.center_index_offset
            center_frame_index = min(max(center_frame_index, 0), root_motion.shape[1] - 1)
            new_root_motion, center_pos = self.motion_rep.recenter_root_motion(
                root_motion, center_frame_index, is_normalized=True, to_normalize=True,
                return_center_pos=True,
            )
            latent = self.tokenizer.requantize(latent)
            history_sequence = self.denoiser.join_hybrid(
                new_root_motion.reshape(bs, -1, self.denoiser.nframe_root_dim), latent
            )
            global_transl = global_transl + center_pos

            if crop_history_length is not None and ablation.refresh_first_heading:
                heading = self.motion_rep.get_root_heading_angle(
                    self.motion_rep.global_root_stats.unnormalize(new_root_motion)
                )
                cropped_start = max(0, new_root_motion.shape[1] - crop_history_length)
                first_heading_angle = heading[:, cropped_start]

            if trace is not None:
                w = trace["windows"][auto_step]
                w["center_frame_index"] = int(center_frame_index)
                w["center_pos"] = np.asarray(center_pos).copy()
                w["global_transl_after"] = np.asarray(global_transl).copy()
                w["history_after_recenter"] = np.asarray(history_sequence).copy()
                w["first_heading_angle_after"] = np.asarray(first_heading_angle).copy()

        # fold the accumulated translation back into the root, then detokenise
        root_motion, latent = self.denoiser.split_hybrid(history_sequence)
        new_root_motion = self.motion_rep.translate_normalized_root_motion(root_motion, global_transl)
        history_sequence = self.denoiser.join_hybrid(
            new_root_motion.reshape(bs, -1, self.denoiser.nframe_root_dim), latent
        )

        root_motion, latent = self.denoiser.split_hybrid(history_sequence)
        if crop_history_length is not None and ablation.autoregressive_decode:
            motion = self.explicit_from_hybrid_autoregressive(latent, root_motion, crop_history_length)
        else:
            motion = self.explicit_from_hybrid_whole(latent, root_motion)
        motion = motion[:, :num_frames]

        if trace is not None:
            trace["final"] = {
                "global_transl": np.asarray(global_transl).copy(),
                "hybrid_translated": np.asarray(history_sequence).copy(),
                "motion_normalized": np.asarray(motion).copy(),
                "crop_tokens": None if crop_history_length is None
                else crop_history_length // self.num_frames_per_token,
                "decode_chunks": autoregressive_chunk_slices(
                    latent.shape[1], self.gen_horizon_len // f,
                    None if crop_history_length is None else crop_history_length // f,
                ),
                "ablation": ablation,
            }

        return {
            "motion_normalized": motion,
            "frames": self.motion_rep.unnormalize(motion),
            "offsets": offsets,
        }


# --------------------------------------------------------------------------- #
# the seam for the sampling side (B)
# --------------------------------------------------------------------------- #
@dataclass
class _FallbackDenoiseResult:
    """Stand-in for ``ardy_mlx.sampling.DenoiseResult`` when that package is absent."""

    generated_frames: mx.array
    offsets: Optional[List[int]] = None


def _make_denoise_result(generated_frames: mx.array, offsets: Optional[List[int]] = None):
    """Return B's ``DenoiseResult`` when importable, else a field-identical fallback.

    Imported lazily so that ``ardy_mlx.model`` never hard-depends on
    ``ardy_mlx.sampling`` (and so B can keep iterating on that package).
    """
    try:  # pragma: no cover - depends on B's package being importable
        from ..sampling.autoregressive import DenoiseResult  # type: ignore

        return DenoiseResult(generated_frames=generated_frames, offsets=offsets)
    except Exception:
        return _FallbackDenoiseResult(generated_frames=generated_frames, offsets=offsets)


def make_step_fn(
    batch: "ArdyBatch",
    text_feat: mx.array,
    text_mask: mx.array,
    *,
    num_steps: int = 10,
    cfg_weight: Sequence[float] = (2.0, 2.0),
    crop_history_length: Optional[int] = 160,
    reuse_latent: bool = True,
) -> Callable[[object], object]:
    """Build the ``step_fn`` for B's ``ArdySampler`` (``DenoiseRequest`` -> ``DenoiseResult``).

    B drives the loop; this adapter owns everything that needs weights: the
    hybrid representation, the diffusion chain, the three-pass CFG and the
    detokenise step. The contract is read off B's own source
    (``src/ardy_mlx/sampling/autoregressive.py``); nothing here duplicates it:

    ==========================  ==============================================
    ``request`` field           what this adapter does with it
    ==========================  ==============================================
    ``plan.history_len``        clean history length in frames (must be a
                                multiple of the patch size)
    ``plan.generation_len``     horizon frames produced (must equal
                                ``batch.gen_horizon_len``)
    ``history_frames``          clean, **normalised 330-dim explicit** frames;
                                converted to the 148-dim hybrid window
    ``generation_noise``        ``x_T`` for this window, ``[B, nt, 148]``
    ``first_heading_angle``     forwarded to the denoiser as-is
    ``text_embeddings`` /       optional per-request override of the two values
    ``text_mask``               captured at construction time
    ``cfg_weight`` /            optional per-request override of the defaults
    ``num_steps``
    ``frame_index_offset``      unused (logging only)
    ==========================  ==============================================

    Returns ``DenoiseResult(generated_frames=[B, generation_len, 330])`` -- the
    newly generated chunk only, in world coordinates, **not** including history.

    ``reuse_latent`` keeps a one-slot cache of the hybrid representation this
    adapter last produced. B's loop hands back the *explicit* frames it was
    given, so re-tokenising them every step would silently replace the
    denoiser's own (requantised) body latent with an encode of the decode.
    When the incoming history is a suffix of the cached frames -- the normal
    flow -- the exact cached latent is reused instead. Set it to ``False`` to
    force the re-encode (useful to measure how much that difference costs).

    Reference-exactness: each window is computed exactly as
    ``Ardy._generate_window`` does (same masks, same ``future_token_mask = 0``
    rule of ``latent_utils.py:207``, same timestep mapping, same CFG). The one
    intentional difference is window geometry: B's ``WindowPlan`` describes
    ``history + horizon`` with no masked *future* region, so the window has
    fewer tokens than the reference's full-sequence window. With no constraints
    the future region is zero-weighted and key-padding-masked either way, so the
    generation tokens come out the same up to float32 reduction order.
    """
    state: Dict[str, Optional[mx.array]] = {"frames": None, "hybrid": None}
    f = batch.num_frames_per_token
    horizon = batch.gen_horizon_len
    gen_tokens = horizon // f

    def _hybrid_for(history_frames: mx.array) -> mx.array:
        cached_frames = state["frames"]
        cached_hybrid = state["hybrid"]
        if (
            reuse_latent
            and cached_frames is not None
            and cached_hybrid is not None
            and history_frames.shape[1] <= cached_frames.shape[1]
        ):
            tail = cached_frames[:, cached_frames.shape[1] - history_frames.shape[1] :]
            if np.array_equal(np.asarray(tail), np.asarray(history_frames)):
                nt = history_frames.shape[1] // f
                if nt > 0:
                    return cached_hybrid[:, cached_hybrid.shape[1] - nt :]
                return cached_hybrid[:, :0]
        if history_frames.shape[1] == 0:
            return mx.zeros(
                (
                    history_frames.shape[0],
                    0,
                    batch.denoiser.nframe_root_dim + batch.denoiser.latent_embedding_dim,
                )
            )
        return hybrid_from_explicit_frames(batch, history_frames)

    def step_fn(request) -> object:
        history_frames = request.history_frames
        generation_noise = request.generation_noise
        b, history_len = history_frames.shape[0], history_frames.shape[1]
        if history_len % f != 0:
            raise ValueError(f"history_len {history_len} must be a multiple of the patch size {f}")
        generation_len = int(request.plan.generation_len)
        if generation_len != horizon:
            raise ValueError(
                f"plan.generation_len {generation_len} != gen_horizon_len {horizon}; "
                "the horizon is a checkpoint property and cannot vary per step"
            )
        if generation_noise.shape[1] != gen_tokens:
            raise ValueError(
                f"generation_noise {tuple(generation_noise.shape)} must have {gen_tokens} tokens"
            )

        history_hybrid = _hybrid_for(history_frames)
        step_cfg = tuple(request.cfg_weight) if request.cfg_weight is not None else tuple(cfg_weight)
        step_num_steps = int(request.num_steps) if request.num_steps else int(num_steps)
        step_feat = request.text_embeddings if request.text_embeddings is not None else text_feat
        step_mask = request.text_mask if request.text_mask is not None else text_mask

        history_token_end = history_len // f
        generation_token_end = history_token_end + gen_tokens
        total_tokens = history_token_end + gen_tokens

        history_len_mx = mx.full((b,), float(history_len), dtype=mx.float32)
        generation_len_mx = mx.full((b,), float(horizon), dtype=mx.float32)
        future_len_mx = mx.zeros((b,), dtype=mx.float32)
        total_frames = history_len + horizon
        history_mask, generation_mask, future_mask = get_three_mask_from_len(
            history_len_mx, generation_len_mx, future_len_mx, total_frames
        )
        h_tok, g_tok, f_tok = convert_frame_mask_to_token_mask(
            history_mask, generation_mask, future_mask, motion_mask=None, num_frames_per_token=f
        )

        x = assemble_window_latent(
            history_hybrid if history_token_end > 0 else None,
            generation_noise,
            history_token_end,
            generation_token_end,
            total_tokens,
        )
        indices = timestep_indices(step_num_steps)
        for i in indices:
            use_timesteps = batch.diffusion.space_timesteps(step_num_steps)[0]
            batch.diffusion.calc_diffusion_vars(use_timesteps)
            t_map = int(batch.diffusion.map_tensor[i])
            clean = batch.cfg(
                mx.array([step_cfg[0]], dtype=mx.float32),
                mx.array([step_cfg[1]], dtype=mx.float32),
                x,
                history_len=history_len_mx,
                generation_len=generation_len_mx,
                future_len=future_len_mx,
                history_mask=history_mask,
                generation_mask=generation_mask,
                future_mask=future_mask,
                history_token_mask=h_tok,
                generation_token_mask=g_tok,
                future_token_mask=f_tok,
                text_feat=step_feat,
                text_feat_pad_mask=step_mask,
                timesteps=mx.full((b,), float(t_map)),
                first_heading_angle=request.first_heading_angle,
                motion_mask=None,
                observed_motion=None,
            )
            gen_t = x[:, history_token_end:generation_token_end]
            gen_clean = clean[:, history_token_end:generation_token_end]
            gen_next = batch.sampler(gen_t, gen_clean, i)
            x = mx.concatenate(
                [x[:, :history_token_end], gen_next, x[:, generation_token_end:]], axis=1
            )

        # The history is clean and known; the whole window is decoded so the
        # generation span sees the same decoder context the reference gives it.
        root_motion, latent = batch.denoiser.split_hybrid(x)
        local_root = batch.motion_rep.global_root_to_local_root(root_motion, normalized=True)
        body = batch.tokenizer.detokenize(latent, local_root, None)["body"]
        explicit = batch.motion_rep.concat_root_body(root_motion, body)
        generated = explicit[:, history_len : history_len + horizon]

        if reuse_latent:
            state["frames"] = mx.concatenate([history_frames, generated], axis=1)
            state["hybrid"] = x
        return _make_denoise_result(generated_frames=generated)

    def _reset():
        state["frames"] = None
        state["hybrid"] = None

    step_fn.batch = batch  # type: ignore[attr-defined]
    step_fn.reset = _reset  # type: ignore[attr-defined]
    return step_fn
