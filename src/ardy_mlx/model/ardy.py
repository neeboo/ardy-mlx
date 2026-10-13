"""The batch generation path: ``Ardy`` -- window loop, recentring, detokenising.

Reference: ``ardy/model/ardy_model.py`` (the ``Ardy`` class) plus
``ardy/model/latent_utils.py``. This is the assembly that turns the two stages
into whole motion, and it is the thing ``scripts/ardy_align.py`` compares against
the golden clip.

Why this file exists separately from the stages: the recentring/translation
arithmetic in ``Ardy.__call__`` is easy to get subtly wrong and impossible to
test through either stage alone. For a 2.0 s Core clip (40 frames at 20 fps,
``gen_horizon_len == 40``) there is exactly **one** autoregressive window, so the
whole clip is: 10 DDIM steps -> recentre -> requantise -> translate -> detokenise.
The loop below is written for the general multi-window case, but only the
single-window path has a golden reference in this round.

The initial noise is **injected**, not sampled here: ``noise_fn(step, shape)``
must supply it. That keeps this module deterministic and lets the alignment
harness feed the reference's own ``x_T`` -- CUDA's Philox RNG is not reproducible
in MLX, so re-deriving the noise would be a separate (and much harder) project.
"""

from __future__ import annotations

from typing import Callable, Optional

import mlx.core as mx

from .denoiser import AutoLatentTwostageDenoiser, AutoLatentClassifierFreeGuidedModelSeparated
from .diffusion import DDIMSampler, Diffusion
from .fsq import FSQVAETransformer
from .motion_rep import MotionRep

__all__ = ["ArdyBatch", "get_three_mask_from_len"]


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

        history_mask, generation_mask, future_mask = get_three_mask_from_len(
            history_len, generation_len, future_len, total_frames - history_start_frame
        )
        history_token_mask = history_mask.reshape(bs, -1, f).all(axis=-1)
        generation_token_mask = generation_mask.reshape(bs, -1, f).all(axis=-1)
        future_token_mask = future_mask.reshape(bs, -1, f).all(axis=-1)

        nt = (total_frames - history_start_frame) // f
        d = x_t.shape[-1]
        x = mx.zeros((bs, nt, d))
        if history_token_end > 0:
            x = x.at[:, :history_token_end].add(history_sequence[:, -history_token_end:])
        x = x.at[:, history_token_end:generation_token_end].add(x_t)

        indices = list(range(num_steps))[::-1]
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
            gen_next = self.sampler(gen_t, gen_clean, t_map)
            x = mx.concatenate(
                [x[:, :history_token_end], gen_next, x[:, generation_token_end:]], axis=1
            )

        if history_sequence is None:
            return x[:, :generation_token_end]
        return mx.concatenate([history_sequence, x[:, history_token_end:generation_token_end]], axis=1)

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
    ) -> dict:
        """``Ardy.__call__`` (ardy_model.py:538-690) for the no-constraint batch path."""
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

        for auto_step in range(num_auto_steps):
            history_end_frame = auto_step * self.gen_horizon_len + init_history_len
            history_start_frame = (
                0 if crop_history_length is None else max(0, history_end_frame - crop_history_length)
            )
            nt = self.gen_horizon_len // f
            if noise_fn is None:
                raise ValueError("noise_fn must supply x_T (MLX cannot reproduce CUDA's RNG)")
            x_t = noise_fn(auto_step, (bs, nt, self.denoiser.nframe_root_dim + self.denoiser.latent_embedding_dim))

            history_sequence = self._generate_window(
                x_t, history_sequence, text_feat, text_pad_mask, first_heading_angle,
                num_steps, cfg_weight, num_gen_frames, history_start_frame, history_end_frame,
            )
            offsets.append(history_sequence.shape[1] * f)

            # recenter on the last generated frame, then requantise the body latent
            center_frame_index = history_end_frame + self.gen_horizon_len - 1
            root_motion, latent = self.denoiser.split_hybrid(history_sequence)
            new_root_motion, center_pos = self.motion_rep.recenter_root_motion(
                root_motion, center_frame_index, is_normalized=True, to_normalize=True,
                return_center_pos=True,
            )
            latent = self.tokenizer.requantize(latent)
            history_sequence = self.denoiser.join_hybrid(
                new_root_motion.reshape(bs, -1, self.denoiser.nframe_root_dim), latent
            )
            global_transl = global_transl + center_pos

            if crop_history_length is not None:
                heading = self.motion_rep.get_root_heading_angle(
                    self.motion_rep.global_root_stats.unnormalize(new_root_motion)
                )
                cropped_start = max(0, new_root_motion.shape[1] - crop_history_length)
                first_heading_angle = heading[:, cropped_start]

        # fold the accumulated translation back into the root, then detokenise
        root_motion, latent = self.denoiser.split_hybrid(history_sequence)
        new_root_motion = self.motion_rep.translate_normalized_root_motion(root_motion, global_transl)
        history_sequence = self.denoiser.join_hybrid(
            new_root_motion.reshape(bs, -1, self.denoiser.nframe_root_dim), latent
        )

        root_motion, latent = self.denoiser.split_hybrid(history_sequence)
        local_root = self.motion_rep.global_root_to_local_root(root_motion, normalized=True)
        decoded = self.tokenizer.detokenize(latent, local_root)
        motion = self.motion_rep.concat_root_body(root_motion, decoded["body"])
        motion = motion[:, :num_frames]

        return {
            "motion_normalized": motion,
            "frames": self.motion_rep.unnormalize(motion),
            "offsets": offsets,
        }
