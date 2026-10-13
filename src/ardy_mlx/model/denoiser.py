"""Stage B: ``AutoLatentTwostageDenoiser`` -- one denoising step.

Reference: ``ardy/model/auto_latent_twostage_denoiser.py`` and
``ardy/model/cfg.py``.

The input and output are both the **hybrid token** of width 148::

    token = [ global root block, flattened over 4 frames (20) | body latent (128) ]

Stage 1 predicts the global root for every token; its output is converted to a
*local* root (which re-normalises through ``local_root_stats`` -- see
``motion_rep.global_root_to_local_root``) and concatenated with the body latent
to form the stage-2 input.

Conditioning is by **prefix concatenation**, not cross-attention: the backbone
prepends ``[text, timestep, first-heading-angle]`` tokens. Classifier-free
guidance runs the *same* network three times batched (``cfg_type: separated``):

    pass 0: real text,  zero constraints
    pass 1: zero text,  real constraints
    pass 2: zero text,  zero constraints
    out = uncond + w_text * (text - uncond) + w_cstr * (cstr - uncond)

With no constraints at all (the batch path with no ``--constraints``), the
``observed_motion`` / ``motion_mask`` tensors are all zeros, so the constraint
projections are still *evaluated* -- they are simply fed zeros. Do not skip them:
they are part of the compute graph that produced the reference numbers.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .backbone import TransformerEncoderBlock
from .motion_rep import MotionRep
from .nn import Linear, WeightStore

__all__ = ["AutoLatentTwostageDenoiser", "AutoLatentClassifierFreeGuidedModelSeparated"]


class AutoLatentTwostageDenoiser:
    """``AutoLatentTwostageDenoiser`` (auto_latent_twostage_denoiser.py:40-400)."""

    def __init__(
        self,
        store: WeightStore,
        motion_rep: MotionRep,
        prefix: str = "denoiser",
        latent_dim: int = 1024,
        num_frames_per_token: int = 4,
        nframe_root_dim: int = 20,
        latent_embedding_dim: int = 128,
        num_layers: int = 8,
        num_heads: int = 8,
        ff_size: int = 2048,
        num_text_tokens: int = 1,
        positional_encoding_mode: str = "learned_prefix_zero_at_first_generation",
        motion_mask_mode: str = "concat",
        input_first_heading_angle: bool = True,
    ):
        self.store = store
        self.motion_rep = motion_rep
        self.prefix = prefix
        self.latent_dim = latent_dim
        self.num_frames_per_token = num_frames_per_token
        self.nframe_root_dim = nframe_root_dim
        self.latent_embedding_dim = latent_embedding_dim
        self.nframe_local_root_dim = motion_rep.local_root_dim * num_frames_per_token
        self.motion_mask_mode = motion_mask_mode
        self.positional_encoding_mode = positional_encoding_mode

        global_root_hybrid_token_dim = nframe_root_dim + latent_embedding_dim            # 148
        local_root_hybrid_token_dim = self.nframe_local_root_dim + latent_embedding_dim  # 144

        self.global_root_hybrid_proj = Linear(store, f"{prefix}.global_root_hybrid_proj")
        self.global_root_hybrid_constraints_proj = Linear(
            store, f"{prefix}.global_root_hybrid_constraints_proj"
        )
        self.local_root_hybrid_proj = Linear(store, f"{prefix}.local_root_hybrid_proj")
        self.local_root_hybrid_constraints_proj = Linear(
            store, f"{prefix}.local_root_hybrid_constraints_proj"
        )
        self.future_constraints_proj = Linear(store, f"{prefix}.future_constraints_proj")

        common = dict(
            store=store,
            latent_dim=latent_dim,
            ff_size=ff_size,
            num_layers=num_layers,
            num_heads=num_heads,
            num_text_tokens=num_text_tokens,
            input_first_heading_angle=input_first_heading_angle,
            positional_encoding_mode=positional_encoding_mode,
        )
        self.root_model = TransformerEncoderBlock(
            prefix=f"{prefix}.backbone.root_model", output_dim=nframe_root_dim, **common
        )
        self.body_model = TransformerEncoderBlock(
            prefix=f"{prefix}.backbone.body_model", output_dim=latent_embedding_dim, **common
        )

        # sanity: the projection widths encode the layout, so a mismatch means
        # the checkpoint is not the one this port was written for.
        ref = motion_rep
        exp_gr = global_root_hybrid_token_dim + (ref.motion_rep_dim + ref.body_dim) * num_frames_per_token
        exp_lr = local_root_hybrid_token_dim + (ref.motion_rep_dim + ref.body_dim) * num_frames_per_token
        exp_fc = ref.motion_rep_dim * 2 * num_frames_per_token
        if self.global_root_hybrid_constraints_proj.in_features != exp_gr:
            raise ValueError(
                f"global_root_hybrid_constraints_proj expects {exp_gr} inputs, checkpoint has "
                f"{self.global_root_hybrid_constraints_proj.in_features}"
            )
        if self.local_root_hybrid_constraints_proj.in_features != exp_lr:
            raise ValueError(
                f"local_root_hybrid_constraints_proj expects {exp_lr} inputs, checkpoint has "
                f"{self.local_root_hybrid_constraints_proj.in_features}"
            )
        if self.future_constraints_proj.in_features != exp_fc:
            raise ValueError(
                f"future_constraints_proj expects {exp_fc} inputs, checkpoint has "
                f"{self.future_constraints_proj.in_features}"
            )

    # ------------------------------------------------------------------ split
    def split_hybrid(self, hybrid: mx.array):
        """``get_root_and_latent_body_motion_from_hybrid`` (latent_utils.py:67-76)."""
        b, num_tokens, _ = hybrid.shape
        num_frames = num_tokens * self.num_frames_per_token
        root = hybrid[:, :, : self.nframe_root_dim].reshape(b, num_frames, self.motion_rep.root_block_dim)
        latent = hybrid[:, :, self.nframe_root_dim :]
        return root, latent

    @staticmethod
    def join_hybrid(root_flat: mx.array, latent: mx.array) -> mx.array:
        """``get_hybrid_motion_from_root_and_latent_body_motion`` (latent_utils.py:143-157)."""
        return mx.concatenate([root_flat, latent], axis=-1)

    # ---------------------------------------------------------------- forward
    def __call__(
        self,
        x: mx.array,
        history_len: mx.array,
        generation_len: mx.array,
        future_len: mx.array,
        history_mask: mx.array,
        generation_mask: mx.array,
        future_mask: mx.array,
        history_token_mask: mx.array,
        generation_token_mask: mx.array,
        future_token_mask: mx.array,
        text_feat: mx.array,
        text_feat_pad_mask: mx.array,
        timesteps: mx.array,
        first_heading_angle: Optional[mx.array] = None,
        motion_mask: Optional[mx.array] = None,
        observed_motion: Optional[mx.array] = None,
    ) -> mx.array:
        """``AutoLatentTwostageDenoiser.forward`` (auto_latent_twostage_denoiser.py:147-400)."""
        bs, num_tokens, dim_token = x.shape
        f = self.num_frames_per_token
        num_frames = num_tokens * f
        mr = self.motion_rep

        global_root_motion, latent_body_motion = self.split_hybrid(x)  # [B,T,5], [B,nt,128]

        token_index = mx.broadcast_to(mx.arange(num_tokens)[None, :], (bs, num_tokens))
        if self.positional_encoding_mode == "learned_prefix_zero_at_first_generation":
            origin_index = history_len // f
            token_index = token_index - origin_index[:, None]

        # ---- constraint tensors; the reference substitutes zeros when absent
        if motion_mask is None or observed_motion is None:
            motion_mask = mx.zeros((bs, num_frames, mr.motion_rep_dim))
            observed_motion = mx.zeros((bs, num_frames, mr.motion_rep_dim))

        observed_root = mr.extract_root(observed_motion)
        observed_body = mr.extract_body(observed_motion)
        root_motion_mask = mr.extract_root(motion_mask)
        global_root_motion = global_root_motion * (1 - root_motion_mask) + observed_root * root_motion_mask

        root_flat = global_root_motion.reshape(bs, num_tokens, self.nframe_root_dim)
        x_infilled_extended = mx.concatenate(
            [
                latent_body_motion,
                root_flat,
                observed_body.reshape(bs, num_tokens, -1),
                motion_mask.reshape(bs, num_tokens, -1),
            ],
            axis=-1,
        )

        future_token_proj = self.future_constraints_proj(
            x_infilled_extended[:, :, self.latent_embedding_dim :]
        ) * future_token_mask[:, :, None]
        root_stage_input = (
            self.global_root_hybrid_constraints_proj(x_infilled_extended)
            * generation_token_mask[:, :, None]
            + self.global_root_hybrid_proj(x) * history_token_mask[:, :, None]
            + future_token_proj
        )
        root_stage_pad_mask = history_token_mask | generation_token_mask | future_token_mask

        global_root_motion_pred = self.root_model(
            root_stage_input,
            root_stage_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            token_index,
        )  # [B, nt, 20]
        global_root_motion_pred = global_root_motion_pred.reshape(bs, num_frames, mr.root_block_dim)
        global_root_motion_pred = (
            global_root_motion_pred * generation_mask[:, :, None]
            + global_root_motion * history_mask[:, :, None]
        )
        lengths = history_len + generation_len

        # ---- stage 1 -> stage 2: convert the predicted global root to a local root
        local_root_motion = mr.global_root_to_local_root(
            global_root_motion_pred, normalized=True
        )  # normalised 4-vector
        local_root_flat = local_root_motion.reshape(bs, num_tokens, self.nframe_local_root_dim)
        x_new = mx.concatenate([local_root_flat, latent_body_motion], axis=-1)  # 144

        x_new_extended = mx.concatenate(
            [
                x_new,
                observed_body.reshape(bs, num_tokens, -1),
                motion_mask.reshape(bs, num_tokens, -1),
            ],
            axis=-1,
        )
        body_stage_input = (
            self.local_root_hybrid_constraints_proj(x_new_extended)
            * generation_token_mask[:, :, None]
            + self.local_root_hybrid_proj(x_new) * history_token_mask[:, :, None]
            + future_token_proj
        )
        body_stage_pad_mask = history_token_mask | generation_token_mask | future_token_mask

        latent_body_motion_pred = self.body_model(
            body_stage_input,
            body_stage_pad_mask,
            text_feat,
            text_feat_pad_mask,
            timesteps,
            first_heading_angle,
            token_index,
        )
        latent_body_motion_pred = (
            latent_body_motion_pred * generation_token_mask[:, :, None]
            + latent_body_motion * history_token_mask[:, :, None]
        )

        return mx.concatenate(
            [global_root_motion_pred.reshape(bs, num_tokens, self.nframe_root_dim), latent_body_motion_pred],
            axis=-1,
        )

    # ---------------------------------------------- single call, no CFG batching
    def forward_batch3(self, x: mx.array, **kwargs) -> mx.array:
        """Run the three CFG passes as one batch of 3, exactly as the reference does.

        ``x`` is ``[B, nt, 148]``; the returned tensor is ``[3B, nt, 148]``.
        """
        x_3 = mx.concatenate([x, x, x], axis=0)
        triple = lambda t: mx.concatenate([t, t, t], axis=0)  # noqa: E731
        text_feat_3 = mx.concatenate([kwargs["text_feat"], mx.zeros_like(kwargs["text_feat"]),
                                      mx.zeros_like(kwargs["text_feat"])], axis=0)
        text_mask_3 = mx.concatenate(
            [kwargs["text_feat_pad_mask"], mx.zeros_like(kwargs["text_feat_pad_mask"]),
             mx.zeros_like(kwargs["text_feat_pad_mask"])],
            axis=0,
        )
        heading = kwargs.get("first_heading_angle")
        mm = kwargs.get("motion_mask")
        om = kwargs.get("observed_motion")
        return self.__call__(
            x_3,
            triple(kwargs["history_len"]),
            triple(kwargs["generation_len"]),
            triple(kwargs["future_len"]),
            triple(kwargs["history_mask"]),
            triple(kwargs["generation_mask"]),
            triple(kwargs["future_mask"]),
            triple(kwargs["history_token_mask"]),
            triple(kwargs["generation_token_mask"]),
            mx.concatenate([mx.zeros_like(kwargs["future_token_mask"]), kwargs["future_token_mask"],
                            mx.zeros_like(kwargs["future_token_mask"])], axis=0),
            text_feat_3,
            text_mask_3,
            triple(kwargs["timesteps"]),
            triple(heading) if heading is not None else None,
            mx.concatenate([mx.zeros_like(mm), mm, mx.zeros_like(mm)], axis=0) if mm is not None else None,
            mx.concatenate([mx.zeros_like(om), om, mx.zeros_like(om)], axis=0) if om is not None else None,
        )


class AutoLatentClassifierFreeGuidedModelSeparated:
    """``AutoLatentClassifierFreeGuidedModelSeparated`` (cfg.py:22-149)."""

    def __init__(self, model: AutoLatentTwostageDenoiser):
        self.model = model

    def __call__(self, cfg_weight_text, cfg_weight_cstr, x, *args, **kwargs):
        """Reproduces the CFG wrapper's ``forward`` including the 1-value weight case."""
        x_3 = mx.concatenate([x, x, x], axis=0)
        kwargs = dict(kwargs)
        kwargs["text_feat"] = mx.concatenate(
            [kwargs["text_feat"], mx.zeros_like(kwargs["text_feat"]), mx.zeros_like(kwargs["text_feat"])],
            axis=0,
        )
        kwargs["text_feat_pad_mask"] = mx.concatenate(
            [kwargs["text_feat_pad_mask"], mx.zeros_like(kwargs["text_feat_pad_mask"]),
             mx.zeros_like(kwargs["text_feat_pad_mask"])], axis=0,
        )
        for key in ("history_len", "generation_len", "future_len", "timesteps"):
            kwargs[key] = mx.concatenate([kwargs[key]] * 3, axis=0)
        for key in ("history_mask", "generation_mask", "future_mask",
                    "history_token_mask", "generation_token_mask"):
            kwargs[key] = mx.concatenate([kwargs[key]] * 3, axis=0)
        ftm = kwargs["future_token_mask"]
        kwargs["future_token_mask"] = mx.concatenate(
            [mx.zeros_like(ftm), ftm, mx.zeros_like(ftm)], axis=0
        )
        heading = kwargs.get("first_heading_angle")
        if heading is not None:
            kwargs["first_heading_angle"] = mx.concatenate([heading] * 3, axis=0)
        mm, om = kwargs.get("motion_mask"), kwargs.get("observed_motion")
        if mm is not None:
            kwargs["motion_mask"] = mx.concatenate([mx.zeros_like(mm), mm, mx.zeros_like(mm)], axis=0)
        if om is not None:
            kwargs["observed_motion"] = mx.concatenate([mx.zeros_like(om), om, mx.zeros_like(om)], axis=0)

        out_3 = self.model(x_3, *args, **kwargs)
        out_text, out_cstr, out_uncond = mx.split(out_3, 3, axis=0)
        return (
            out_uncond
            + cfg_weight_text * (out_text - out_uncond)
            + cfg_weight_cstr * (out_cstr - out_uncond)
        )
