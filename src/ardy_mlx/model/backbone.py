"""``TransformerEncoderBlock`` -- the denoiser's shared transformer backbone.

Reference: ``ardy/model/backbone.py``. One block is instantiated twice (once for
the root stage, once for the body stage); both share the structure but have
their own weights under ``denoiser.backbone.{root,body}_model``.

The released Core checkpoint uses
``positional_encoding_mode: learned_prefix_zero_at_first_generation``
(``config.yaml:62``), which means:

* the prefix is ``[text(1), timestep(1), first_heading_angle(1)]`` and gets a
  **learned** positional embedding (3 rows, ``learned_prefix_embedding``);
* the motion tokens get the sinusoidal table indexed by ``token_index``, which is
  *relative* to the first generation token and may be negative;
* the plain sinusoidal ``sequence_pos_encoder`` is therefore **unused** in the
  forward pass -- but its table is still the one the timestep embedder indexes.

There is **no causal mask** here: ``seqTransEncoder`` is called with only
``src_key_padding_mask`` (backbone.py:391-394), so attention is bidirectional.
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx

from .nn import (
    LearnedPositionalEncoding,
    Linear,
    PositionalEncoding,
    PositionalEncodingNegativeIndex,
    TimestepEmbedder,
    TransformerEncoder,
    WeightStore,
    build_additive_mask,
)

__all__ = ["TransformerEncoderBlock"]


class TransformerEncoderBlock:
    """``TransformerEncoderBlock`` (backbone.py:265-399).

    ``add_input_proj=False`` for both denoiser stages, so ``input_linear`` is an
    identity and the caller supplies an already-projected 1024-wide sequence.
    """

    def __init__(
        self,
        store: WeightStore,
        prefix: str,
        latent_dim: int = 1024,
        ff_size: int = 2048,
        num_layers: int = 8,
        num_heads: int = 8,
        output_dim: int = 20,
        num_text_tokens: int = 1,
        input_first_heading_angle: bool = True,
        positional_encoding_mode: str = "learned_prefix_zero_at_first_generation",
    ):
        self.prefix = prefix
        self.latent_dim = latent_dim
        self.num_text_tokens = num_text_tokens
        self.output_dim = output_dim
        self.input_first_heading_angle = input_first_heading_angle
        self.positional_encoding_mode = positional_encoding_mode

        self.embed_text = Linear(store, f"{prefix}.embed_text")
        self.sequence_pos_encoder = PositionalEncoding(latent_dim)
        self.embed_timestep = TimestepEmbedder(store, f"{prefix}.embed_timestep", latent_dim)
        self.output_linear = Linear(store, f"{prefix}.output_linear")

        if input_first_heading_angle:
            self.linear_first_heading_angle = Linear(store, f"{prefix}.linear_first_heading_angle")

        if positional_encoding_mode == "learned_prefix_zero_at_first_generation":
            prefix_length = num_text_tokens + 1
            if input_first_heading_angle:
                prefix_length += 1
            self.prefix_length = prefix_length
            self.learned_prefix_embedding = LearnedPositionalEncoding(
                store, f"{prefix}.learned_prefix_embedding.embedding", max_len=prefix_length
            )
            self.motion_token_embedding = PositionalEncodingNegativeIndex(latent_dim)
        elif positional_encoding_mode == "default":
            self.prefix_length = None
        else:
            raise ValueError(f"unsupported positional_encoding_mode: {positional_encoding_mode!r}")

        self.seqTransEncoder = TransformerEncoder(
            store, f"{prefix}.seqTransEncoder", num_layers, latent_dim, num_heads
        )

    def __call__(
        self,
        x: mx.array,
        x_pad_mask: mx.array,
        text_feat: mx.array,
        text_feat_pad_mask: mx.array,
        timesteps: mx.array,
        first_heading_angle: Optional[mx.array] = None,
        token_index: Optional[mx.array] = None,
    ) -> mx.array:
        """``TransformerEncoderBlock.forward`` (backbone.py:300-396)."""
        batch_size = x.shape[0]

        text_feat = text_feat[:, : self.num_text_tokens]
        emb_text = self.embed_text(text_feat)          # [B, 1, D]
        emb_time = self.embed_timestep(timesteps)      # [B, 1, D]
        time_mask = mx.ones((batch_size, 1), dtype=mx.bool_)

        prefix_feats = mx.concatenate([emb_text, emb_time], axis=1)
        # use_text_mask == False -> every text token is unmasked
        text_feat_pad_mask = mx.ones((batch_size, emb_text.shape[1]), dtype=mx.bool_)
        prefix_mask = mx.concatenate([text_feat_pad_mask, time_mask], axis=1)

        if self.input_first_heading_angle:
            angle = first_heading_angle
            feats = mx.stack([mx.cos(angle), mx.sin(angle)], axis=-1)   # [B, 2]
            feats = self.linear_first_heading_angle(feats)[:, None]     # [B, 1, D]
            prefix_feats = mx.concatenate([prefix_feats, feats], axis=1)
            prefix_mask = mx.concatenate(
                [prefix_mask, mx.ones((batch_size, 1), dtype=mx.bool_)], axis=1
            )

        pose_start_ind = prefix_feats.shape[1]

        if self.positional_encoding_mode == "default":
            xseq = self.sequence_pos_encoder(mx.concatenate([prefix_feats, x], axis=1))
        else:
            prefix_feats = self.learned_prefix_embedding(prefix_feats)
            x_pe = self.motion_token_embedding(x, token_index)
            xseq = mx.concatenate([prefix_feats, x_pe], axis=1)

        src_key_padding_mask = ~mx.concatenate([prefix_mask, x_pad_mask], axis=1)
        mask = build_additive_mask(xseq.shape[1], src_key_padding_mask, None)
        output = self.seqTransEncoder(xseq, mask)
        output = output[:, pose_start_ind:]
        return self.output_linear(output)
