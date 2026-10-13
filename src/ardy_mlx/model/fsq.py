"""Stage A: ``FSQVAETransformer`` -- the motion tokenizer (encode/decode).

Reference: ``ardy/model/autoencoder/fsq.py`` and
``ardy/model/autoencoder/transformer.py``. Both sub-transformers are **causal**
for the released Core checkpoint (``config.yaml:38-39``: ``causal_encoder: true``,
``causal_decoder: true``), so the attention mask is causal *and* padding-masked.

Two details that silently break alignment if guessed:

1. ``latent_embedding_dim`` is **overwritten** by ``num_fsq_levels``
   (``fsq.py:85``) -- both are 128 here, but the argument is not what is used.
2. The FSQ quantiser is **not** a plain ``round(z*32)/32``. ``FSQ.bound``
   (``finite_scalar_quantization.py:124-131``) first squeezes ``z`` through
   ``tanh`` with a shift derived from ``levels`` and ``eps=1e-3``::

       half_l = (levels - 1) * (1 + eps) / 2
       offset = 0.5                       # levels is even
       shift  = atanh(offset / half_l)
       zhat   = round(tanh(z + shift) * half_l - offset) / (levels // 2)

   ``detokenize`` then re-quantises differently -- it assumes the value is
   already bounded and does ``round(clamp(x, -1, 1) * 32) / 32``
   (``fsq.py:236-238``).
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import numpy as np

from .motion_rep import MotionRep, Stats
from .nn import (
    Linear,
    PositionalEncoding,
    TransformerEncoder,
    WeightStore,
    build_additive_mask,
)

__all__ = ["FSQVAETransformer", "fsq_bound", "re_round"]

#: ``FSQ(levels=[64]*128)`` and ``FSQ.bound``'s default eps.
FSQ_LEVELS = 64
FSQ_EPS = 1e-3


def _fsq_shift(levels: int = FSQ_LEVELS, eps: float = FSQ_EPS) -> tuple[float, float]:
    """``(shift, half_l)`` from ``FSQ.bound`` (finite_scalar_quantization.py:124-129)."""
    half_l = (levels - 1) * (1 + eps) / 2.0
    offset = 0.5 if levels % 2 == 0 else 0.0
    shift = math.atanh(offset / half_l)
    return shift, half_l


def _round_ste(z: mx.array) -> mx.array:
    """``round_ste`` value behaviour (gradients are irrelevant at inference)."""
    return mx.round(z)


def fsq_bound(z: mx.array, levels: int = FSQ_LEVELS, eps: float = FSQ_EPS) -> mx.array:
    """``FSQ.bound`` at eval -- see the module docstring for the formula."""
    shift, half_l = _fsq_shift(levels, eps)
    bounded = mx.tanh(z + shift) * half_l - (0.5 if levels % 2 == 0 else 0.0)
    return _round_ste(bounded) / (levels // 2)


def re_round(z: mx.array, levels: int = FSQ_LEVELS) -> mx.array:
    """The inline re-rounding step inside ``detokenize`` (fsq.py:236-238).

    Assumes ``z`` is already *un-normalised*; ``round(clamp(z, -1, 1) * 32) / 32``.
    Note this is **not** the same as the ``requantize`` method, which wraps this
    step in the ``post_quantization`` stats.
    """
    half_width = levels // 2
    return _round_ste(mx.clip(z, -1.0, 1.0) * half_width) / half_width


def _causal_mask(seq_len: int) -> mx.array:
    """``torch.nn.Transformer.generate_square_subsequent_mask`` with ``dtype=bool``.

    ``True`` means "not allowed to attend" (the diagonal and below are ``False``).
    """
    idx = np.arange(seq_len)
    return mx.array((idx[None, :] > idx[:, None]))


class FSQVAETransformer:
    """Motion tokenizer: 4 frames <-> 1 latent token.

    Encode: ``body(325) x 4`` -> ``input_proj(512)`` -> 8 causal layers ->
    ``output_proj(128)`` -> FSQ -> ``post_quantization`` normalisation.
    Decode: unnormalise -> re-round -> ``input_proj(128->512)`` -> concat local
    root (4x4=16) -> ``external_cond_blocks(528->512)`` -> PE -> 8 causal layers
    -> ``output_proj(512->1316)`` -> ``(root 4 | body 325)`` per frame.
    """

    def __init__(self, store: WeightStore, motion_rep: MotionRep, prefix: str = "pose_net"):
        self.store = store
        self.motion_rep = motion_rep
        self.prefix = prefix
        self.num_frames_per_token = 4
        self.latent_dim = 512
        self.num_layers = 8
        self.num_heads = 4
        self.latent_embedding_dim = 128
        self._causal_encoder = True
        self._causal_decoder = True

        # --- encoder ---------------------------------------------------------
        self.enc_input_proj = Linear(store, f"{prefix}.encoder.input_proj")
        self.enc_layers = TransformerEncoder(
            store, f"{prefix}.encoder.seqTransEncoder", self.num_layers, self.latent_dim, self.num_heads
        )
        self.enc_output_proj = Linear(store, f"{prefix}.encoder.output_proj")

        # --- decoder ---------------------------------------------------------
        self.dec_input_proj = Linear(store, f"{prefix}.decoder.input_proj")
        self.dec_external_cond_blocks = Linear(store, f"{prefix}.decoder.external_cond_blocks.0")
        self.dec_layers = TransformerEncoder(
            store, f"{prefix}.decoder.seqTransEncoder", self.num_layers, self.latent_dim, self.num_heads
        )
        self.dec_output_proj = Linear(store, f"{prefix}.decoder.output_proj")

        # The reference passes no `target_cond` on this path, so
        # `target_cond_blocks` is intentionally not wired up. Flag it loudly if
        # someone later needs it: the weights are present but unused here.
        self._has_target_cond = store.has(f"{prefix}.decoder.target_cond_blocks.0.weight")

        # Shared PE tables (identical construction in both transformers).
        self._pe = PositionalEncoding(self.latent_dim)
        self.post_quantization_stats: Stats | None = None
        self.pre_quantization_stats: Stats | None = None

    def load_stats(self, stats_dir) -> "FSQVAETransformer":
        """Attach ``stats/post_quantization`` + ``stats/pre_quantization`` (fsq.py:147-152)."""
        self.post_quantization_stats = Stats.from_folder(f"{stats_dir}/post_quantization")
        self.pre_quantization_stats = Stats.from_folder(f"{stats_dir}/pre_quantization")
        return self

    # ---------------------------------------------------------------- encode
    def encode_body(self, body: mx.array, motion_pad_mask: Optional[mx.array] = None) -> mx.array:
        """Encoder transformer only: body ``[B,T,325]`` -> ``[B,T/4,128]``."""
        b, t, d = body.shape
        f = self.num_frames_per_token
        # rearrange(x, "b (t f) d -> b t (f d)")
        x = body.reshape(b, t // f, f * d)
        x = self.enc_input_proj(x)
        x = self._pe(x)
        attn_mask = _causal_mask(x.shape[1]) if self._causal_encoder else None
        kpm = None
        if motion_pad_mask is not None:
            token_pad = motion_pad_mask.reshape(b, t // f, f).all(axis=-1)
            kpm = ~token_pad
        mask = build_additive_mask(x.shape[1], kpm, attn_mask)
        x = self.enc_layers(x, mask)
        return self.enc_output_proj(x)

    def tokenize(self, features: mx.array, motion_pad_mask: mx.array) -> mx.array:
        """``FSQVAETransformer.tokenize`` (fsq.py:186-213).

        ``features`` are **normalised** 330-dim features; only the body block is
        used (``encoder_input_feature_mode == "body"``).
        """
        body = self.motion_rep.extract_body(features)
        x_encoder = self.encode_body(body, motion_pad_mask)
        token_embeddings = fsq_bound(x_encoder)
        if self.post_quantization_stats is None:
            raise RuntimeError("post_quantization stats not loaded; call load_stats()")
        return self.post_quantization_stats.normalize(token_embeddings)

    # ---------------------------------------------------------------- decode
    def decode_latent(
        self,
        token_embeddings: mx.array,
        external_cond: mx.array,
        motion_pad_mask: Optional[mx.array] = None,
    ) -> mx.array:
        """Decoder transformer only: ``[B,T/4,128]`` + local root -> ``[B,T,329]``."""
        b, num_tokens, _ = token_embeddings.shape
        f = self.num_frames_per_token
        num_frames = num_tokens * f

        h = self.dec_input_proj(token_embeddings)
        ext = external_cond.reshape(b, num_tokens, -1)
        h = mx.concatenate([h, ext], axis=-1)
        h = mx.maximum(self.dec_external_cond_blocks(h), 0.0)  # ReLU
        h = self._pe(h)

        attn_mask = _causal_mask(num_tokens) if self._causal_decoder else None
        kpm = None
        if motion_pad_mask is not None:
            token_pad = motion_pad_mask.reshape(b, num_frames // f, f).all(axis=-1)
            kpm = ~token_pad
        mask = build_additive_mask(num_tokens, kpm, attn_mask)
        h = self.dec_layers(h, mask)
        h = self.dec_output_proj(h)
        return h.reshape(b, num_frames, -1)

    def detokenize(
        self,
        token_embeddings: mx.array,
        external_cond: mx.array,
        motion_pad_mask: Optional[mx.array] = None,
    ) -> dict:
        """``FSQVAETransformer.detokenize`` (fsq.py:215-255) -> ``{"root", "body"}``."""
        stats = self.post_quantization_stats
        if stats is None:
            raise RuntimeError("post_quantization stats not loaded; call load_stats()")
        t = stats.unnormalize(token_embeddings)
        t = re_round(t)
        x_decoder = self.decode_latent(t, external_cond, motion_pad_mask)
        return {"root": x_decoder[..., : self.motion_rep.local_root_dim], "body": x_decoder[..., self.motion_rep.local_root_dim:]}

    def requantize(self, token_embeddings: mx.array) -> mx.array:
        """``FSQVAETransformer.requantize`` (fsq.py:257-278).

        Note this *is* stats-wrapped (un-normalise -> re-round -> normalise),
        unlike the inline step inside :meth:`detokenize`.
        """
        stats = self.post_quantization_stats
        if stats is None:
            raise RuntimeError("post_quantization stats not loaded; call load_stats()")
        return stats.normalize(re_round(stats.unnormalize(token_embeddings)))

    def forward(self, token_embeddings: mx.array, external_cond: mx.array = None,
                motion_pad_mask: mx.array = None) -> dict:
        """The reference rebinds ``forward = detokenize`` (fsq.py:128)."""
        return self.detokenize(token_embeddings, external_cond, motion_pad_mask)

    # -------------------------------------------------------------- round-trip
    def roundtrip(self, features: mx.array, motion_pad_mask: mx.array) -> dict:
        """encode -> decode on the same features; used by the alignment harness."""
        latent = self.tokenize(features, motion_pad_mask)
        local_root = self.motion_rep.global_root_to_local_root(
            self.motion_rep.extract_root(features), normalized=True
        )
        out = self.detokenize(latent, local_root, motion_pad_mask)
        return {"latent": latent, "root": out["root"], "body": out["body"]}
