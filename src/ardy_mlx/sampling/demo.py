"""Placeholder backend: shape-faithful, semantically meaningless.

**What this is.** A random-weight stand-in for the two-stage denoiser and for the
tokenizer's decoder, sized exactly like the contract describes, so the sampler,
the autoregressive loop, the CLI and the on-disk layout can be built and tested
*before* A's converted weights land (``src/ardy_mlx/model/**``).

**What this is not.** It is not the model. Its weights are drawn with
``mx.random`` and are never trained, so:

* the generated motion is **meaningless** -- do not read anything into it;
* it does **not** verify numerical equivalence with the reference; only
  ``scripts/ardy_align.py`` (A's side) does that, against the golden outputs;
* three contract transformations are *stubbed*, not implemented, and are
  recorded as open in ``docs/status-claude.md``: ``global_root_to_local_root``
  (S3.3 stage 1 -> stage 2 bridge), the frozen FSQ encode/decode path (S3.2),
  and the ``requantize=True`` history re-quantization after recentring (S5.4).

Shapes it does honour (contract S3.2/S3.3 plus ``io``):

===========================  ================================================
hybrid token                 148 = 20 (global root, 5 x 4) + 128 (body latent)
denoiser input / output      ``[B, T/4, 148]``, identical shape
body stage input             144 = 16 (local root) + 128 (body latent)
detokenized frames           330 (Core) / 414 (G1), ``io.BLOCK_ORDER`` order
===========================  ================================================
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx

from ..io import LOCAL_ROOT_DIM, NUM_FRAMES_PER_TOKEN, TEXT_FEAT_DIM, FeatureLayout
from .autoregressive import DenoiseRequest, DenoiseResult
from .cfg import CfgWeights, apply_separated_cfg, make_three_pass_batch
from .diffusion import DDIMSampler, get_diffusion_schedule

__all__ = [
    "DEMO_HIDDEN_DIM",
    "DEMO_PREFIX_TOKENS",
    "DemoDetokenizer",
    "DemoHybridDenoiser",
    "DemoStepRunner",
    "make_demo_denoise_fn",
]

#: Denoiser transformer width (config.yaml:53-56) and text projection width
#: (backbone.py:321: ``embed_text = Linear(4096 -> 1024)``).
DEMO_HIDDEN_DIM = 1024

#: Learnable prefix tokens (backbone.py: ``learned_prefix_embedding`` is [3,1024]).
DEMO_PREFIX_TOKENS = 3

#: Local-root slots in the body stage input (contract S3.3: 16 = 4 x 4).
DEMO_LOCAL_ROOT_TOKENS_DIM = LOCAL_ROOT_DIM * NUM_FRAMES_PER_TOKEN


def _init(shape, key, scale: float = 0.02) -> mx.array:
    return (mx.random.normal(shape, key=key) * scale).astype(mx.float32)


@dataclass
class DemoHybridDenoiser:
    """Random-weight stand-in for ``AutoLatentTwostageDenoiser`` (contract S3.3).

    ``mode="dense"`` uses the real hidden width (1024); ``mode="fast"`` shrinks it
    so a test run stays instant. Both accept and return ``[B, T/4, 148]``.
    """

    layout: FeatureLayout
    mode: str = "fast"
    seed: int = 0

    def __post_init__(self) -> None:
        if self.mode not in ("fast", "dense"):
            raise ValueError(f"mode must be 'fast' or 'dense', got {self.mode!r}")
        hidden = DEMO_HIDDEN_DIM if self.mode == "dense" else 64
        self.hidden = hidden
        token_dim = self.layout.dim_token
        # One independent key per parameter: reusing a single key would make every
        # "random" matrix identical (MLX keys are deterministic generators).
        keys = mx.random.split(mx.random.key(self.seed), 7)
        self.embed_text = _init((hidden, TEXT_FEAT_DIM), keys[0])
        # backbone.py:342-361: linear_first_heading_angle = Linear(2 -> hidden)
        self.embed_heading = _init((hidden, 2), keys[1])
        # auto_latent_twostage_denoiser.py:87-90 / :184-186
        self.root_hybrid_proj = _init((hidden, token_dim), keys[2])
        # Root output is the flattened global-root block only (20 dims, = 5 x 4),
        # not the whole 148-wide token (contract S3.3).
        self.root_out = _init((self.layout.nframe_root_dim, hidden), keys[3])
        self.local_root_hybrid_proj = _init((hidden, DEMO_LOCAL_ROOT_TOKENS_DIM + 128), keys[4])
        self.body_out = _init((128, hidden), keys[5])
        self.prefix = _init((DEMO_PREFIX_TOKENS, hidden), keys[6])

    def __call__(
        self,
        x_t: mx.array,
        first_heading_angle: mx.array,
        text_embeddings: Optional[mx.array],
        text_mask: Optional[mx.array],
    ) -> mx.array:
        """Predict clean hybrid tokens for a ``3B`` batch (same shape as ``x_t``)."""
        if x_t.ndim != 3 or x_t.shape[-1] != self.layout.dim_token:
            raise ValueError(f"x_t must be [B, T/4, {self.layout.dim_token}], got {x_t.shape}")
        root_dim = self.layout.nframe_root_dim
        latent_body = x_t[..., root_dim:]

        if text_embeddings is None:
            prefix = mx.zeros((x_t.shape[0], self.hidden), dtype=mx.float32) + self.prefix[0]
        else:
            if text_embeddings.shape[-1] != TEXT_FEAT_DIM:
                raise ValueError(
                    f"text_embeddings last dim must be {TEXT_FEAT_DIM}, got {text_embeddings.shape}"
                )
            if text_mask is not None:
                mask = text_mask.astype(mx.float32)[..., None]  # [3B, L, 1]
                pooled = mx.sum(text_embeddings * mask, axis=1) / mx.maximum(
                    mx.sum(mask, axis=1), 1.0
                )
            else:
                pooled = mx.mean(text_embeddings, axis=1)
            prefix = mx.matmul(pooled, self.embed_text.T)
        # Heading enters every pass the same way it enters the real backbone.
        prefix = prefix + mx.matmul(first_heading_angle.reshape(-1, 2), self.embed_heading.T)

        root_pred = self._root_stage(x_t, prefix)
        latent_pred = self._body_stage(root_pred, latent_body)
        return mx.concatenate([root_pred, latent_pred], axis=-1)

    def _root_stage(self, x_t: mx.array, prefix: mx.array) -> mx.array:
        """Stage 1: predict the global root block (``[B, T/4, 20]``).

        ``x_t`` is the full hybrid token (``[B, T/4, 148]``), matching
        ``global_root_hybrid_proj``'s ``[1024, 148]`` (contract S3.3).
        ``prefix`` is ``[B, hidden]`` (the projected text prefix token).
        """
        h = mx.matmul(x_t, self.root_hybrid_proj.T)
        h = h + prefix.reshape(prefix.shape[0], 1, prefix.shape[1])
        return mx.matmul(h, self.root_out.T)

    def _body_stage(self, root_pred: mx.array, latent_body: mx.array) -> mx.array:
        """Stage 2: refine the body latent given the predicted root.

        The real bridge is ``global_root_to_local_root`` (contract S3.3); the
        stand-in reuses the first 16 dims of the root prediction because that has
        the right width. Recorded as open in ``docs/status-claude.md``.
        """
        local_root = root_pred[..., :DEMO_LOCAL_ROOT_TOKENS_DIM]
        x_new = mx.concatenate([local_root, latent_body], axis=-1)
        h = mx.matmul(x_new, self.local_root_hybrid_proj.T)
        return mx.matmul(h, self.body_out.T)


@dataclass
class DemoDetokenizer:
    """Random-weight stand-in for the tokenizer decoder (contract S3.2).

    Maps ``[B, T/4, 148]`` back to per-frame features with the contract's block
    order. The global root block is passed through from the token's root part
    (the root side is *conditioning*, not decoder output, in the reference), and
    the body part is a fixed linear map -- so the block layout and the shapes are
    exact while the values are meaningless.
    """

    layout: FeatureLayout
    mode: str = "fast"
    seed: int = 1

    def __post_init__(self) -> None:
        if self.mode not in ("fast", "dense"):
            raise ValueError(f"mode must be 'fast' or 'dense', got {self.mode!r}")
        token_dim = self.layout.dim_token
        # The contract's ``frames`` row is the global root block plus ``body_dim``
        # (contract S1.2/S1.3: 5 + 325 = 330). The tokenizer decoder nominally
        # produces ``local_motion_rep_dim`` = local root (4) + body (325) and the
        # local-root channels are consumed by the root conditioning; since the
        # placeholder does not implement that path, it emits ``body_dim`` directly.
        body_dim = self.layout.body_dim
        if self.mode == "dense":
            width = body_dim * NUM_FRAMES_PER_TOKEN
        else:
            width = body_dim
        self.width = width
        self.proj = _init((width, token_dim), mx.random.key(self.seed))

    def __call__(self, hybrid: mx.array) -> mx.array:
        if hybrid.ndim != 3 or hybrid.shape[-1] != self.layout.dim_token:
            raise ValueError(
                f"hybrid must be [B, T/4, {self.layout.dim_token}], got {hybrid.shape}"
            )
        batch, tokens, _ = hybrid.shape
        body = mx.matmul(hybrid, self.proj.T)  # [B, T/4, width]
        if self.mode == "dense":
            body = body.reshape(batch, tokens * NUM_FRAMES_PER_TOKEN, -1)
        else:
            body = mx.repeat(body, NUM_FRAMES_PER_TOKEN, axis=1)
        if body.shape[-1] != self.layout.body_dim:
            raise ValueError(
                f"detokenized body width {body.shape[-1]} != body_dim {self.layout.body_dim}"
            )
        # The root block is already one block per frame (5 dims); only the generation
        # side is flattened over the patch. The placeholder gives every frame in a
        # token the same root, which is why fast mode repeats rather than reshapes.
        root_tok = hybrid[..., : self.layout.nframe_root_dim].reshape(
            batch, tokens, NUM_FRAMES_PER_TOKEN, self.layout.root_block_dim
        )
        root = mx.mean(root_tok, axis=2)
        root = mx.repeat(root, NUM_FRAMES_PER_TOKEN, axis=1)
        root = root[:, : body.shape[1], :]
        return mx.concatenate([root, body], axis=-1)


def sanitize_placeholder_output(frames: mx.array, layout: FeatureLayout) -> mx.array:
    """Force the placeholder output into the contract's value domains (S1.1).

    The reference's output is a valid motion representation by construction; a
    random-weight stand-in is not. Two domains are enforced here so that anything
    saved by the placeholder backend is at least *self-consistent*:

    * ``global_root_heading`` is renormalised to a unit ``(cos, sin)``;
    * ``foot_contacts`` is thresholded to exact 0/1.

    This is a **placeholder-only** transform. It is not part of the contract and
    must not be applied to real model output.
    """
    if frames.ndim != 3 or frames.shape[-1] != layout.motion_rep_dim:
        raise ValueError(
            f"frames must be [B, T, {layout.motion_rep_dim}], got {frames.shape}"
        )
    # Heading: renormalise the (cos, sin) pair in place. An interior-slice
    # concatenate mis-reports its width on MLX 0.32.3, so this uses .at[] plus a
    # literal 5 / tail split (same workaround as window.shift_root_xy).
    head = frames[..., 0:5]
    norm = mx.sqrt(mx.sum(head[..., 3:5] * head[..., 3:5], axis=-1, keepdims=True))
    head = head.at[..., 3:5].multiply(1.0 / mx.maximum(norm, 1e-6))
    result = mx.concatenate([head, frames[..., 5:]], axis=-1)

    # foot_contacts is the last block: 4 dims at the very end (io.BLOCK_ORDER).
    body = result[..., 5:]
    contacts = body[..., -4:]
    binary = mx.where(contacts > 0.5, mx.ones_like(contacts), mx.zeros_like(contacts))
    body = mx.concatenate([body[..., :-4], binary], axis=-1)
    result = mx.concatenate([result[..., 0:5], body], axis=-1)
    if result.shape != frames.shape:
        raise AssertionError(
            f"sanitize changed the shape: {frames.shape} -> {result.shape}"
        )
    return result


@dataclass
class DemoStepRunner:
    """One autoregressive step: CFG -> DDIM chain -> detokenize (contract S3/S4).

    The history frames from the request are encoded to tokens, concatenated with
    the noisy generation tokens, and **only the generation positions are
    replaced** by the denoised result (``ardy_model.py:302-313``); the clean
    history frames are then copied back verbatim.
    """

    layout: FeatureLayout
    denoiser: DemoHybridDenoiser
    detokenizer: DemoDetokenizer

    def __call__(self, request: DenoiseRequest) -> DenoiseResult:
        batch = request.history_frames.shape[0]
        horizon = request.generation_len
        patch = NUM_FRAMES_PER_TOKEN
        if horizon % patch != 0:
            raise ValueError(f"generation_len {horizon} must be a multiple of {patch}")
        gen_tokens = horizon // patch

        history_tokens = self.encode_history(request.history_frames)
        history_len = history_tokens.shape[1]

        text = (
            request.text_embeddings
            if request.text_embeddings is not None
            else mx.zeros((batch, 1, TEXT_FEAT_DIM), dtype=mx.float32)
        )
        mask = (
            request.text_mask
            if request.text_mask is not None
            else mx.ones((batch, 1), dtype=mx.bool_)
        )
        # The constraint conditioning is the model's business (A's side); the
        # placeholder passes zeros of the local-motion width, which is the width
        # the denoiser's constraint projection consumes (contract S1.3 item 3).
        constraint = mx.zeros((batch, horizon, self.layout.local_motion_rep_dim), dtype=mx.float32)
        text_3b, mask_3b, _ = make_three_pass_batch(text, mask, constraint)

        x_t = request.generation_noise
        if x_t.shape != (batch, gen_tokens, self.layout.dim_token):
            raise ValueError(
                f"generation_noise must be {(batch, gen_tokens, self.layout.dim_token)}, "
                f"got {x_t.shape}"
            )
        schedule = get_diffusion_schedule(request.num_steps)
        ddim = DDIMSampler(schedule)
        weights = CfgWeights.from_arg(request.cfg_weight)
        # The backbone projects a 2-vector (cos, sin) per batch element
        # (backbone.py:342-361); only the angle is known here, so the pair is
        # (angle, 0). Passes are replicated along the batch axis, matching the
        # three-pass layout of the text and constraint tensors.
        heading_pair = mx.stack(
            [request.first_heading_angle, mx.zeros_like(request.first_heading_angle)], axis=-1
        )
        heading_3b = mx.concatenate([heading_pair] * 3, axis=0)

        for t in schedule.timesteps:
            window = x_t if history_len == 0 else mx.concatenate([history_tokens, x_t], axis=1)
            full = mx.concatenate([window] * 3, axis=0)
            pred = apply_separated_cfg(
                self.denoiser(full, heading_3b, text_3b, mask_3b), weights
            )
            # ``pred`` covers the whole window (history + generation); the diffusion
            # update only applies to the generation tokens (contract S5.3).
            gen_tokens = x_t.shape[1]
            window_tokens = pred.shape[1]
            x_t = ddim.step(x_t, pred[:, window_tokens - gen_tokens :, :], t)

        hybrid = x_t if history_len == 0 else mx.concatenate([history_tokens, x_t], axis=1)
        decoded = self.detokenizer(hybrid)
        generated = decoded[:, request.history_len :, :]
        if generated.shape[1] != horizon:
            generated = generated[:, :horizon, :]
        return DenoiseResult(generated_frames=generated)

    def encode_history(self, frames: mx.array) -> mx.array:
        """Stand-in encoder for the clean history (contract S3.2's frozen encoder).

        Deterministic and parameter-free: the root block is flattened verbatim and
        the latent part is the body block averaged over each ``patch`` (truncated
        to 128 channels). The real path is the frozen FSQVAE encoder, which A owns.
        """
        batch, num_frames, _ = frames.shape
        if num_frames == 0:
            return mx.zeros((batch, 0, self.layout.dim_token), dtype=mx.float32)
        if num_frames % NUM_FRAMES_PER_TOKEN != 0:
            raise ValueError(
                f"history length {num_frames} must be a multiple of {NUM_FRAMES_PER_TOKEN}"
            )
        tokens = num_frames // NUM_FRAMES_PER_TOKEN
        root = frames[..., : self.layout.root_block_dim].reshape(
            batch, tokens, self.layout.nframe_root_dim
        )
        body = frames[..., self.layout.root_block_dim :].reshape(
            batch, tokens, NUM_FRAMES_PER_TOKEN, -1
        )
        body = mx.mean(body, axis=2)
        latent = mx.zeros((batch, tokens, 128), dtype=mx.float32)
        keep = min(128, body.shape[-1])
        latent = mx.concatenate([body[..., :keep], latent[..., keep:]], axis=-1)
        return mx.concatenate([root, latent], axis=-1)


def make_demo_denoise_fn(
    layout: FeatureLayout,
    mode: str = "fast",
    seed: int = 0,
) -> DemoStepRunner:
    """Build the placeholder step runner used by the CLI and the end-to-end test."""
    return DemoStepRunner(
        layout=layout,
        denoiser=DemoHybridDenoiser(layout=layout, mode=mode, seed=seed),
        detokenizer=DemoDetokenizer(layout=layout, mode=mode, seed=seed + 1),
    )
