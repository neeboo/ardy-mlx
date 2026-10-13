"""MLX port of the ARDY motion model (A side: weights, forward, alignment).

Layout
------
``nn``
    Primitives that mirror specific PyTorch modules used by the reference
    (``Linear``, ``LayerNorm``, exact-erf GELU, explicit scaled-dot-product
    attention, sinusoidal positional encodings, ``WeightStore``).
``motion_rep``
    Feature block slicing, normalisation statistics, and the global-root ->
    local-root conversion.
``fsq``
    **Stage A**: ``FSQVAETransformer`` -- the motion tokenizer (encode/decode).
``backbone``
    ``TransformerEncoderBlock``, the denoiser's prefix-conditioned backbone.
``denoiser``
    **Stage B**: ``AutoLatentTwostageDenoiser`` plus the ``separated``
    three-pass classifier-free guidance wrapper.
``diffusion``
    The hardcoded cosine beta schedule and the deterministic DDIM step.
``ardy``
    The batch generation path: window loop, recentring, translation, detokenise.

Where the numbers come from
---------------------------
Every non-obvious constant carries the reference ``file:line`` it was read from
(``nv-tlabs/ardy`` @ ``693f74d``). The API contract with the sampling side (B) is
``docs/motion-contract.md``; do not invent values here.

Numerical status
----------------
``scripts/ardy_align.py`` compares this port against the single authorised
reference run on the Core / 20 fps / Horizon40 checkpoint. Measured worst-case
absolute residues (float32 on both sides, different hardware):

* tokenizer ``encode -> decode`` round trip: ``<= 7.0e-06``
* denoiser single step (all 10 DDIM steps): ``<= 5.4e-05``
* whole 40-frame clip (``frames (T, 330)``): ``<= 3.9e-05``
"""

from __future__ import annotations

__all__ = [
    "ArdyBatch",
    "AutoLatentClassifierFreeGuidedModelSeparated",
    "AutoLatentTwostageDenoiser",
    "DDIMSampler",
    "Diffusion",
    "FSQVAETransformer",
    "MotionRep",
    "Stats",
    "TransformerEncoderBlock",
    "WeightStore",
]


def __getattr__(name: str):
    """Import submodules lazily so ``import ardy_mlx.model`` stays cheap."""
    if name == "ArdyBatch":
        from .ardy import ArdyBatch

        return ArdyBatch
    if name in ("AutoLatentTwostageDenoiser", "AutoLatentClassifierFreeGuidedModelSeparated"):
        from .denoiser import (
            AutoLatentClassifierFreeGuidedModelSeparated,
            AutoLatentTwostageDenoiser,
        )

        return {
            "AutoLatentTwostageDenoiser": AutoLatentTwostageDenoiser,
            "AutoLatentClassifierFreeGuidedModelSeparated": AutoLatentClassifierFreeGuidedModelSeparated,
        }[name]
    if name in ("DDIMSampler", "Diffusion"):
        from .diffusion import DDIMSampler, Diffusion

        return {"DDIMSampler": DDIMSampler, "Diffusion": Diffusion}[name]
    if name == "FSQVAETransformer":
        from .fsq import FSQVAETransformer

        return FSQVAETransformer
    if name in ("MotionRep", "Stats"):
        from .motion_rep import MotionRep, Stats

        return {"MotionRep": MotionRep, "Stats": Stats}[name]
    if name == "TransformerEncoderBlock":
        from .backbone import TransformerEncoderBlock

        return TransformerEncoderBlock
    if name == "WeightStore":
        from .nn import WeightStore

        return WeightStore
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
