"""``separated`` classifier-free guidance (contract S4.4).

The released checkpoints all use ``cfg_type: separated`` (``config.yaml:3``),
i.e. ``AutoLatentClassifierFreeGuidedModelSeparated`` (``cfg.py:12-19``). The
batch is expanded ``B -> 3B`` and split into three passes (``cfg.py:74-77``)::

    Pass 0: text-only        (real text, zero constraint)
    Pass 1: constraint-only  (zero text, real constraint)
    Pass 2: unconditional    (zero text, zero constraint)

and composed (``cfg.py:148-149``)::

    out = out_uncond
        + cfg_weight_text * (out_text  - out_uncond)
        + cfg_weight_cstr * (out_cstr  - out_uncond)

Defaults: ``--cfg_weight = [2.0, 2.0]`` (``generate.py:90-96``). A single value
means "text weight only, constraint weight 0.0" (``generate.py:176-181``,
``ardy_model.py:272-277``); ``io.DEFAULT_CFG_WEIGHT`` is the two-value form.

How the model actually builds the *zeroed* text and constraint inputs is inside
the denoiser, which is A's side of the boundary. This module only owns the batch
layout and the arithmetic, so B can test the composition in isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

import mlx.core as mx

#: Order of the three passes inside the 3B batch (cfg.py:74-77). Fixed: the
#: composition reads the chunks back in exactly this order.
CFG_PASS_ORDER: Tuple[str, str, str] = ("text_only", "constraint_only", "unconditional")

__all__ = [
    "CFG_PASS_ORDER",
    "CfgWeights",
    "apply_separated_cfg",
    "make_three_pass_batch",
    "split_three_pass",
]


def _normalize_cfg_weight(
    cfg_weight: Optional[Union[float, int, Sequence[float]]]
) -> Tuple[float, float]:
    """``--cfg_weight`` parsing rules (generate.py:90-96,176-181).

    * ``None``                 -> (2.0, 2.0)? No: ``None`` -> (1.0,) is *not* a
      thing here; callers pass either a scalar, a 2-sequence, or nothing.
    * scalar ``w``             -> ``(w, 0.0)``  (text weight only)
    * ``[w_text, w_cstr]``     -> as given
    """
    if cfg_weight is None:
        raise ValueError(
            "cfg_weight must be a scalar or a 2-sequence; use io.DEFAULT_CFG_WEIGHT for the default"
        )
    if isinstance(cfg_weight, (int, float)):
        return (float(cfg_weight), 0.0)
    values = tuple(float(v) for v in cfg_weight)
    if len(values) == 1:
        return (values[0], 0.0)
    if len(values) != 2:
        raise ValueError(
            f"cfg_weight must have 1 or 2 entries (text, constraint), got {len(values)}"
        )
    return (values[0], values[1])


@dataclass(frozen=True)
class CfgWeights:
    """``(text_weight, constraint_weight)`` as used by the composition."""

    text: float
    constraint: float

    @classmethod
    def from_arg(
        cls, cfg_weight: Optional[Union[float, int, Sequence[float]]]
    ) -> "CfgWeights":
        text, cstr = _normalize_cfg_weight(cfg_weight)
        return cls(text=text, constraint=cstr)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.text, self.constraint)

    def as_list(self) -> list:
        """For the sidecar JSON (contract S7 writes ``"cfg_weight": [2.0, 2.0]``)."""
        return [self.text, self.constraint]


def make_three_pass_batch(
    text_embeddings: mx.array,
    text_mask: mx.array,
    constraint: mx.array,
    zero_text: Optional[mx.array] = None,
    zero_constraint: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Tile ``(text, mask, constraint)`` into the ``B -> 3B`` separated layout.

    Returns ``(text_3b, mask_3b, constraint_3b)``, each with leading dimension
    ``3B`` in ``CFG_PASS_ORDER`` order.

    ``zero_text`` / ``zero_constraint`` default to zeros of the right shape. The
    reference builds the null conditioning inside the model, so passing explicit
    tensors keeps the placeholder backend and the real model on the same
    interface; when A wires the real denoiser it may ignore these and use its own
    null embedding, which is why they are optional rather than required.
    """
    if text_embeddings.ndim != 3:
        raise ValueError(
            f"text_embeddings must be [B, L, TEXT_FEAT_DIM], got shape {text_embeddings.shape}"
        )
    if text_mask.ndim != 2:
        raise ValueError(f"text_mask must be [B, L], got shape {text_mask.shape}")
    if text_mask.shape[:2] != text_embeddings.shape[:2]:
        raise ValueError(
            f"text_mask shape {text_mask.shape} does not match text_embeddings {text_embeddings.shape}"
        )
    if constraint.ndim != 3:
        raise ValueError(f"constraint must be [B, T, C], got shape {constraint.shape}")
    if constraint.shape[0] != text_embeddings.shape[0]:
        raise ValueError(
            f"batch mismatch: constraint {constraint.shape[0]} vs text {text_embeddings.shape[0]}"
        )

    batch = text_embeddings.shape[0]
    null_text = (
        mx.zeros_like(text_embeddings) if zero_text is None else zero_text
    )
    null_constraint = mx.zeros_like(constraint) if zero_constraint is None else zero_constraint
    if null_text.shape != text_embeddings.shape:
        raise ValueError(
            f"zero_text shape {null_text.shape} must equal text_embeddings shape {text_embeddings.shape}"
        )
    if null_constraint.shape != constraint.shape:
        raise ValueError(
            f"zero_constraint shape {null_constraint.shape} must equal constraint shape {constraint.shape}"
        )

    texts = [text_embeddings, null_text, null_text]
    masks = [text_mask, mx.zeros_like(text_mask), mx.zeros_like(text_mask)]
    constraints = [null_constraint, constraint, null_constraint]

    text_3b = mx.concatenate(texts, axis=0)
    mask_3b = mx.concatenate(masks, axis=0)
    constraint_3b = mx.concatenate(constraints, axis=0)
    expected = 3 * batch
    if text_3b.shape[0] != expected:
        raise AssertionError(f"three-pass batch should have {expected} rows, got {text_3b.shape[0]}")
    return text_3b, mask_3b, constraint_3b


def split_three_pass(out_3b: mx.array, batch_size: Optional[int] = None) -> Tuple[mx.array, mx.array, mx.array]:
    """Split a ``3B`` output into ``(out_text, out_cstr, out_uncond)`` (cfg.py:148)."""
    total = out_3b.shape[0]
    if total % 3 != 0:
        raise ValueError(
            f"three-pass output leading dim must be divisible by 3, got {total}"
        )
    batch = total // 3
    if batch_size is not None and batch != batch_size:
        raise ValueError(f"expected batch {batch_size}, got {batch} from leading dim {total}")
    return (
        out_3b[0 * batch : 1 * batch],
        out_3b[1 * batch : 2 * batch],
        out_3b[2 * batch : 3 * batch],
    )


def apply_separated_cfg(
    out_3b: mx.array,
    weights: Union[CfgWeights, Sequence[float], float],
) -> mx.array:
    """Compose the three passes back into one ``B``-sized output (cfg.py:148-149).

    ``out_uncond + w_text * (out_text - out_uncond) + w_cstr * (out_cstr - out_uncond)``
    """
    if not isinstance(weights, CfgWeights):
        weights = CfgWeights.from_arg(weights)
    out_text, out_cstr, out_uncond = split_three_pass(out_3b)
    return (
        out_uncond
        + weights.text * (out_text - out_uncond)
        + weights.constraint * (out_cstr - out_uncond)
    )
