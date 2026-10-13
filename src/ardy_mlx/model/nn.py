"""Neural-network primitives for the ARDY MLX port.

Every primitive here mirrors exactly one PyTorch module used by the reference
implementation, because the goal is numerical alignment, not a nice API. Where
PyTorch's default is surprising, the docstring says so and cites the reference
file:line.

Conventions
-----------
* All tensors are ``mx.array`` in float32, laid out ``[B, T, D]``.
* ``Linear`` weights are stored in PyTorch's ``[out_features, in_features]``
  layout, so the forward is ``x @ W.T + b`` with no transposes at load time.
* Masks follow PyTorch's ``src_key_padding_mask`` convention: ``True`` means
  "ignore this position".
"""

from __future__ import annotations

import math
from typing import Optional

import mlx.core as mx
import numpy as np

__all__ = [
    "WeightStore",
    "Linear",
    "LayerNorm",
    "gelu",
    "silu",
    "sinusoidal_pe_table",
    "PositionalEncoding",
    "PositionalEncodingNegativeIndex",
    "LearnedPositionalEncoding",
    "TimestepEmbedder",
    "sdpa",
    "TransformerEncoderLayer",
    "TransformerEncoder",
    "build_additive_mask",
]


class WeightStore:
    """Read-only view over a converted MLX checkpoint.

    Keys are the *reference* state-dict names (``pose_net.encoder...`` /
    ``denoiser.backbone...``), so a missing or renamed tensor fails loudly at
    lookup time instead of silently becoming zeros.
    """

    def __init__(self, arrays: dict):
        self._arrays = arrays
        self._used: set[str] = set()

    @classmethod
    def load(cls, path) -> "WeightStore":
        return cls(dict(mx.load(str(path))))

    def get(self, name: str) -> mx.array:
        if name not in self._arrays:
            raise KeyError(f"checkpoint has no tensor {name!r}")
        self._used.add(name)
        return self._arrays[name]

    def has(self, name: str) -> bool:
        return name in self._arrays

    def names(self) -> list[str]:
        return sorted(self._arrays)

    def unused(self) -> list[str]:
        """Tensors never looked up -- a non-empty list means the port is incomplete."""
        return sorted(set(self._arrays) - self._used)


class Linear:
    """``nn.Linear``: ``y = x @ W.T + b`` (weights kept in PyTorch layout)."""

    def __init__(self, store: WeightStore, prefix: str, bias: bool = True):
        self.weight = store.get(f"{prefix}.weight")
        self.bias = store.get(f"{prefix}.bias") if bias else None
        self.in_features = int(self.weight.shape[1])
        self.out_features = int(self.weight.shape[0])

    def __call__(self, x: mx.array) -> mx.array:
        y = x @ self.weight.T
        return y + self.bias if self.bias is not None else y


class LayerNorm:
    """``nn.LayerNorm``: biased variance, ``eps=1e-5`` by default.

    PyTorch normalises by ``sqrt(var + eps)`` with the *biased* variance
    (divide by N, not N-1) and then applies ``weight``/``bias``.
    """

    def __init__(self, store: WeightStore, prefix: str, eps: float = 1e-5):
        self.weight = store.get(f"{prefix}.weight")
        self.bias = store.get(f"{prefix}.bias")
        self.eps = eps

    def __call__(self, x: mx.array) -> mx.array:
        mean = mx.mean(x, axis=-1, keepdims=True)
        centred = x - mean
        var = mx.mean(centred * centred, axis=-1, keepdims=True)
        return centred / mx.sqrt(var + self.eps) * self.weight + self.bias


def gelu(x: mx.array) -> mx.array:
    """Exact (erf-based) GELU -- ``torch.nn.functional.gelu``'s default.

    Deliberately *not* the tanh approximation (which ``mx.nn.gelu`` may use).
    """
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def silu(x: mx.array) -> mx.array:
    """``torch.nn.SiLU`` == ``x * sigmoid(x)``."""
    return x * mx.sigmoid(x)


def sinusoidal_pe_table(d_model: int, max_len: int = 5000) -> np.ndarray:
    """The reference ``PositionalEncoding.pe`` table, built in float32.

    Mirrors ``ardy/model/backbone.py:414-431`` (and the identical copy in
    ``ardy/model/autoencoder/transformer.py``)::

        position = arange(max_len, float32)[:, None]
        div_term = pow(10000.0, -arange(0, d_model, 2, float32) / d_model)
        pe[:, 0::2] = sin(position * div_term)
        pe[:, 1::2] = cos(position * div_term)

    The reference comment notes that ``torch.pow`` is used instead of
    ``exp``/``log`` for MKL reasons; float32 throughout is what makes this
    reproduce the reference table rather than merely approximate it.
    """
    position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.power(
        np.float32(10000.0),
        -np.arange(0, d_model, 2, dtype=np.float32) / np.float32(d_model),
    )
    pe = np.zeros((max_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def _negative_pe_table(d_model: int, max_len: int = 5000) -> np.ndarray:
    """``PositionalEncodingNegativeIndex.pe`` -- [2*max_len-1, d_model].

    Layout: rows ``0..max_len-1`` are the non-negative table; the rest are the
    reversed negative table, so a Python-style negative index ``-k`` addresses
    ``pe[len - k]`` (backbone.py:520-540).
    """
    pe = sinusoidal_pe_table(d_model, max_len)
    position = np.arange(0, max_len, dtype=np.float32).reshape(-1, 1)
    div_term = np.power(
        np.float32(10000.0),
        -np.arange(0, d_model, 2, dtype=np.float32) / np.float32(d_model),
    )
    pe_negative = np.zeros((max_len - 1, d_model), dtype=np.float32)
    pe_negative[:, 0::2] = np.sin(-position[1:] * div_term)
    pe_negative[:, 1::2] = np.cos(-position[1:] * div_term)
    return np.concatenate([pe, np.flip(pe_negative, axis=0)], axis=0)


class PositionalEncoding:
    """Non-learned sinusoidal PE added to a ``[B, T, D]`` sequence."""

    def __init__(self, d_model: int, max_len: int = 5000, cache: dict | None = None):
        key = ("pe", d_model, max_len)
        if cache is not None and key in cache:
            table = cache[key]
        else:
            table = sinusoidal_pe_table(d_model, max_len)
            if cache is not None:
                cache[key] = table
        self.pe = mx.array(table)  # [max_len, D]

    def __call__(self, x: mx.array) -> mx.array:
        return x + self.pe[: x.shape[1]][None]


class PositionalEncodingNegativeIndex(PositionalEncoding):
    """Sinusoidal PE whose indices may be negative (backbone.py:482-556)."""

    def __init__(self, d_model: int, max_len: int = 5000):
        self.max_len = max_len
        self.d_model = d_model
        self.pe = mx.array(_negative_pe_table(d_model, max_len))  # [2*max_len-1, D]

    def __call__(self, x: mx.array, index: mx.array) -> mx.array:
        safe = mx.where(index >= 0, index, index + self.pe.shape[0])
        return x + self.pe[safe.astype(mx.int32)]


class LearnedPositionalEncoding:
    """``nn.Embedding(max_len, d_model)`` added position-wise (backbone.py:463-480)."""

    def __init__(self, store: WeightStore, prefix: str, max_len: int):
        self.weight = store.get(f"{prefix}.weight")  # [max_len, d_model]
        self.max_len = max_len

    def __call__(self, x: mx.array) -> mx.array:
        positions = mx.arange(x.shape[1], dtype=mx.int32)
        return x + self.weight[positions][None]


class TimestepEmbedder:
    """``TimestepEmbedder`` (backbone.py:439-461).

    Subtle and easy to get wrong: the timestep is **not** embedded
    sinusoidally from its value. ``F.embedding(timesteps.int(), pe_table)``
    *looks up row ``t`` of the sinusoidal table*, and that row is then passed
    through ``Linear -> SiLU -> Linear``. ``timesteps`` is therefore an integer
    index into ``[0, num_base_steps)``, not a continuous noise level.
    """

    def __init__(self, store: WeightStore, prefix: str, latent_dim: int, cache: dict | None = None):
        self.linear1 = Linear(store, f"{prefix}.time_embed.0")
        self.linear2 = Linear(store, f"{prefix}.time_embed.2")
        self.pe = PositionalEncoding(latent_dim, cache=cache).pe  # [max_len, D]

    def __call__(self, timesteps: mx.array) -> mx.array:
        emb = self.pe[timesteps.astype(mx.int32)]  # [B, D]
        return self.linear2(silu(self.linear1(emb)))[:, None]  # [B, 1, D]


def build_additive_mask(
    seq_len: int,
    key_padding_mask: Optional[mx.array] = None,
    attn_mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """Replicate ``SDPATransformerEncoderLayer._build_attn_mask`` (backbone.py:111-139).

    Returns an additive float mask broadcasting to ``[B, nhead, L, S]`` with
    ``-inf`` at ignored positions. Boolean masks use the PyTorch convention:
    ``True == ignore``. The padding mask contributes ``[B, 1, 1, S]`` and the
    attention mask ``[L, S]``, exactly as in the reference.
    """
    if key_padding_mask is None and attn_mask is None:
        return None
    parts = []
    if key_padding_mask is not None:
        kpm = key_padding_mask
        add = mx.where(kpm, mx.full(kpm.shape, float("-inf"), mx.float32), mx.zeros(kpm.shape, mx.float32))
        parts.append(add[:, None, None, :])
    if attn_mask is not None:
        add = mx.where(attn_mask, mx.full(attn_mask.shape, float("-inf"), mx.float32),
                       mx.zeros(attn_mask.shape, mx.float32))
        parts.append(add)
    out = parts[0]
    for extra in parts[1:]:
        out = out + extra
    return out


def sdpa(q: mx.array, k: mx.array, v: mx.array, mask: Optional[mx.array]) -> mx.array:
    """Scaled dot-product attention, ``[B, H, L, D]`` inputs.

    Written out explicitly rather than delegating to
    ``mx.fast.scaled_dot_product_attention`` so the mask semantics and the
    ``1/sqrt(head_dim)`` scaling are unambiguous.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = (q @ k.transpose(0, 1, 3, 2)) * scale
    if mask is not None:
        scores = scores + mask
    weights = mx.softmax(scores, axis=-1)
    return weights @ v


class TransformerEncoderLayer:
    """``SDPATransformerEncoderLayer`` (backbone.py:58-175).

    Parameter names are shared with ``nn.TransformerEncoderLayer``; the fused
    ``in_proj_weight`` ``[3*D, D]`` is chunked into q, k, v exactly as PyTorch's
    ``MultiheadAttention`` does.
    """

    def __init__(self, store: WeightStore, prefix: str, d_model: int, nhead: int, norm_first: bool = False):
        self.d_model = d_model
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.norm_first = norm_first
        self.in_proj_weight = store.get(f"{prefix}.self_attn.in_proj_weight")
        self.in_proj_bias = store.get(f"{prefix}.self_attn.in_proj_bias")
        self.out_proj_weight = store.get(f"{prefix}.self_attn.out_proj.weight")
        self.out_proj_bias = store.get(f"{prefix}.self_attn.out_proj.bias")
        self.linear1 = Linear(store, f"{prefix}.linear1")
        self.linear2 = Linear(store, f"{prefix}.linear2")
        self.norm1 = LayerNorm(store, f"{prefix}.norm1")
        self.norm2 = LayerNorm(store, f"{prefix}.norm2")

    def _sa_block(self, x: mx.array, mask: Optional[mx.array]) -> mx.array:
        b, seq_len, _ = x.shape
        qkv = x @ self.in_proj_weight.T + self.in_proj_bias
        q, k, v = mx.split(qkv, 3, axis=-1)
        q = q.reshape(b, seq_len, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        k = k.reshape(b, seq_len, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        v = v.reshape(b, seq_len, self.nhead, self.head_dim).transpose(0, 2, 1, 3)
        attn = sdpa(q, k, v, mask)
        attn = attn.transpose(0, 2, 1, 3).reshape(b, seq_len, self.d_model)
        return attn @ self.out_proj_weight.T + self.out_proj_bias

    def _ff_block(self, x: mx.array) -> mx.array:
        return self.linear2(gelu(self.linear1(x)))

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        if self.norm_first:
            x = x + self._sa_block(self.norm1(x), mask)
            x = x + self._ff_block(self.norm2(x))
            return x
        x = self.norm1(x + self._sa_block(x, mask))
        x = self.norm2(x + self._ff_block(x))
        return x


class TransformerEncoder:
    """``SDPATransformerEncoder``: N identical layers, optional final norm."""

    def __init__(self, store: WeightStore, prefix: str, num_layers: int, d_model: int, nhead: int,
                 norm_first: bool = False):
        self.layers = [
            TransformerEncoderLayer(store, f"{prefix}.layers.{i}", d_model, nhead, norm_first)
            for i in range(num_layers)
        ]

    def __call__(self, x: mx.array, mask: Optional[mx.array] = None) -> mx.array:
        for layer in self.layers:
            x = layer(x, mask)
        return x
