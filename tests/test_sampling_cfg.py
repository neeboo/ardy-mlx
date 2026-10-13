"""Tests for ``separated`` three-pass classifier-free guidance (contract S4.4).

The composition is a linear formula, so most of the value here is in pinning the
**order** of the three passes and the expansion ``B -> 3B``. A swapped pass order
would still produce the right shapes and would only show up as wrong numbers, so
the order is asserted structurally (not just numerically).

Reference: ``cfg.py:12-19`` (class), ``cfg.py:74-77`` (B -> 3B), ``cfg.py:148-149``
(composition), ``generate.py:90-96`` and ``:176-181`` (CLI weights).
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import DEFAULT_CFG_WEIGHT, TEXT_FEAT_DIM  # noqa: E402
from ardy_mlx.sampling.cfg import (  # noqa: E402
    CFG_PASS_ORDER,
    CfgWeights,
    apply_separated_cfg,
    make_three_pass_batch,
    split_three_pass,
)


def _text(batch: int = 2, length: int = 3) -> mx.array:
    values = [
        [[float(100 * b + 10 * l + d) for d in range(TEXT_FEAT_DIM)] for l in range(length)]
        for b in range(batch)
    ]
    return mx.array(values, dtype=mx.float32)


def _constraint(batch: int = 2, horizon: int = 4, width: int = 329) -> mx.array:
    values = [
        [[float(-1000 * b - 10 * t - d) for d in range(width)] for t in range(horizon)]
        for b in range(batch)
    ]
    return mx.array(values, dtype=mx.float32)


def test_pass_order_is_the_contract_order():
    assert CFG_PASS_ORDER == ("text_only", "constraint_only", "unconditional")


def test_three_pass_batch_expands_b_by_three_in_pass_order():
    text = _text()
    mask = mx.array([[True, True, False], [True, False, False]])
    constraint = _constraint()

    text_3b, mask_3b, cstr_3b = make_three_pass_batch(text, mask, constraint)

    assert text_3b.shape == (6, 3, TEXT_FEAT_DIM)
    assert mask_3b.shape == (6, 3)
    assert cstr_3b.shape == (6, 4, 329)

    # Pass 0: real text, zero constraint
    assert text_3b[0:2].tolist() == text.tolist()
    assert mask_3b[0:2].tolist() == mask.tolist()
    assert all(v == 0.0 for row in cstr_3b[0:2].tolist() for t in row for v in t)

    # Pass 1: zero text, real constraint
    assert all(v == 0.0 for row in text_3b[2:4].tolist() for l in row for v in l)
    assert all(not v for row in mask_3b[2:4].tolist() for v in row)
    assert cstr_3b[2:4].tolist() == constraint.tolist()

    # Pass 2: zero text, zero constraint
    assert all(v == 0.0 for row in text_3b[4:6].tolist() for l in row for v in l)
    assert all(v == 0.0 for row in cstr_3b[4:6].tolist() for t in row for v in t)

    # The three passes must genuinely differ, otherwise the test above is vacuous.
    assert text_3b[0:2].tolist() != text_3b[2:4].tolist()
    assert cstr_3b[0:2].tolist() != cstr_3b[2:4].tolist()


def test_three_pass_batch_accepts_explicit_null_conditioning():
    text = _text(1, 2)
    mask = mx.ones((1, 2), dtype=mx.bool_)
    constraint = _constraint(1, 4, 8)
    null_text = mx.full(text.shape, 7.0, dtype=mx.float32)
    null_cstr = mx.full(constraint.shape, -3.0, dtype=mx.float32)

    text_3b, _mask_3b, cstr_3b = make_three_pass_batch(
        text, mask, constraint, zero_text=null_text, zero_constraint=null_cstr
    )
    # Pass order is text_only, constraint_only, unconditional: the null text
    # replaces passes 1 and 2, the null constraint replaces passes 0 and 2.
    assert text_3b[1:2].tolist() == null_text.tolist()
    assert text_3b[2:3].tolist() == null_text.tolist()
    assert cstr_3b[0:1].tolist() == null_cstr.tolist()
    assert cstr_3b[2:3].tolist() == null_cstr.tolist()


def test_three_pass_batch_rejects_bad_shapes():
    text = _text(2, 3)
    mask = mx.ones((2, 3), dtype=mx.bool_)
    constraint = _constraint(2, 4, 8)
    with pytest.raises(ValueError):
        make_three_pass_batch(text, mask, _constraint(3, 4, 8))  # batch mismatch
    with pytest.raises(ValueError):
        make_three_pass_batch(text, mx.ones((2, 5), dtype=mx.bool_), constraint)  # mask length
    with pytest.raises(ValueError):
        make_three_pass_batch(text, mask, mx.zeros((2, 4)))  # constraint must be 3-D


def test_split_three_pass_returns_contiguous_chunks():
    out = mx.arange(6 * 2 * 3, dtype=mx.float32).reshape(6, 2, 3)
    text, cstr, uncond = split_three_pass(out)
    assert text.shape == cstr.shape == uncond.shape == (2, 2, 3)
    assert text.tolist() == out[0:2].tolist()
    assert cstr.tolist() == out[2:4].tolist()
    assert uncond.tolist() == out[4:6].tolist()
    with pytest.raises(ValueError):
        split_three_pass(mx.zeros((5, 2, 3)))


def test_composition_matches_the_contract_formula():
    # cfg.py:148-149
    out_text = mx.array([[[1.0, 2.0], [3.0, 4.0]]], dtype=mx.float32)
    out_cstr = mx.array([[[10.0, 20.0], [30.0, 40.0]]], dtype=mx.float32)
    out_uncond = mx.array([[[0.5, 0.5], [0.5, 0.5]]], dtype=mx.float32)
    out_3b = mx.concatenate([out_text, out_cstr, out_uncond], axis=0)

    weights = CfgWeights.from_arg([2.0, 2.0])
    got = apply_separated_cfg(out_3b, weights)

    expected = [
        [
            [
                u + 2.0 * (t - u) + 2.0 * (c - u)
                for t, c, u in zip(row_t, row_c, row_u)
            ]
            for row_t, row_c, row_u in zip(
                out_text[0].tolist(), out_cstr[0].tolist(), out_uncond[0].tolist()
            )
        ]
    ]
    assert got.shape == (1, 2, 2)
    np.testing.assert_allclose(got.tolist(), expected, rtol=0, atol=1e-6)

    # text_weight=1, constraint_weight=0 must reproduce the text pass exactly.
    np.testing.assert_allclose(
        apply_separated_cfg(out_3b, [1.0, 0.0]).tolist(), out_text.tolist(), rtol=0, atol=1e-6
    )
    # weight 0 on both must reproduce the unconditional pass exactly.
    np.testing.assert_allclose(
        apply_separated_cfg(out_3b, [0.0, 0.0]).tolist(), out_uncond.tolist(), rtol=0, atol=1e-6
    )


def test_swapping_text_and_constraint_passes_changes_the_result():
    """Order matters: the test above would be blind to a swapped layout."""
    out_text = mx.array([[[1.0]]], dtype=mx.float32)
    out_cstr = mx.array([[[5.0]]], dtype=mx.float32)
    out_uncond = mx.array([[[0.0]]], dtype=mx.float32)
    ordered = apply_separated_cfg(
        mx.concatenate([out_text, out_cstr, out_uncond], axis=0), [2.0, 1.0]
    )
    swapped = apply_separated_cfg(
        mx.concatenate([out_cstr, out_text, out_uncond], axis=0), [2.0, 1.0]
    )
    assert ordered.tolist() == [[[7.0]]]
    assert swapped.tolist() == [[[11.0]]]
    # 2*1 + 1*5 = 7 vs 2*5 + 1*1 = 11: the numeric result identifies the layout.
    assert ordered.tolist() != swapped.tolist()


def test_cfg_weight_normalisation_rules():
    # generate.py:90-96 -- default is the two-value form.
    assert DEFAULT_CFG_WEIGHT == (2.0, 2.0)
    assert CfgWeights.from_arg(DEFAULT_CFG_WEIGHT).as_tuple() == (2.0, 2.0)
    assert CfgWeights.from_arg([2.0, 2.0]).as_list() == [2.0, 2.0]
    # generate.py:176-181 -- a single value is the text weight, constraint 0.0.
    assert CfgWeights.from_arg(3.5).as_tuple() == (3.5, 0.0)
    assert CfgWeights.from_arg([3.5]).as_tuple() == (3.5, 0.0)
    with pytest.raises(ValueError):
        CfgWeights.from_arg([1.0, 2.0, 3.0])
    with pytest.raises(ValueError):
        apply_separated_cfg(mx.zeros((3, 1, 1)), None)
    with pytest.raises(ValueError):
        apply_separated_cfg(mx.zeros((4, 1, 1)), [1.0, 1.0])
