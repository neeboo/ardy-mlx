"""Numerical tests for the cosine schedule and the DDIM step.

Every expected value is recomputed here **from the contract's formula**
(``docs/motion-contract.md`` S4.1/S4.2), not copied from the implementation, so
the test fails if either side drifts. The last test in the file is an explicit
negative control: it shows that a plausible transcription slip in the cosine
formula is caught by the first test.

Reference: ``model/diffusion.py:12-26`` (schedule), ``:113-133`` (DDIM).
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import mlx.core as mx
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import NUM_BASE_STEPS  # noqa: E402
from ardy_mlx.sampling.diffusion import (  # noqa: E402
    COSINE_SCHEDULE_OFFSET,
    COSINE_SCHEDULE_SCALE,
    MAX_BETA,
    DDIMSampler,
    alpha_bar,
    cosine_beta_schedule,
    get_diffusion_schedule,
    resolve_num_steps,
    space_timesteps,
)

TOL = 1e-6


def scalar(value: mx.array) -> float:
    return float(mx.reshape(value, (1,))[0])


# --------------------------------------------------------------------------
# contract S4.1 -- cosine beta schedule
# --------------------------------------------------------------------------
def test_alpha_bar_matches_the_contract_formula_pointwise():
    """``alpha_bar(t) = cos((t + 0.008) / 1.008 * pi/2) ** 2`` (diffusion.py:14-15)."""
    for t in (0.0, 1e-6, 0.008, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0):
        expected = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        assert alpha_bar(t) == pytest.approx(expected, abs=1e-15)
    # The constants themselves are part of the formula.
    assert COSINE_SCHEDULE_OFFSET == 0.008
    assert COSINE_SCHEDULE_SCALE == 1.008


def test_betas_match_an_independent_recomputation():
    """``betas[i] = min(1 - alpha_bar(t2)/alpha_bar(t1), 0.999)`` (diffusion.py:18-26)."""
    n = NUM_BASE_STEPS
    betas = cosine_beta_schedule(n)

    def reference_alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2

    assert len(betas) == n
    for i, got in enumerate(betas):
        t1, t2 = i / n, (i + 1) / n
        expected = min(1 - reference_alpha_bar(t2) / reference_alpha_bar(t1), 0.999)
        assert got == pytest.approx(expected, abs=1e-15)
        assert got <= MAX_BETA


def test_schedule_tables_are_internally_consistent():
    sched = get_diffusion_schedule(None)
    assert sched.num_steps == NUM_BASE_STEPS == 10  # contract S4.3

    betas = [float(v) for v in sched.base.betas]
    cumprod = []
    acc = 1.0
    for b in betas:
        acc *= 1.0 - b
        cumprod.append(acc)

    for i, (got, expected) in enumerate(zip([float(v) for v in sched.base.alphas_cumprod], cumprod)):
        assert got == pytest.approx(expected, abs=1e-6), f"alphas_cumprod[{i}]"
    # First cumulative product is alpha_0 = 1 - beta_0 and must stay below 1.
    assert 0.0 < cumprod[0] < 1.0
    assert cumprod == sorted(cumprod, reverse=True), "alpha_bar must be monotonically decreasing"


def test_alphas_cumprod_prev_is_a_shift_with_a_leading_one():
    sched = get_diffusion_schedule(None)
    prev = [float(v) for v in sched.base.alphas_cumprod_prev]
    alpha = [float(v) for v in sched.base.alphas_cumprod]
    assert prev[0] == pytest.approx(1.0, abs=1e-6)
    for i in range(1, len(alpha)):
        assert prev[i] == pytest.approx(alpha[i - 1], abs=1e-6)


def test_reciprocal_tables_match_their_definitions():
    sched = get_diffusion_schedule(None)
    alpha = [float(v) for v in sched.base.alphas_cumprod]
    recip = [float(v) for v in sched.base.sqrt_recip_alphas_cumprod]
    recipm1 = [float(v) for v in sched.base.sqrt_recipm1_alphas_cumprod]
    for a, r, r1 in zip(alpha, recip, recipm1):
        assert r == pytest.approx(1.0 / math.sqrt(a), rel=1e-6)
        assert r1 == pytest.approx(math.sqrt(1.0 / a - 1.0), rel=1e-5)


# --------------------------------------------------------------------------
# contract S4.3 -- steps and sub-sampling
# --------------------------------------------------------------------------
def test_resolve_num_steps_follows_generate_py():
    # generate.py:196-202
    assert resolve_num_steps(None) == NUM_BASE_STEPS
    assert resolve_num_steps(1) == 1
    assert resolve_num_steps(10) == 10
    with pytest.raises(ValueError):
        resolve_num_steps(0)
    with pytest.raises(ValueError):
        resolve_num_steps(11)
    with pytest.raises(ValueError):
        resolve_num_steps(-3)


def test_space_timesteps_matches_the_reference_arithmetic():
    # diffusion.py:50-58: round(arange(N) * (N-1)/max(1,n-1)), clamped, deduped.
    assert space_timesteps(10, 10) == list(range(10))
    assert space_timesteps(10, 1) == [0, 9]
    assert space_timesteps(10, 3) == [0, 4, 9]
    # n=2 is the documented quirk of the reference arithmetic: round() maps
    # 0,1,...,9 to {0,9} only after dedup, so the chain is 2 long. Recorded as
    # "steps is a request, not a guarantee" in docs/status-claude.md.
    assert space_timesteps(10, 2) == [0, 9]
    for n in range(1, 11):
        used = space_timesteps(10, n)
        assert used == sorted(set(used))
        assert used[0] >= 0 and used[-1] <= 9
        assert 1 <= len(used) <= n + 1


def test_timestep_order_is_large_to_small():
    # ardy_model.py:622 -- indices = list(range(num_denoising_steps))[::-1]
    sched = get_diffusion_schedule(4)
    assert sched.timesteps == [3, 2, 1, 0]
    assert sched.num_steps == 4
    assert sched.used_timesteps == space_timesteps(10, 4)


# --------------------------------------------------------------------------
# contract S4.2 -- DDIM, eta = 0
# --------------------------------------------------------------------------
def test_ddim_single_step_matches_the_closed_form():
    """Compare against ``pred_xstart * sqrt(a_prev) + sqrt(1 - a_prev) * eps``."""
    sched = get_diffusion_schedule(10)
    ddim = DDIMSampler(sched)
    t = 7
    x_t = mx.array([[0.3, -1.2, 0.7]], dtype=mx.float32)
    pred = mx.array([[0.5, -0.25, 1.0]], dtype=mx.float32)

    a_t = scalar(sched.alpha_bar_at(t))
    a_prev = scalar(sched.alpha_bar_prev_at(t))
    x_values = x_t[0].tolist()
    pred_values = pred[0].tolist()
    eps = [
        (x - p * math.sqrt(a_t)) / math.sqrt(1.0 - a_t)
        for x, p in zip(x_values, pred_values)
    ]
    expected = [
        p * math.sqrt(a_prev) + e * math.sqrt(1.0 - a_prev)
        for p, e in zip(pred_values, eps)
    ]

    got = ddim.step(x_t, pred, t)[0].tolist()
    for g, e in zip(got, expected):
        assert g == pytest.approx(e, abs=max(TOL, abs(e) * 1e-5))


def test_ddim_constant_prediction_is_linear_in_the_start():
    """If the model predicts the same x_start every step, the chain must follow the table.

    The whole chain is recomputed here from the closed form, so an implementation
    that used the wrong per-step table would diverge.
    """
    sched = get_diffusion_schedule(3)
    ddim = DDIMSampler(sched)
    pred_value = 0.25
    x_t = mx.array([[2.0]], dtype=mx.float32)
    pred = mx.array([[pred_value]], dtype=mx.float32)

    traces = []
    for t in sched.timesteps:
        x_t = ddim.step(x_t, pred, t)
        traces.append(scalar(x_t))

    expected_x = 2.0
    expected_trace = []
    for t in sched.timesteps:
        a_t = scalar(sched.alpha_bar_at(t))
        a_prev = scalar(sched.alpha_bar_prev_at(t))
        eps = (expected_x - pred_value * math.sqrt(a_t)) / math.sqrt(1.0 - a_t)
        expected_x = pred_value * math.sqrt(a_prev) + math.sqrt(1.0 - a_prev) * eps
        expected_trace.append(expected_x)

    assert len(traces) == 3
    for got, expected in zip(traces, expected_trace):
        assert got == pytest.approx(expected, abs=1e-5)


def test_ddim_is_deterministic_for_identical_inputs():
    sched = get_diffusion_schedule(10)
    ddim = DDIMSampler(sched)
    x_t = mx.array([[0.3, -1.2, 0.7]], dtype=mx.float32)
    pred = mx.array([[0.5, -0.25, 1.0]], dtype=mx.float32)
    first = ddim.step(x_t, pred, 5)
    second = ddim.step(x_t, pred, 5)
    assert first.tolist() == second.tolist(), "eta=0 DDIM must be bit-identical on repeat"

    # Running the whole chain twice must also agree bit for bit.
    def run() -> list:
        x = mx.array([[0.11, 0.22]], dtype=mx.float32)
        for t in sched.timesteps:
            x = ddim.step(x, mx.array([[0.05, -0.05]], dtype=mx.float32), t)
        return x.tolist()

    assert run() == run()


def test_predict_eps_round_trips_the_forward_process():
    """``x_t = sqrt(a_t) * x0 + sqrt(1 - a_t) * eps`` inverted by ``predict_eps``."""
    sched = get_diffusion_schedule(10)
    t = 4
    x0 = mx.array([[0.4, -0.9, 1.3]], dtype=mx.float32)
    eps = mx.array([[0.2, 0.5, -0.7]], dtype=mx.float32)
    a_t = scalar(sched.alpha_bar_at(t))
    x_t = math.sqrt(a_t) * x0 + math.sqrt(1.0 - a_t) * eps
    got = DDIMSampler.predict_eps(x_t, x0, sched, t)
    for g, e in zip(got[0].tolist(), eps[0].tolist()):
        assert g == pytest.approx(e, abs=1e-5)


def test_ddim_rejects_shape_mismatch():
    ddim = DDIMSampler(get_diffusion_schedule(2))
    with pytest.raises(ValueError):
        ddim.step(mx.zeros((1, 3)), mx.zeros((1, 4)), 0)


# --------------------------------------------------------------------------
# negative controls: the tests above must be able to fail
# --------------------------------------------------------------------------
def test_negative_control_dropping_the_cosine_offset_is_detected():
    """A plausible transcription slip must not pass the pointwise test.

    Dropping the ``+ 0.008`` numerator term (a common way to shorten the
    formula) changes ``alpha_bar`` enough that the comparison in
    ``test_alpha_bar_matches_the_contract_formula_pointwise`` would fail.
    """

    def wrong_alpha_bar(t: float) -> float:
        return math.cos(t / 1.008 * math.pi / 2) ** 2

    checked = 0
    for t in (0.0, 0.1, 0.5, 0.9, 1.0):
        expected = math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
        assert abs(wrong_alpha_bar(t) - expected) > 1e-9
        checked += 1
    assert checked == 5

    # And the difference propagates into beta_0, which the beta test pins.
    n = NUM_BASE_STEPS
    right_beta0 = min(
        1 - math.cos((1 / n + 0.008) / 1.008 * math.pi / 2) ** 2
        / math.cos((0 / n + 0.008) / 1.008 * math.pi / 2) ** 2,
        0.999,
    )
    wrong_beta0 = min(
        1 - math.cos((1 / n) / 1.008 * math.pi / 2) ** 2
        / math.cos((0 / n) / 1.008 * math.pi / 2) ** 2,
        0.999,
    )
    assert cosine_beta_schedule(n)[0] == pytest.approx(right_beta0, abs=1e-15)
    assert abs(wrong_beta0 - right_beta0) > 1e-4


def test_negative_control_skipping_the_shift_would_break_alphas_cumprod_prev():
    """If ``prev`` were just ``alphas_cumprod``, the DDIM anchor would be wrong."""
    sched = get_diffusion_schedule(None)
    alpha = [float(v) for v in sched.base.alphas_cumprod]
    prev = [float(v) for v in sched.base.alphas_cumprod_prev]
    # Unshifted would give prev[0] == alpha[0]; the contract requires 1.0.
    assert abs(prev[0] - 1.0) < 1e-6
    assert abs(alpha[0] - 1.0) > 1e-3, "a wrong implementation would be indistinguishable"
    # And the rest must be the shifted table, not the table itself.
    for i in range(1, len(alpha)):
        assert prev[i] == pytest.approx(alpha[i - 1], abs=1e-6)
        assert abs(prev[i] - alpha[i]) > 1e-4


def test_negative_control_ddim_without_eps_recovers_a_different_value():
    """Dropping the ``eps`` term changes the update; show the two differ."""
    sched = get_diffusion_schedule(10)
    ddim = DDIMSampler(sched)
    t = 6
    x_t = mx.array([[0.8, -0.4]], dtype=mx.float32)
    pred = mx.array([[0.1, 0.6]], dtype=mx.float32)

    correct = ddim.step(x_t, pred, t)[0].tolist()
    a_prev = scalar(sched.alpha_bar_prev_at(t))
    broken = [v * math.sqrt(a_prev) for v in pred[0].tolist()]
    assert correct != pytest.approx(broken, abs=1e-3)
