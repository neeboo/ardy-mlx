"""The ``make_step_fn`` seam: the A-side adapter B's ``ArdySampler`` drives.

``make_step_fn`` (``src/ardy_mlx/model/ardy.py``) is the only thing B has to call to
run the real model behind ``DenoiseRequest -> DenoiseResult``. These tests cover it
directly, and end to end through B's own ``ArdySampler`` -- **without touching any
file under ``src/ardy_mlx/sampling/``**, which is B's.

Two kinds of input are used:

* a **synthetic** ``DenoiseRequest`` (hand-built with B's own dataclasses, zeros for
  the text conditioning and a fixed ``generation_noise``), for the contract,
  validation, determinism and caching assertions;
* B's ``ArdySampler`` itself, with a synthetic 80-frame request, to prove the whole
  chain runs.

The checkpoint is needed for both, so without it the module **skips with a reason**.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import PRESETS  # noqa: E402
from ardy_mlx.model.ardy import ArdyBatch, hybrid_from_explicit_frames, make_step_fn  # noqa: E402
from ardy_mlx.model.denoiser import AutoLatentTwostageDenoiser  # noqa: E402
from ardy_mlx.model.fsq import FSQVAETransformer  # noqa: E402
from ardy_mlx.model.motion_rep import MotionRep  # noqa: E402
from ardy_mlx.model.nn import WeightStore  # noqa: E402

CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"

# B's side of the contract. Imported (never modified) so the seam is tested against
# the *actual* signatures rather than a copy of them.
sampling = pytest.importorskip(
    "ardy_mlx.sampling", reason="B's sampling package is not importable yet"
)
DenoiseRequest = sampling.DenoiseRequest
DenoiseResult = sampling.DenoiseResult
WindowPlan = sampling.WindowPlan
plan_window = sampling.plan_window
ArdySampler = sampling.ArdySampler
ArdyAutoregressiveConfig = sampling.ArdyAutoregressiveConfig

LAYOUT = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]


def _require_model():
    if not (CKPT / "mlx" / "denoiser.mlx.safetensors").is_file():
        pytest.skip(f"{CKPT}/mlx/*.safetensors missing (run scripts/ardy_convert.py)")


@pytest.fixture(scope="module")
def batch():
    _require_model()
    mx.set_default_device(mx.gpu)
    motion_rep = MotionRep(CKPT / "stats" / "motion")
    tok = FSQVAETransformer(
        WeightStore.load(CKPT / "mlx" / "tokenizer.mlx.safetensors"), motion_rep
    ).load_stats(CKPT / "stats")
    den = AutoLatentTwostageDenoiser(
        WeightStore.load(CKPT / "mlx" / "denoiser.mlx.safetensors"), motion_rep
    )
    return ArdyBatch(tok, den, motion_rep)


def _text(b=1):
    """Synthetic text conditioning: only the shape/dtype matter for the seam."""
    return mx.zeros((b, 1, 4096), dtype=mx.float32), mx.ones((b, 1), dtype=mx.bool_)


def _request(step=0, held_frames=0, history_frames=None, gen_tokens=10, seed=0):
    plan = plan_window(
        step=step,
        held_frames=held_frames,
        gen_horizon_len=LAYOUT.horizon,
        crop_history_length=160,
    )
    if history_frames is None:
        history_frames = mx.zeros((1, plan.history_len, LAYOUT.motion_rep_dim), dtype=mx.float32)
    noise = mx.random.normal((1, gen_tokens, LAYOUT.dim_token), key=mx.random.key(seed)).astype(mx.float32)
    text, mask = _text()
    return DenoiseRequest(
        step=step,
        plan=plan,
        history_frames=history_frames,
        generation_noise=noise,
        first_heading_angle=mx.zeros((1,)),
        text_embeddings=text,
        text_mask=mask,
        cfg_weight=(2.0, 2.0),
        num_steps=10,
        seed=0,
    )


def test_step_fn_is_callable_and_returns_a_denoise_result(batch):
    text, mask = _text()
    step_fn = make_step_fn(batch, text, mask)
    req = _request()
    assert req.history_len == 0 and req.generation_len == 40
    res = step_fn(req)
    assert isinstance(res, DenoiseResult)
    assert res.generated_frames.shape == (1, 40, 330)
    assert res.generated_frames.dtype == mx.float32
    assert res.offsets is None
    assert bool(mx.all(mx.isfinite(res.generated_frames)))


def test_step_fn_accepts_a_clean_history_window(batch):
    """Step 1 of a 2-window chain: one token of history + the horizon."""
    text, mask = _text()
    step_fn = make_step_fn(batch, text, mask)
    first = step_fn(_request(step=0, held_frames=0))
    history = first.generated_frames
    second = step_fn(_request(step=1, held_frames=history.shape[1], history_frames=history))
    assert second.generated_frames.shape == (1, 40, 330)
    assert bool(mx.all(mx.isfinite(second.generated_frames)))
    # the chunk must not be a copy of the history it was given
    assert not np.array_equal(
        np.asarray(second.generated_frames), np.asarray(history)
    )


def test_step_fn_is_deterministic_for_the_same_request(batch):
    text, mask = _text()
    step_fn = make_step_fn(batch, text, mask)
    req = _request(seed=3)
    a = np.asarray(step_fn(req).generated_frames)
    b = np.asarray(step_fn(req).generated_frames)
    np.testing.assert_array_equal(a, b)


def test_step_fn_reuses_the_cached_latent_instead_of_re_encoding(batch):
    """B hands back explicit frames; re-tokenising them every step would silently
    replace the denoiser's own (requantised) latent with an encode of the decode.

    The adapter therefore caches the hybrid it produced and hands the tail back when
    the incoming history is a suffix of it. ``reuse_latent=False`` forces the
    re-encode, and the two must not be identical -- if they were, the cache would be
    pointless and this test would be lying.
    """
    text, mask = _text()
    req = _request(step=0, seed=11)
    out = {}

    for reuse in (True, False):
        step_fn = make_step_fn(batch, text, mask, reuse_latent=reuse)
        first = step_fn(req).generated_frames
        second = step_fn(_request(step=1, held_frames=40, history_frames=first, seed=12)).generated_frames
        mx.eval(second)
        out[reuse] = np.asarray(second)

    assert out[True].shape == out[False].shape == (1, 40, 330)
    assert not np.allclose(out[True], out[False], atol=1e-9), (
        "the latent cache must change the result; otherwise B's explicit-frame "
        "history is being re-encoded and the chain has drifted from the reference"
    )


def test_step_fn_validates_its_inputs(batch):
    text, mask = _text()
    step_fn = make_step_fn(batch, text, mask)

    bad_history = _request(step=0, held_frames=3)
    with pytest.raises(ValueError, match="multiple of the patch size"):
        step_fn(bad_history)

    bad_noise = _request(step=0, gen_tokens=5)
    with pytest.raises(ValueError, match="generation_noise"):
        step_fn(bad_noise)

    # a plan whose generation length is not the checkpoint horizon cannot be honoured
    plan = WindowPlan(
        step=0, history_end=0, history_start=0, history_len=0,
        generation_start=0, generation_len=8, window_len=8, cropped=False,
    )
    text_, mask_ = _text()
    bad_plan = DenoiseRequest(
        step=0, plan=plan, history_frames=mx.zeros((1, 0, 330)),
        generation_noise=mx.zeros((1, 2, 148)), first_heading_angle=mx.zeros((1,)),
        text_embeddings=text_, text_mask=mask_, cfg_weight=(2.0, 2.0),
        num_steps=10, seed=0,
    )
    with pytest.raises(ValueError, match="gen_horizon_len"):
        step_fn(bad_plan)


def test_hybrid_from_explicit_frames_is_the_documented_layout(batch):
    """``latent_utils.py:159-188``: root block first, tokenizer latent second."""
    frames = mx.zeros((1, 8, 330), dtype=mx.float32)
    hybrid = hybrid_from_explicit_frames(batch, frames)
    assert hybrid.shape == (1, 2, 148)
    # root block = 5 dims x 4 frames; latent = 128
    assert batch.denoiser.nframe_root_dim == 20
    assert batch.denoiser.latent_embedding_dim == 128


def test_step_fn_drives_ardysampler_end_to_end(batch):
    """The whole chain through **B's** loop, with a synthetic 80-frame request."""
    text, mask = _text()
    config = ArdyAutoregressiveConfig(
        layout=LAYOUT,
        num_frames=2 * LAYOUT.horizon,
        batch_size=1,
        history_frames=4,          # smaller than the default, to keep the test quick
        seed=0,
        text_embeddings=text,
        text_mask=mask,
    )
    sampler = ArdySampler(config, make_step_fn(batch, text, mask))
    result = sampler.run()

    assert result.frames.shape == (1, 2 * LAYOUT.horizon, LAYOUT.motion_rep_dim)
    assert result.frames.dtype == mx.float32
    assert bool(mx.all(mx.isfinite(result.frames)))
    assert result.global_transl.shape == (1, 3)
    assert len(result.offsets) == 2 + 1
    assert result.offsets[-1] == 2 * LAYOUT.horizon
    assert len(result.diagnostics) == 2
    assert all(d.clean_history_only for d in result.diagnostics)
    assert result.num_steps == 10

    # and it is reproducible
    again = ArdySampler(config, make_step_fn(batch, text, mask)).run()
    np.testing.assert_array_equal(np.asarray(result.frames), np.asarray(again.frames))


def test_step_fn_window_geometry_is_Bs_choice_not_the_reference_batch_path(batch):
    """Record, in code, the one interface difference the seam cannot paper over.

    The reference batch path starts window 0 with **zero** history
    (``history_end_frame = auto_step * gen_horizon_len + init_history_len`` with no
    ``init_history_sequence``), while B's loop right-aligns every window on the
    context it holds, which defaults to ``default_history_frames`` = 160 frames of
    zeros. So B's step 0 describes ``history_len = 160`` frames, not 0.

    This adapter honours the request it is given (that is the contract); the
    *caller* decides the window. Nothing here fails, but the numbers are pinned so
    the difference cannot drift unnoticed.
    """
    p0_b = plan_window(step=0, held_frames=LAYOUT.default_history_frames,
                       gen_horizon_len=LAYOUT.horizon, crop_history_length=160)
    p0_ref = plan_window(step=0, held_frames=0,
                         gen_horizon_len=LAYOUT.horizon, crop_history_length=160)
    assert LAYOUT.default_history_frames == 160
    assert p0_b.history_len == 160 and p0_b.generation_len == 40
    assert p0_ref.history_len == 0 and p0_ref.generation_len == 40
    # both are legal requests as far as this adapter is concerned
    for plan in (p0_b, p0_ref):
        assert plan.history_len % 4 == 0


@pytest.mark.xfail(
    strict=False,
    reason=(
        "KNOWN INTERFACE DIFFERENCE (reported in docs/decisions/0005 section 4): B's "
        "ArdySampler right-aligns every window on its 160-frame zero seed, so its window 0 "
        "is '160 history + 40 generation' where the reference batch path uses '0 history + "
        "40 generation + 40 masked future'. Measured with the reference's own x_T injected: "
        "5.466 vs a golden |max| of 1.682. If this test XPASSes, B's loop has become "
        "golden-equivalent -- update docs/decisions/0005 section 4 and this marker."
    ),
)
def test_ardysampler_chain_is_golden_equivalent_with_the_reference_noise(batch):
    """End-to-end: B's loop, the reference's noise, the golden clip.

    The noise is the reference's own ``x_T``, injected by monkeypatching
    ``mx.random.normal`` **inside this test** (no file under ``src/ardy_mlx/sampling/``
    is touched); CUDA's RNG is not reproducible in MLX, so this is the only way to
    make the comparison apples-to-apples.

    ``num_frames`` counts seed + generated for B's loop, hence 160 + 80 = 240 for an
    80-frame output. Note that ``num_frames=80, history_frames=160`` yields **zero**
    steps, which is itself part of the difference.
    """
    raw4 = ROOT / "artifacts" / "golden-run-4s" / "raw"
    if not (raw4 / "randn_0.npy").is_file():
        pytest.skip("4.0 s golden dumps not present")
    text, mask = _text()
    config = ArdyAutoregressiveConfig(
        layout=LAYOUT,
        num_frames=LAYOUT.default_history_frames + 2 * LAYOUT.horizon,
        batch_size=1,
        history_frames=LAYOUT.default_history_frames,
        seed=0,
        text_embeddings=text,
        text_mask=mask,
    )
    sampler = ArdySampler(config, make_step_fn(batch, text, mask))
    golden = mx.array(np.load(raw4 / "randn_0.npy"))
    original = mx.random.normal
    mx.random.normal = lambda shape, key=None, **kw: golden  # noqa: ARG005
    try:
        result = sampler.run()
    finally:
        mx.random.normal = original

    assert len(result.diagnostics) == 2
    assert [d.history_len for d in result.diagnostics] == [160, 160]
    gold_frames = np.asarray(
        batch.motion_rep.unnormalize(mx.array(np.load(raw4 / "motion_normalized.npy")))
    )
    mine = np.asarray(result.frames)
    assert mine.shape == gold_frames.shape
    np.testing.assert_allclose(mine, gold_frames, atol=1e-3)
