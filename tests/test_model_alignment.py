"""Golden alignment tests: the MLX port against the reference run.

These are the ``改坏就红`` tests for the port. They need two things that are
**deliberately not committed** (see NOTICE: we do not redistribute NVIDIA weights
or outputs derived from them):

* ``checkpoints/ARDY-Core-RP-20FPS-Horizon40/mlx/*.safetensors`` -- produced by
  ``scripts/ardy_convert.py`` (the reference weights themselves are gitignored);
* ``artifacts/golden-run/raw/*.npy`` -- the single authorised reference run's
  dumps.

When either is missing the tests **skip with an explicit reason** rather than
passing quietly, so a green run never implies coverage it did not have.

Tolerances and their justification are documented in ``scripts/ardy_align.py``;
briefly: both sides are float32 but on different hardware with different
reduction orders, so the honest target is "residue at the level of reordered
float32 summation" (measured: 1e-7..5e-5) rather than bit-equality. A structural
bug produces O(1e-1) errors, which the negative-control test demonstrates.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"
RAW = ROOT / "artifacts" / "golden-run" / "raw"
NPZ = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"

TOL = 1e-3


def _load_align():
    path = ROOT / "scripts" / "ardy_align.py"
    spec = importlib.util.spec_from_file_location("ardy_align", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


align = _load_align()


def _require_golden():
    missing = []
    if not (CKPT / "mlx" / "tokenizer.mlx.safetensors").is_file():
        missing.append(f"{CKPT}/mlx/tokenizer.mlx.safetensors (run scripts/ardy_convert.py)")
    if not (CKPT / "mlx" / "denoiser.mlx.safetensors").is_file():
        missing.append(f"{CKPT}/mlx/denoiser.mlx.safetensors (run scripts/ardy_convert.py)")
    if not (RAW / "text_feat.npy").is_file():
        missing.append(f"{RAW}/ (the golden run dumps)")
    if missing:
        pytest.skip("golden data not present: " + "; ".join(missing))


@pytest.fixture(scope="module")
def harness():
    _require_golden()
    h = align.Harness(CKPT, RAW, NPZ, device="gpu")
    h.load()
    return h


def test_checkpoint_inventory_matches_the_contract():
    """Tensor counts and parameter totals recorded from the reference (contract S3.0)."""
    from ardy_mlx.io.safetensors import param_count, read_header, tensors_of

    _require_golden()
    for name, tensors, params in (
        ("tokenizer.mlx.safetensors", 204, 35_430_052),
        ("denoiser.mlx.safetensors", 224, 155_823_252),
    ):
        header, _hl, _fs = read_header(CKPT / "mlx" / name)
        entries = tensors_of(header)
        assert len(entries) == tensors
        assert sum(param_count(e) for e in entries.values()) == params
        assert {e["dtype"] for e in entries.values()} == {"F32"}


def test_L1_tokenizer_roundtrip(harness):
    res = harness.level1()
    for name, st in res.items():
        assert st["max_abs"] <= TOL, f"{name}: max_abs={st['max_abs']:.3e} (tol {TOL:g})"
    # the discrete latent must be essentially exact, not merely within 1e-3
    assert res["latent"]["max_abs"] <= 1e-5


def test_L2_denoiser_single_step_all_ten_steps(harness):
    res = harness.level2()
    for key in ("raw", "cfg", "ddim"):
        worst = res["summary"][key]["max_abs"]
        assert worst <= TOL, f"{key}: worst max_abs={worst:.3e} over 10 steps (tol {TOL:g})"
    # every step individually, so a single bad timestep cannot hide in a max
    for row in res["per_step"]:
        assert row["cfg_max_abs"] <= TOL, row
        assert row["ddim_max_abs"] <= TOL, row


def test_L3_full_clip(harness):
    res = harness.level3()
    for name, st in res.items():
        assert st["max_abs"] <= TOL, f"{name}: max_abs={st['max_abs']:.3e} (tol {TOL:g})"
    assert res["frames"]["shape"] == [1, 40, 330]


def test_L3_root_matches_the_reference_npz_physical_values(harness):
    """The physical root is an independent target: npz root_positions / heading."""
    res = harness.level3()
    assert res["root_positions_vs_npz"]["max_abs"] <= TOL
    assert res["heading_vs_npz"]["max_abs"] <= TOL


def test_negative_control_the_tolerance_is_falsifiable(harness):
    """Corrupt x_T and the harness MUST fail. A tolerance that cannot fail is a lie."""
    bad = harness.level3(corrupt=True)
    worst = max(v["max_abs"] for v in bad.values())
    assert worst > 1e-1, f"corrupted run only moved by {worst:.3e}; the check is not discriminating"


def test_no_checkpoint_tensor_is_left_unused(harness):
    """Every tensor in the denoiser checkpoint must be exercised by the port.

    The tokenizer legitimately leaves the two ``target_cond_blocks`` tensors
    unused: the batch path never passes a target condition (fsq.py:253-259).
    """
    den = harness.den
    assert den.store.unused() == []
    assert harness.tok.store.unused() == [
        "pose_net.decoder.target_cond_blocks.0.bias",
        "pose_net.decoder.target_cond_blocks.0.weight",
    ]
