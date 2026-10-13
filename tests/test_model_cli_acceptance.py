"""End-to-end acceptance: the shipped CLI reproduces the golden clip.

This is the "does the command we tell people to run actually reproduce the
reference" test, one level above the level comparisons (which drive
``ArdyBatch`` directly). It drives ``scripts/ardy_generate.py`` **unchanged**
through its ``main()`` with B's own backend selection (``--backend mlx``, which
``0004`` ruling 5 makes the default path to A's ``ArdyBatch``) and injects the
reference ``x_T`` through the RNG, because the CLI cannot inject per-window noise
itself (its ``--noise-file`` multi-window branch compares ``injected.shape[1:]``,
which still carries the batch axis, against ``shape[1:]``; see
``scripts/ardy_align.py:cli_acceptance``). Nothing in B's tree is modified.

Needs the converted weights and the golden run dumps, neither of which is
committed (``NOTICE``), so it skips with an explicit reason when they are absent.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

CKPT = ROOT / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"
RAW = ROOT / "artifacts" / "golden-run" / "raw"
NPZ = ROOT / "artifacts" / "golden-run" / "out" / "golden_core_h40_2s_seed0.npz"
RAW4 = ROOT / "artifacts" / "golden-run-4s" / "raw"
NPZ4 = ROOT / "artifacts" / "golden-run-4s" / "out" / "golden_core_h40_4s_seed0.npz"

#: The frozen alignment tolerance (``0003`` ruling 1) and the tighter bound the
#: CLI actually achieves, so a regression that stays "within tolerance" but moves
#: by an order of magnitude still shows up.
TOL = 1e-3
TIGHT = 1e-4


def _load_align():
    path = ROOT / "scripts" / "ardy_align.py"
    spec = importlib.util.spec_from_file_location("ardy_align_cli", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


align = _load_align()


@pytest.fixture(scope="module")
def cli_runs():
    missing = []
    for p in (CKPT / "mlx" / "tokenizer.mlx.safetensors", CKPT / "mlx" / "denoiser.mlx.safetensors"):
        if not p.is_file():
            missing.append(f"{p} (run scripts/ardy_convert.py)")
    for p in (RAW / "text_feat.npy", NPZ, RAW4 / "randn_0.npy", NPZ4):
        if not p.is_file():
            missing.append(f"{p} (the golden run dumps)")
    if missing:
        pytest.skip("golden data / converted weights not present: " + "; ".join(missing))
    return align.cli_acceptance(device="gpu")


def test_cli_reproduces_the_single_window_golden_clip(cli_runs):
    entry = cli_runs["2.0s/40f/1window"]
    assert entry["exit_code"] == 0
    assert entry["num_frames"] == 40 and entry["windows"] == 1
    assert entry["rng_draws_injected"] == 1
    assert entry["frames"]["max_abs"] <= TIGHT, entry["frames"]
    assert entry["frames"]["reference_abs_max"] > 1.0  # O(1) metres, not noise
    assert entry["motion_normalized"]["max_abs"] <= TIGHT
    assert entry["root_positions_vs_npz"]["max_abs"] <= TIGHT
    assert entry["heading_vs_npz"]["max_abs"] <= TIGHT


def test_cli_reproduces_the_multi_window_golden_clip(cli_runs):
    """80 frames = two autoregressive windows, the shipped workload."""
    entry = cli_runs["4.0s/80f/2windows"]
    assert entry["exit_code"] == 0
    assert entry["num_frames"] == 80 and entry["windows"] == 2
    assert entry["rng_draws_injected"] == 2  # one reference x_T per window
    assert entry["frames"]["max_abs"] <= TIGHT, entry["frames"]
    assert entry["motion_normalized"]["max_abs"] <= TIGHT
    assert entry["root_positions_vs_npz"]["max_abs"] <= TIGHT
    assert entry["heading_vs_npz"]["max_abs"] <= TIGHT


def test_cli_exported_posed_joints_is_the_skeleton_fk_diagnostic(cli_runs):
    """What the *files* contain, as opposed to what A's FK can compute.

    The FK is real (``posed_joints_vs_npz`` below) but ``scripts/ardy_generate.py``
    still exports ``synthesize_posed_joints`` and labels it "real FK not
    implemented" -- B's sidecar wiring is outstanding, and this test says so
    without pinning the placeholder's exact wording.
    """
    for label in ("2.0s/40f/1window", "4.0s/80f/2windows"):
        entry = cli_runs[label]
        assert entry["posed_joints_vs_npz"]["max_abs"] <= 1e-5, label
        assert entry["posed_joints_placeholder_vs_npz"]["max_abs"] > 0.1, label
        # the broken-hierarchy control proves the comparison can fail
        assert entry["posed_joints_flat_hierarchy_vs_npz"]["max_abs"] > TOL, label
        assert isinstance(entry["sidecar"]["posed_joints_source"], str)
        assert entry["sidecar"]["posed_joints_source"]  # non-empty: the field exists
        assert entry["sidecar"]["crop_history_frames"] == 160  # the reference default
