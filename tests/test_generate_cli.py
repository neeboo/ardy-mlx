"""End-to-end tests for ``scripts/ardy_generate.py`` (contract S7).

The CLI is exercised as a subprocess so argument parsing, the sidecar and the
npz members are all covered the way a user would hit them. The placeholder
backend is used: the run is shape-correct but semantically meaningless (see
``sampling/demo.py``), which is exactly what these tests check.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "ardy_generate.py"
PYTHON = REPO / ".venv" / "bin" / "python"
if not PYTHON.exists():  # pragma: no cover - fall back to the running interpreter
    PYTHON = Path(sys.executable)

SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import PRESETS, TEXT_FEAT_DIM  # noqa: E402

CORE = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
#: 12 s at 20 fps -> 240 frames; 160 of seed context plus 2 chunks of 40. The
#: default batch history is 160, so anything shorter is rejected (S5.2).
DURATION_S = 12.0
#: ``--duration`` is the *requested clip length* (240 frames); the loop's seed
#: context is not part of the output, so the artifact holds the generated frames
#: only: ``num_frames - init_history_len`` = 240 - 160 = 80 = 2 chunks of 40.
EXPECTED_FRAMES = 80
EXPECTED_CHUNKS = 2


def write_embedding(tmp_path: Path, length: int = 4) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    path = tmp_path / "embedding.npy"
    values = np.linspace(0.0, 1.0, length * TEXT_FEAT_DIM, dtype=np.float32).reshape(
        1, length, TEXT_FEAT_DIM
    )
    np.save(path, values)
    return path


def run_cli(tmp_path: Path, *extra: str) -> subprocess.CompletedProcess:
    embedding = write_embedding(tmp_path)
    out_dir = tmp_path / "out"
    cmd = [
        str(PYTHON),
        str(SCRIPT),
        "--embedding",
        str(embedding),
        "--model",
        "ARDY-Core-RP-20FPS-Horizon40",
        "--duration",
        str(DURATION_S),
        "--seed",
        "0",
        "--mode",
        "fast",
        "--prompt",
        "a person walks",
        "--no-postprocess",
        "--out",
        str(out_dir),
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO), timeout=900)


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("ardy-generate")
    completed = run_cli(tmp_path)
    return tmp_path, completed


def test_cli_succeeds_and_reports_shapes(cli_run):
    tmp_path, completed = cli_run
    assert completed.returncode == 0, completed.stderr
    assert "frames=" in completed.stdout
    assert "wrote" in completed.stdout


def test_cli_writes_the_contract_artifacts(cli_run):
    tmp_path, completed = cli_run
    assert completed.returncode == 0, completed.stderr
    model_dir = tmp_path / "out" / "ARDY-Core-RP-20FPS-Horizon40"
    npz_files = list(model_dir.glob("*.npz"))
    json_files = list(model_dir.glob("*.json"))
    assert len(npz_files) == 1 and len(json_files) == 1
    assert npz_files[0].stem == json_files[0].stem
    assert npz_files[0].stem.endswith("-0")


def test_cli_npz_shapes_match_the_contract(cli_run):
    tmp_path, completed = cli_run
    assert completed.returncode == 0, completed.stderr
    npz_path = next((tmp_path / "out").rglob("*.npz"))
    with np.load(npz_path, allow_pickle=False) as data:
        frames = data["frames"]
        # The batch axis is kept: one prompt -> batch 1.
        assert frames.shape == (1, EXPECTED_FRAMES, CORE.motion_rep_dim)
        assert frames.dtype == np.float32
        assert data["posed_joints"].shape == (1, EXPECTED_FRAMES, CORE.nbjoints, 3)
        assert data["root_positions"].shape == (1, EXPECTED_FRAMES, 3)
        assert np.array_equal(frames[..., 0:3], data["root_positions"])
        assert data["foot_contacts"].shape == (1, EXPECTED_FRAMES, 4)
        assert data["foot_contacts"].dtype == np.bool_
        offsets = data["offsets"]
        assert offsets.dtype == np.int64
        assert offsets[0] == 0
        assert offsets[-1] == EXPECTED_FRAMES
        assert list(offsets) == [0] + [40 * k for k in range(1, EXPECTED_CHUNKS + 1)]
        assert int(data["fps"]) == 20
        assert str(data["text"]) == "a person walks"


def test_cli_output_respects_the_feature_domains(cli_run):
    tmp_path, completed = cli_run
    assert completed.returncode == 0, completed.stderr
    npz_path = next((tmp_path / "out").rglob("*.npz"))
    with np.load(npz_path, allow_pickle=False) as data:
        frames = data["frames"]
    heading = frames[..., 3:5]
    np.testing.assert_allclose(np.linalg.norm(heading, axis=-1), 1.0, atol=1e-5)
    contacts = np.unique(frames[..., -4:])
    assert set(contacts.tolist()) <= {0.0, 1.0}


def test_cli_sidecar_carries_the_frozen_fields(cli_run):
    tmp_path, completed = cli_run
    assert completed.returncode == 0, completed.stderr
    json_path = next((tmp_path / "out").rglob("*.json"))
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["model"] == "ARDY-Core-RP-20FPS-Horizon40"
    assert sidecar["skeleton"] == "CoreSkeleton27"
    assert sidecar["fps"] == 20
    assert sidecar["horizon"] == 40
    assert sidecar["num_frames_per_token"] == 4
    assert sidecar["feature_dim"] == 330
    assert sidecar["num_frames"] == EXPECTED_FRAMES
    # duration_s records the requested duration; num_frames is the generated count.
    assert sidecar["duration_s"] == pytest.approx(DURATION_S)
    assert sidecar["extra"]["init_history_frames"] == 160
    assert sidecar["extra"]["num_autoregressive_steps"] == EXPECTED_CHUNKS
    assert sidecar["seed"] == 0
    assert sidecar["num_steps"] == 10  # --diffusion-steps default (S4.3)
    assert sidecar["cfg_weight"] == [2.0, 2.0]  # S4.4 default
    assert sidecar["postprocess"] is False
    assert len(sidecar["text_embedding_sha256"]) == 64
    assert sidecar["reference"] == {"repo": "nv-tlabs/ardy", "commit": "693f74d"}
    # The embedding checksum must be the real file digest, not a placeholder.
    import hashlib

    expected = hashlib.sha256((tmp_path / "embedding.npy").read_bytes()).hexdigest()
    assert sidecar["text_embedding_sha256"] == expected


def test_cli_is_deterministic_for_a_seed(tmp_path):
    first = run_cli(tmp_path / "a")
    second = run_cli(tmp_path / "b")
    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    npz_a = next((tmp_path / "a").rglob("*.npz"))
    npz_b = next((tmp_path / "b").rglob("*.npz"))
    with np.load(npz_a, allow_pickle=False) as a, np.load(npz_b, allow_pickle=False) as b:
        assert np.array_equal(a["frames"], b["frames"])


def test_cli_rejects_a_duration_shorter_than_the_seed_context(tmp_path):
    embedding = write_embedding(tmp_path)
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--embedding",
            str(embedding),
            "--duration",
            "1.0",
            "--mode",
            "fast",
            "--no-postprocess",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
    )
    assert completed.returncode != 0
    assert "shorter than" in (completed.stderr + completed.stdout)


def test_cli_refuses_to_claim_postprocessing_was_applied(tmp_path):
    """Core defaults to MotionCorrection upstream; v1 pins "both sides skip it"."""
    embedding = write_embedding(tmp_path)
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--embedding",
            str(embedding),
            "--duration",
            str(DURATION_S),
            "--mode",
            "fast",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
    )
    assert completed.returncode == 2
    assert "no-postprocess" in completed.stderr


def test_cli_mlx_backend_reports_a_missing_checkpoint(tmp_path):
    """The real backend must fail loudly when its converted weights are absent."""
    embedding = write_embedding(tmp_path)
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--embedding",
            str(embedding),
            "--duration",
            str(DURATION_S),
            "--backend",
            "mlx",
            "--no-postprocess",
            "--checkpoint-dir",
            str(tmp_path / "no-such-checkpoint"),
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
    )
    assert completed.returncode != 0
    assert "checkpoint directory not found" in completed.stderr


GOLDEN = REPO / "artifacts" / "golden-run"
CHECKPOINT = REPO / "checkpoints" / "ARDY-Core-RP-20FPS-Horizon40"


@pytest.mark.skipif(
    not (GOLDEN / "raw" / "motion_normalized.npy").exists() or not CHECKPOINT.exists(),
    reason="golden run artifacts or converted checkpoint not present",
)
def test_cli_mlx_backend_reproduces_the_golden_clip(tmp_path):
    """The whole CLI path against A's golden reference clip (2 s, Core/H40).

    The reference's own ``x_T`` is injected (``--noise-file``) because CUDA's
    Philox stream is not reproducible in MLX; everything after that runs through
    the real weights via ``--backend mlx``. The comparison is against
    ``motion_normalized.npy`` un-normalised with the checkpoint's own stats --
    the same lossless quantity A's ``ardy_align.py`` L3 uses.
    """
    import mlx.core as mx

    from ardy_mlx.model.motion_rep import MotionRep

    raw = GOLDEN / "raw"
    npz_path = tmp_path / "golden_embedding.npz"
    np.savez(
        npz_path,
        embedding=np.load(raw / "text_feat.npy"),
        mask=np.load(raw / "text_pad_mask.npy"),
    )
    noise_path = tmp_path / "x_t.npy"
    np.save(noise_path, np.load(raw / "denoise_step0_x_in.npy"))

    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--embedding",
            str(npz_path),
            "--model",
            "ARDY-Core-RP-20FPS-Horizon40",
            "--duration",
            "2.0",
            "--seed",
            "0",
            "--backend",
            "mlx",
            "--checkpoint-dir",
            str(CHECKPOINT),
            "--prompt",
            "A person walks forward.",
            "--noise-file",
            str(noise_path),
            "--no-postprocess",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr

    produced = next((tmp_path / "out").rglob("*.npz"))
    with np.load(produced, allow_pickle=False) as data:
        frames = data["frames"]
    assert frames.shape == (1, 40, CORE.motion_rep_dim)

    motion_rep = MotionRep(CHECKPOINT / "stats" / "motion")
    expected = np.asarray(
        motion_rep.unnormalize(mx.array(np.load(raw / "motion_normalized.npy")))
    )
    residue = float(np.abs(frames - expected).max())
    # Same tolerance as A's L3 (1e-3): float32 on Metal vs CUDA, different
    # reduction orders. Measured residue is ~1.9e-5.
    assert residue < 1e-3, f"golden residue {residue:.3e} exceeds the 1e-3 tolerance"


def test_cli_rejects_a_single_window_noise_file_for_a_two_window_run(tmp_path):
    """The exact pre-fix silent fallback: one window's noise, two windows to fill.

    The check runs before any weight is loaded, so this needs no checkpoint: a
    wrong x_T still produces plausible-looking motion, and the pre-fix code
    printed nothing at all (contract S11 N16).
    """
    embedding = write_embedding(tmp_path)
    noise = tmp_path / "one_window.npy"
    np.save(noise, np.zeros((1, 10, 148), dtype=np.float32))  # reference x_T is (1,10,148)
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--embedding",
            str(embedding),
            "--duration",
            "4.0",
            "--backend",
            "mlx",
            "--noise-file",
            str(noise),
            "--no-postprocess",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
    )
    assert completed.returncode != 0
    assert "refusing to inject only window 0" in completed.stderr


def test_cli_rejects_a_structurally_wrong_noise_file(tmp_path):
    """A shape that is neither one window nor a per-window stack fails too."""
    embedding = write_embedding(tmp_path)
    noise = tmp_path / "wrong.npy"
    np.save(noise, np.zeros((1, 10, 147), dtype=np.float32))  # token width off by one
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--embedding",
            str(embedding),
            "--duration",
            "2.0",
            "--backend",
            "mlx",
            "--noise-file",
            str(noise),
            "--no-postprocess",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
    )
    assert completed.returncode != 0
    assert "expected" in completed.stderr
    assert "(1, 10, 148)" in completed.stderr


def test_cli_honours_diffusion_steps_and_cfg_weight(tmp_path):
    completed = run_cli(
        tmp_path, "--diffusion-steps", "4", "--cfg-weight", "1.5", "0.5"
    )
    assert completed.returncode == 0, completed.stderr
    json_path = next((tmp_path / "out").rglob("*.json"))
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["num_steps"] == 4
    assert sidecar["cfg_weight"] == [1.5, 0.5]


def test_cli_rejects_out_of_range_diffusion_steps(tmp_path):
    embedding = write_embedding(tmp_path)
    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--embedding",
            str(embedding),
            "--duration",
            str(DURATION_S),
            "--diffusion-steps",
            "11",
            "--mode",
            "fast",
            "--no-postprocess",
            "--out",
            str(tmp_path / "out"),
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=300,
    )
    assert completed.returncode != 0
    assert "num_base_steps" in (completed.stderr + completed.stdout)


GOLDEN4 = REPO / "artifacts" / "golden-run-4s"


@pytest.mark.skipif(
    not (GOLDEN4 / "raw" / "motion_normalized.npy").exists()
    or not (GOLDEN4 / "raw" / "randn_0.npy").exists()
    or not (GOLDEN4 / "raw" / "randn_1.npy").exists()
    or not CHECKPOINT.exists(),
    reason="4 s golden run artifacts or converted checkpoint not present",
)
def test_cli_defaults_and_multi_window_noise_reproduce_the_4s_golden(tmp_path):
    """Two contract-S11-N16 fixes at once, on the shipped multi-window workload.

    * ``--history-frames`` is **omitted**: the CLI must resolve it to the preset's
      ``default_history_frames`` (160) instead of forwarding ``None``, which would
      disable cropping, the heading refresh and the chunked decode (4.42e-02).
    * ``--noise-file`` is the reference's per-window stack ``(2, 1, 10, 148)`` --
      the shape the old ``injected.shape[1:] == shape[1:]`` check could never
      match, so every window silently fell back to MLX's RNG (also 4.42).

    It also locks task 1: the npz's ``posed_joints`` must now be the skeleton FK,
    i.e. agree with the golden npz's ``posed_joints`` (pre-FK it was 0.236).
    """
    import mlx.core as mx

    from ardy_mlx.model.motion_rep import MotionRep

    raw = GOLDEN4 / "raw"
    npz_path = tmp_path / "golden4_embedding.npz"
    np.savez(
        npz_path,
        embedding=np.load(raw / "text_feat.npy"),
        mask=np.load(raw / "text_pad_mask.npy"),
    )
    noise_path = tmp_path / "x_t_4s.npy"
    np.save(noise_path, np.stack([np.load(raw / "randn_0.npy"), np.load(raw / "randn_1.npy")]))

    completed = subprocess.run(
        [
            str(PYTHON),
            str(SCRIPT),
            "--embedding",
            str(npz_path),
            "--model",
            "ARDY-Core-RP-20FPS-Horizon40",
            "--duration",
            "4.0",
            "--seed",
            "0",
            "--backend",
            "mlx",
            "--checkpoint-dir",
            str(CHECKPOINT),
            "--prompt",
            "A person walks forward.",
            "--noise-file",
            str(noise_path),
            "--no-postprocess",
            "--out",
            str(tmp_path / "out"),
            # no --history-frames: the default must be the preset's 160
        ],
        capture_output=True,
        text=True,
        cwd=str(REPO),
        timeout=900,
    )
    assert completed.returncode == 0, completed.stderr
    assert "shape=(2, 1, 10, 148)" in completed.stdout

    produced = next((tmp_path / "out").rglob("*.npz"))
    sidecar = json.loads(produced.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["extra"]["init_history_frames"] == CORE.default_history_frames == 160
    assert sidecar["extra"]["posed_joints_source"].startswith("skeleton FK")

    motion_rep = MotionRep(CHECKPOINT / "stats" / "motion")
    expected = np.asarray(
        motion_rep.unnormalize(mx.array(np.load(raw / "motion_normalized.npy")))
    )
    golden_joints = np.load(GOLDEN4 / "out" / "golden_core_h40_4s_seed0.npz")["posed_joints"]
    with np.load(produced, allow_pickle=False) as data:
        frames = data["frames"]
        joints = data["posed_joints"]
    assert frames.shape == (1, 80, CORE.motion_rep_dim)
    residue = float(np.abs(frames - expected).max())
    joint_residue = float(np.abs(joints - golden_joints).max())
    assert residue < 1e-3, f"golden residue {residue:.3e} exceeds the 1e-3 tolerance"
    # Task 1: the FK output, not the placeholder (which was 2.4e-01 here).
    assert joint_residue < 1e-4, f"posed_joints residue {joint_residue:.3e} is not the FK"
