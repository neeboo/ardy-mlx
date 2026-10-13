"""Tests for embedding loading, output assembly and the on-disk layout (contract S7)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import PRESETS, TEXT_FEAT_DIM  # noqa: E402
from ardy_mlx.sampling.pipeline import (  # noqa: E402
    REFERENCE,
    compute_feature_layout,
    duration_to_frames,
    git_sha,
    load_text_embedding,
    output_stem,
    prompt_hash,
    save_outputs,
    sha256_file,
    synthesize_posed_joints,
)

CORE = PRESETS["ARDY-Core-RP-20FPS-Horizon40"]


def write_embedding(tmp_path: Path, batch: int = 1, length: int = 4, name: str = "emb.npy"):
    values = np.arange(batch * length * TEXT_FEAT_DIM, dtype=np.float32).reshape(
        batch, length, TEXT_FEAT_DIM
    )
    path = tmp_path / name
    np.save(path, values)
    return path, values


# --------------------------------------------------------------------------
# contract S7 -- naming and hashing
# --------------------------------------------------------------------------
def test_prompt_hash_is_sha256_prefix_of_the_utf8_prompt():
    import hashlib

    prompt = "a person walks forward 走路"
    expected = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:16]
    assert prompt_hash(prompt) == expected
    assert len(prompt_hash(prompt)) == 16
    # Raw UTF-8, not normalised (decision 0002 ruling 4).
    assert prompt_hash(prompt) != prompt_hash(prompt + " ")


def test_output_stem_uses_noseed_when_the_seed_is_unset():
    assert output_stem("hi", 7) == f"{prompt_hash('hi')}-7"
    assert output_stem("hi", None) == f"{prompt_hash('hi')}-noseed"
    assert output_stem("hi", 0) == f"{prompt_hash('hi')}-0"


def test_sha256_file_matches_hashlib(tmp_path):
    import hashlib

    path = tmp_path / "blob.bin"
    path.write_bytes(b"ardy" * 1000)
    assert sha256_file(path) == hashlib.sha256(path.read_bytes()).hexdigest()


def test_git_sha_reports_a_commit_or_unknown():
    value = git_sha(Path(__file__).resolve().parents[1])
    assert value == "unknown" or len(value) == 40


# --------------------------------------------------------------------------
# embedding loading
# --------------------------------------------------------------------------
def test_load_npy_embedding_makes_a_batch_of_one(tmp_path):
    path, values = write_embedding(tmp_path, batch=1, length=5)
    loaded = load_text_embedding(path)
    assert loaded.embeddings.shape == (1, 5, TEXT_FEAT_DIM)
    assert loaded.mask.shape == (1, 5)
    assert bool(mx.all(loaded.mask)) is True
    assert loaded.sha256 == sha256_file(path)
    np.testing.assert_allclose(np.asarray(loaded.embeddings.tolist()), values, atol=0)


def test_load_npy_embedding_accepts_a_2d_single_prompt(tmp_path):
    values = np.zeros((6, TEXT_FEAT_DIM), dtype=np.float32)
    path = tmp_path / "flat.npy"
    np.save(path, values)
    loaded = load_text_embedding(path)
    assert loaded.embeddings.shape == (1, 6, TEXT_FEAT_DIM)


def test_load_npz_embedding_with_mask(tmp_path):
    embeddings = np.zeros((2, 3, TEXT_FEAT_DIM), dtype=np.float32)
    mask = np.array([[True, True, False], [True, False, False]])
    path = tmp_path / "emb.npz"
    np.savez(path, embedding=embeddings, mask=mask)
    loaded = load_text_embedding(path)
    assert loaded.embeddings.shape == (2, 3, TEXT_FEAT_DIM)
    assert np.asarray(loaded.mask.tolist()).tolist() == mask.tolist()


def test_load_embedding_rejects_a_wrong_feature_dim(tmp_path):
    path = tmp_path / "bad.npy"
    np.save(path, np.zeros((1, 4, TEXT_FEAT_DIM - 1), dtype=np.float32))
    with pytest.raises(ValueError):
        load_text_embedding(path)


def test_load_embedding_rejects_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_text_embedding(tmp_path / "nope.npy")


def test_load_embedding_rejects_a_missing_array_name(tmp_path):
    path = tmp_path / "emb.npz"
    np.savez(path, other=np.zeros((1, 2, TEXT_FEAT_DIM), dtype=np.float32))
    with pytest.raises(KeyError):
        load_text_embedding(path, array="embedding")


# --------------------------------------------------------------------------
# frame arithmetic
# --------------------------------------------------------------------------
def test_duration_to_frames_rounds_up():
    assert duration_to_frames(2.0, 20) == 40
    assert duration_to_frames(0.05, 20) == 1
    assert duration_to_frames(5.0, 20) == 100
    with pytest.raises(ValueError):
        duration_to_frames(0.0, 20)


def test_compute_feature_layout_uses_the_batch_max_history():
    layout, history, steps = compute_feature_layout(
        "ARDY-Core-RP-20FPS-Horizon40", 240
    )
    assert layout is CORE
    assert history == 160  # the batch max, not the GUI min of 4 (S5.2)
    # ceil(240 / 40): the reference has no seed context in the batch path, so the
    # history crop does not reduce the window count (ardy_model.py:569-570).
    assert steps == 6

    # Any positive frame count is accepted; only the history crop needs to be a
    # patch multiple.
    layout, history, steps = compute_feature_layout(
        "ARDY-Core-RP-20FPS-Horizon40", 40
    )
    assert steps == 1
    with pytest.raises(ValueError):
        compute_feature_layout("ARDY-Core-RP-20FPS-Horizon40", 0)


def test_compute_feature_layout_accepts_an_explicit_history():
    layout, history, steps = compute_feature_layout(
        "ARDY-Core-RP-20FPS-Horizon40", 40, history_frames=8
    )
    assert history == 8
    assert steps == 1
    with pytest.raises(ValueError):
        compute_feature_layout("ARDY-Core-RP-20FPS-Horizon40", 100, history_frames=6)
    with pytest.raises(KeyError):
        compute_feature_layout("not-a-model", 100)


# --------------------------------------------------------------------------
# posed joints placeholder
# --------------------------------------------------------------------------
def test_synthesize_posed_joints_shape_and_root_column():
    frames = mx.zeros((1, 3, CORE.motion_rep_dim), dtype=mx.float32)
    frames = frames.at[:, :, 0].add(mx.array([[1.0, 2.0, 3.0]], dtype=mx.float32))
    joints = synthesize_posed_joints(frames, CORE)
    assert joints.shape == (1, 3, CORE.nbjoints, 3)
    # Joint 0 is the root position; the rest come from local_joints_positions.
    assert joints[0, 0, 0].tolist() == [1.0, 0.0, 0.0]


# --------------------------------------------------------------------------
# contract S7 -- artifacts
# --------------------------------------------------------------------------
def test_save_outputs_writes_the_contract_members(tmp_path):
    frames = mx.zeros((1, 4, CORE.motion_rep_dim), dtype=mx.float32)
    frames = frames.at[:, :, 3:5].add(mx.array([[[1.0, 0.0]]], dtype=mx.float32))
    frames = frames.at[:, :, -4:].add(0.25)
    root_positions = frames[..., 0:3]
    contacts = frames[..., -4:] > 0.5
    joints = synthesize_posed_joints(frames, CORE)

    npz_path, json_path = save_outputs(
        out_dir=tmp_path / "ARDY-Core-RP-20FPS-Horizon40",
        model="ARDY-Core-RP-20FPS-Horizon40",
        layout=CORE,
        frames=frames,
        posed_joints=joints,
        root_positions=root_positions,
        foot_contacts=contacts,
        offsets=[0, 4],
        prompt="hello",
        seed=3,
        num_steps=10,
        cfg_weight=[2.0, 2.0],
        postprocess=False,
        duration_s=1.0,
        embedding_sha256="deadbeef",
        embedding_source="/tmp/emb.npy",
    )

    assert npz_path.name == f"{prompt_hash('hello')}-3.npz"
    assert json_path.name == f"{prompt_hash('hello')}-3.json"

    with np.load(npz_path, allow_pickle=False) as data:
        assert set(data.files) == {
            "frames",
            "posed_joints",
            "root_positions",
            "foot_contacts",
            "offsets",
            "fps",
            "text",
        }
        assert data["frames"].shape == (1, 4, CORE.motion_rep_dim)
        assert data["frames"].dtype == np.float32
        assert data["posed_joints"].shape == (1, 4, CORE.nbjoints, 3)
        assert data["root_positions"].shape == (1, 4, 3)
        assert data["foot_contacts"].dtype == np.bool_
        assert data["offsets"].tolist() == [0, 4]
        assert int(data["fps"]) == 20
        assert str(data["text"]) == "hello"

    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    for key in (
        "model",
        "skeleton",
        "fps",
        "horizon",
        "num_frames_per_token",
        "feature_dim",
        "num_frames",
        "duration_s",
        "seed",
        "num_steps",
        "cfg_weight",
        "postprocess",
        "text_embedding_sha256",
        "git_sha",
        "reference",
        "created_utc",
    ):
        assert key in sidecar, f"missing sidecar key {key}"
    assert sidecar["model"] == "ARDY-Core-RP-20FPS-Horizon40"
    assert sidecar["skeleton"] == "CoreSkeleton27"
    assert sidecar["fps"] == 20
    assert sidecar["horizon"] == 40
    assert sidecar["feature_dim"] == 330
    assert sidecar["num_frames"] == 4
    assert sidecar["seed"] == 3
    assert sidecar["num_steps"] == 10
    assert sidecar["cfg_weight"] == [2.0, 2.0]
    assert sidecar["postprocess"] is False
    assert sidecar["text_embedding_sha256"] == "deadbeef"
    assert sidecar["reference"] == REFERENCE
    assert sidecar["created_utc"].endswith("Z")
    # Provenance extras are additive, never a rename of a frozen key.
    assert sidecar["extra"]["backend"] == "placeholder"


def test_save_outputs_uses_noseed_in_the_file_name(tmp_path):
    frames = mx.zeros((1, 2, CORE.motion_rep_dim), dtype=mx.float32)
    npz_path, json_path = save_outputs(
        out_dir=tmp_path,
        model="ARDY-Core-RP-20FPS-Horizon40",
        layout=CORE,
        frames=frames,
        posed_joints=synthesize_posed_joints(frames, CORE),
        root_positions=frames[..., 0:3],
        foot_contacts=frames[..., -4:] > 0.5,
        offsets=[0, 2],
        prompt="p",
        seed=None,
        num_steps=10,
        cfg_weight=[2.0, 2.0],
        postprocess=False,
        duration_s=None,
        embedding_sha256="x",
    )
    assert npz_path.name.endswith("-noseed.npz")
    assert json_path.name.endswith("-noseed.json")
    sidecar = json.loads(json_path.read_text(encoding="utf-8"))
    assert sidecar["seed"] is None
    assert sidecar["duration_s"] is None
