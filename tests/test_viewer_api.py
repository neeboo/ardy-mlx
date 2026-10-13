"""Viewer unit tests: prompt lookup, duration rules, noise selection, payload shape.

None of these need a checkpoint or the golden dumps: the ``app`` fixture (see
``tests/conftest.py``) wires a :class:`FakeRunner` into a real
:class:`ardy_viewer.ViewerApp`, so the real ``CoreSkeleton27`` FK and the real
payload encoder run without loading 765 MB of weights.

The negative controls live in ``tests/test_viewer_negative_controls.py``; the tests
that do need the real weights are in ``tests/test_viewer_service.py`` and skip with a
reason.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def _load_viewer():
    """Load ``scripts/ardy_viewer.py`` under a stable module name.

    ``tests/`` and ``scripts/`` are not packages, and pytest's import modes disagree
    about whether ``conftest`` can be imported by name. This by-path load is what
    every viewer test file uses, and because the module name is fixed, all four files
    -- and the fixtures in ``tests/conftest.py`` -- get the **same** module object.
    Anything else would make ``monkeypatch.setattr`` patch a copy, which is a green
    test that proves nothing.
    """
    import importlib.util

    name = "ardy_viewer"
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / "ardy_viewer.py")
    assert spec and spec.loader, "cannot load scripts/ardy_viewer.py"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ardy_viewer = _load_viewer()

from ardy_mlx.model.skeleton import core_skeleton, posed_joints_from_features  # noqa: E402
from ardy_mlx.sampling import pipeline  # noqa: E402

#: The shipped Core layout, a 40x cheaper layout with the same arithmetic, and the
#: prompt the one authorised reference run encoded.
CORE = ardy_viewer.PRESETS["ARDY-Core-RP-20FPS-Horizon40"]
TINY = CORE.__class__(nbjoints=27, fps=5, horizon=5, skeleton="CoreSkeleton27")
PROMPT = "A person walks forward."
PROMPT_HASH = pipeline.prompt_hash(PROMPT)
WRONG_PROMPT = "A person runs backwards."


def synthetic_frames(num_frames: int, seed: int = 0):
    """A deterministic ``(num_frames, 330)`` feature block, in physical units.

    The layout is the contract's: ``root_pos(3) | heading(2) | local_joints(78) |
    global_rot_data(162) | velocities(81) | foot_contacts(4)``. The rotation block is
    a small perturbation of the identity 6D basis so ``cont6d_to_matrix`` produces a
    real rotation, and a contact is set on the first frame of every 5-frame group so
    a "no contacts anywhere" bug cannot pass silently.

    It lives in each file that needs it rather than behind an import: the only thing
    the viewer test files share is the module under test, and a fixture cannot be
    used to build a module-level constant.
    """
    import numpy as np

    rng = np.random.default_rng(seed)
    frames = np.zeros((num_frames, CORE.motion_rep_dim), dtype=np.float32)
    t = np.arange(num_frames, dtype=np.float32)
    frames[:, 0] = 0.05 * t
    frames[:, 1] = 0.9 + 0.02 * np.sin(t)
    frames[:, 3] = 1.0
    frames[:, 5 : 5 + 26 * 3] = 0.01 * rng.standard_normal((num_frames, 26 * 3))
    rot = np.zeros((num_frames, 27, 6), dtype=np.float32)
    rot[..., 0] = 1.0
    rot[..., 4] = 1.0
    rot[..., 2] = 0.05 * np.sin(t)[:, None]
    rot[..., 5] = 0.05 * np.cos(t)[:, None]
    frames[:, 83 : 83 + 27 * 6] = rot.reshape(num_frames, -1)
    frames[:, -4:] = 0.0
    frames[::5, -4:] = 1.0
    return frames


class FakeRunner:
    """The parts of :class:`ardy_mlx.sampling.pipeline.MlxRunner` the viewer uses."""

    def __init__(self, layout=None, num_windows: int = 1, seed: int = 0):
        self.model = "ARDY-Core-RP-20FPS-Horizon40"
        self.layout = layout or CORE
        self.parameter_count = 1234
        self.load_seconds = {"tokenizer_s": 0.0, "denoiser_s": 0.0, "total_s": 0.0}
        self.device = "fake"
        self.embedding = None
        self.num_windows = num_windows
        self.seed = seed
        self.calls: list[dict] = []

    @property
    def noise_window_shape(self):
        from ardy_mlx.io import NUM_FRAMES_PER_TOKEN

        return (1, self.layout.horizon // NUM_FRAMES_PER_TOKEN, self.layout.dim_token)

    def generate(self, *, num_frames, num_steps=10, cfg_weight=(2.0, 2.0), history_frames=None,
                 noise_fn=None, seed=None, windows=None):
        import numpy as np

        self.calls.append(
            {"num_frames": num_frames, "noise_fn": noise_fn, "windows": windows, "seed": seed}
        )
        windows = self.num_windows if windows is None else windows
        noise = None
        if noise_fn is not None:
            noise = np.asarray(noise_fn(0, self.noise_window_shape), dtype=np.float32)
        frames = synthetic_frames(num_frames, seed=self.seed)
        if noise is not None:
            # A deterministic, *tame* coupling: the noise nudges the root's y and z,
            # so two different x_T produce measurably different joints while the
            # numbers stay small enough for the FK to stay well-conditioned.
            level = float(np.abs(noise).mean())
            frames[:, 1] += 0.05 * level
            frames[:, 2] += 0.10 * level
        return {
            "frames": frames[None, ...],
            "offsets": [0] + [self.layout.horizon * (i + 1) for i in range(windows)],
            "history_frames": (
                history_frames if history_frames is not None else self.layout.default_history_frames
            ),
            "windows": windows,
        }


def test_prompt_hash_matches_the_repository_convention():
    """The cache is named by ``prompt_hash``, not by an ad-hoc digest."""
    text = "A person walks forward."
    assert pipeline.prompt_hash(text) == "e9cf8141cac79f35"
    assert len(pipeline.prompt_hash(text)) == 16


def test_resolve_by_hash_by_text_and_by_full_sha256(app):
    entry = app.embeddings.list()[0]
    digest = entry["prompt_hash"]
    assert app.embeddings.resolve(digest)["prompt_hash"] == digest
    assert app.embeddings.resolve(entry["text"])["prompt_hash"] == digest
    full = __import__("hashlib").sha256(entry["text"].encode("utf-8")).hexdigest()
    assert app.embeddings.resolve(full)["prompt_hash"] == digest


def test_missing_embedding_fails_by_name_and_never_falls_back(app):
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.embeddings.resolve("A person runs backwards.")
    assert excinfo.value.name == "no_cached_embedding_for_prompt"
    assert "451c936ec07c700e" in excinfo.value.message
    # The one cached prompt must still resolve: the failure is about the miss, not
    # about the cache being broken.
    assert app.embeddings.resolve("A person walks forward.")["text"] == "A person walks forward."


def test_a_deleted_embedding_stops_resolving(app, embeddings_dir, tmp_path):
    """Removing the file removes the prompt -- there is no second source."""
    path = next(embeddings_dir.glob("*.npy"))
    path.unlink()
    app.embeddings.refresh()
    assert app.embeddings.list() == []
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.embeddings.resolve("A person walks forward.")
    assert excinfo.value.name == "no_cached_embedding_for_prompt"


def test_unreadable_or_misnamed_files_are_not_silently_promoted(embeddings_dir):
    """A file whose name is not a prompt hash is not addressable."""
    (embeddings_dir / "notes.npy").write_bytes(b"not an array")
    (embeddings_dir / "0123456789abcdeff.npy").write_bytes(b"not an array")
    cache = ardy_viewer.EmbeddingCache(embeddings_dir)
    entries = cache.refresh()
    assert [entry["prompt_hash"] for entry in entries] == [pipeline.prompt_hash("A person walks forward.")]


def test_embedding_without_a_sidecar_is_labelled_unknown(embeddings_dir):
    (embeddings_dir / (pipeline.prompt_hash("A person walks forward.") + ".json")).unlink()
    cache = ardy_viewer.EmbeddingCache(embeddings_dir)
    entry = cache.refresh()[0]
    assert entry["has_prompt_text"] is False
    assert "unknown prompt" in entry["text"]
    # By hash it still resolves; by text it cannot, because no text is recorded.
    assert cache.resolve(entry["prompt_hash"])["prompt_hash"] == entry["prompt_hash"]


# --------------------------------------------------------------------------- #
# duration rules
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("duration", [2.0, 4.0, 6.0, 20.0])
def test_whole_horizons_are_accepted(duration):
    assert ardy_viewer.validate_duration(duration, CORE) == int(duration * CORE.fps)


@pytest.mark.parametrize("duration", [1.0, 3.0, 0.5, 2.2, 39.9])
def test_partial_horizons_are_rejected_by_name(duration):
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        ardy_viewer.validate_duration(duration, CORE)
    assert excinfo.value.name == "duration_not_horizon_multiple"
    assert "nearest legal values" in excinfo.value.message


def test_duration_is_checked_in_frames_not_seconds():
    """A 40-frame horizon at 20 fps is exactly 2 s; float error must not matter."""
    assert ardy_viewer.validate_duration(2.0, CORE) == 40
    # The arithmetic is on frames: `ceil(duration * fps)` is what decides, and a
    # duration whose *frames* are a whole horizon is accepted however the seconds
    # value is spelled.
    assert ardy_viewer.validate_duration(ardy_viewer.duration_to_frames(1.99, CORE.fps) / CORE.fps, CORE) == 40
    # 1.95 s -> 39 frames: not a whole horizon, and the seconds value alone would
    # not tell you that (39/20 = 1.95 is not a multiple of 2 either, but the point
    # is that the frame count is what is checked).
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        ardy_viewer.validate_duration(1.95, CORE)
    assert excinfo.value.name == "duration_not_horizon_multiple"
    assert "39 frames" in excinfo.value.detail


@pytest.mark.parametrize("duration", [0, -1.0, None, float("nan"), float("inf")])
def test_nonsense_durations_are_rejected(duration):
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        ardy_viewer.validate_duration(duration, CORE)
    assert excinfo.value.name == "invalid_duration"


def test_window_count_is_the_model_s_own_arithmetic():
    assert ardy_viewer.duration_to_windows(2.0, CORE) == 1
    assert ardy_viewer.duration_to_windows(4.0, CORE) == 2
    assert ardy_viewer.duration_to_windows(20.0, CORE) == 10


# --------------------------------------------------------------------------- #
# health
# --------------------------------------------------------------------------- #
def test_health_reports_the_skeleton_verbatim(app):
    health = app.health()
    skeleton = core_skeleton()
    assert health["skeleton"]["jointNames"] == list(skeleton.joint_names)
    assert health["skeleton"]["jointParents"] == list(skeleton.joint_parents)
    assert health["skeleton"]["rootIndex"] == skeleton.root_index
    assert health["skeleton"]["footJointIdx"] == list(skeleton.foot_joint_idx)
    assert len(health["skeleton"]["jointNames"]) == 27
    assert health["skeleton"]["neutralJoints"][0] == [0.0, 0.0, 0.0]


def test_health_lists_the_cached_prompt_and_the_duration_grid(app, embeddings_dir):
    health = app.health()
    assert health["embeddings_dir"] == str(embeddings_dir)
    assert [e["prompt_hash"] for e in health["embeddings"]] == [pipeline.prompt_hash("A person walks forward.")]
    assert health["duration_step_s"] == pytest.approx(CORE.horizon / CORE.fps)
    assert health["fps"] == CORE.fps
    assert health["horizon"] == CORE.horizon
    assert health["parameters"] == 1234


def test_health_names_the_golden_runs_and_the_limits(app):
    health = app.health()
    assert sorted(r["duration_s"] for r in health["golden_references"]) == [2.0, 4.0]
    assert "text_encoder" in health["limits"]
    # The honesty label moved with the feature: the boundary is no longer "there is
    # no encoder", it is that a locally-conditioned clip is *not* aligned with the
    # golden one. Both halves have to be in the string, or the label is wrong in one
    # direction or the other.
    text_encoder = health["limits"]["text_encoder"]
    assert "local MLX text encoder is available" in text_encoder
    assert "NOT" in text_encoder and "golden" in text_encoder
    assert "auth" in health["limits"]


def test_health_reports_the_encoder_block_even_when_it_is_disabled(app):
    """No encoder wired in is a *named state*, not an absent key.

    The page renders ``encoder.state`` directly; a missing block would make "this
    viewer cannot encode" indistinguishable from "this build is too old to say".
    """
    health = app.health()
    assert health["encoder"]["enabled"] is False
    assert health["encoder"]["state"] == "disabled"
    assert health["encoder"]["available"] is False
    assert health["encoder"]["encoder_id"] is None
    # And it says how to turn it on, rather than only that it is off.
    assert "--llm2vec-repo" in health["encoder"]["reason"]


def test_health_lists_the_cache_key_of_every_vector(app, embeddings_dir):
    """``(prompt_hash, encoder_id)`` is the key, and health shows both halves."""
    health = app.health()
    digest = pipeline.prompt_hash("A person walks forward.")
    assert health["embeddings_detail"]["count"] == 1
    assert health["embeddings_detail"]["variants"] == 1
    assert health["embeddings_detail"]["cache_keys"] == [[digest, "reference-llm2vec-bf16"]]
    assert health["embeddings_detail"]["by_source"] == {"reference": 1, "local": 0}
    # A sidecar with no ``encoder`` field is the historical reference convention --
    # labelled as the reference, not as "unknown".
    assert health["embeddings"][0]["encoder_id"] == "reference-llm2vec-bf16"
    assert health["embeddings"][0]["source"] == "reference"


# --------------------------------------------------------------------------- #
# plan payload
# --------------------------------------------------------------------------- #
def _decode(payload: dict, key: str, shape):
    import base64

    return np.frombuffer(base64.b64decode(payload[key]), dtype="<f4").reshape(shape)


def test_plan_payload_shape_and_decoding(app):
    payload = app.plan(prompt=pipeline.prompt_hash("A person walks forward."), seed=3,
                       duration=2.0, noise="random")
    assert payload["kind"] == "plan"
    assert payload["frames"] == 40
    assert payload["joints"] == 27
    assert payload["fps"] == CORE.fps
    assert payload["duration_s"] == pytest.approx(2.0)
    assert payload["windows"] == 1
    assert payload["seed"] == 3
    assert payload["postprocess"] is False
    assert payload["generate_ms"] >= 0

    joints = _decode(payload, "posed_joints_b64", (40, 27, 3))
    roots = _decode(payload, "root_positions_b64", (40, 3))
    import base64

    contacts = np.frombuffer(base64.b64decode(payload["foot_contacts_b64"]), dtype=np.uint8).reshape(40, 4)
    assert np.isfinite(joints).all()
    assert np.isfinite(roots).all()
    # The FK puts the root joint at root_positions exactly (contract S1.5).
    assert np.allclose(joints[:, 0, :], roots, atol=1e-6)
    assert contacts.dtype == np.uint8
    assert set(np.unique(contacts)) <= {0, 1}
    assert payload["foot_contact_order"] == ["left_heel", "left_toe", "right_heel", "right_toe"]
    assert payload["foot_joint_names"] == ["LeftFoot", "LeftToeBase", "RightFoot", "RightToeBase"]


def test_plan_reports_the_embedding_fingerprint_and_the_prompt_actually_used(app):
    payload = app.plan(prompt="A person walks forward.", duration=2.0, noise="random")
    entry = app.embeddings.list()[0]
    assert payload["embedding"]["prompt_hash"] == entry["prompt_hash"]
    assert payload["embedding"]["embedding_sha256"] == entry["embedding_sha256"]
    assert payload["embedding"]["text"] == "A person walks forward."
    assert payload["embedding"]["embedding_path"] == entry["path"]


def test_plan_can_carry_the_raw_330_wide_frames(app):
    payload = app.plan(prompt="A person walks forward.", duration=2.0, noise="random",
                       include_frames=True)
    frames = _decode(payload, "frames_330_b64", (40, CORE.motion_rep_dim))
    assert frames.shape == (40, 330)
    joints = _decode(payload, "posed_joints_b64", (40, 27, 3))
    # The clip's posed_joints really are the FK of the frames it reports.
    recomputed = np.asarray(posed_joints_from_features(__import__("mlx.core", fromlist=["core"]).array(frames)))
    assert np.abs(recomputed - joints).max() < 1e-5


def test_plan_does_not_carry_the_frames_unless_asked(app):
    payload = app.plan(prompt="A person walks forward.", duration=2.0, noise="random")
    assert "frames_330_b64" not in payload


def test_six_seconds_is_three_windows(app):
    payload = app.plan(prompt="A person walks forward.", duration=6.0, noise="random")
    assert payload["frames"] == 120
    assert payload["windows"] == 3
    assert len(payload["offsets"]) == 4


def test_plan_refuses_a_partial_horizon_before_touching_the_model(app):
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.plan(prompt="A person walks forward.", duration=1.0, noise="random")
    assert excinfo.value.name == "duration_not_horizon_multiple"
    assert app.runner.calls == []          # nothing was sampled
    # The prompt was resolved first, so a bad duration with a bad prompt still
    # reports the prompt: the user asked for a prompt that does not exist.
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.plan(prompt="nope", duration=1.0, noise="random")
    assert excinfo.value.name == "no_cached_embedding_for_prompt"
    assert app.runner.calls == []


def test_plan_without_a_prompt_is_named(app):
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.plan(prompt=None, duration=2.0, noise="random")
    assert excinfo.value.name == "no_prompt_given"


def test_plan_without_a_loaded_model_is_a_503_viewer_error(embeddings_dir):
    application = ardy_viewer.ViewerApp(embeddings_dir=embeddings_dir)
    application.embeddings.refresh()
    application.error = "FileNotFoundError: checkpoint directory not found"
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        application.plan(prompt="A person walks forward.", duration=2.0, noise="random")
    assert excinfo.value.name == "model_unavailable"
    assert excinfo.value.status == 503


def test_cfg_weight_and_step_resolution_matches_the_cli(app):
    payload = app.plan(prompt="A person walks forward.", duration=2.0, noise="random",
                       cfg_weight=[2.0, 2.0], diffusion_steps=5)
    assert payload["cfg_weight"] == [2.0, 2.0]
    assert payload["diffusion_steps"] == 5
    payload = app.plan(prompt="A person walks forward.", duration=2.0, noise="random", cfg_weight=[3.0])
    assert payload["cfg_weight"] == [3.0, 0.0]      # one value = text only, constraint 0.0
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.plan(prompt="A person walks forward.", duration=2.0, noise="random",
                 cfg_weight=[1.0, 2.0, 3.0])
    assert excinfo.value.name == "invalid_cfg_weight"


# --------------------------------------------------------------------------- #
# noise selection
# --------------------------------------------------------------------------- #
def test_unknown_noise_mode_is_named(app):
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.plan(prompt="A person walks forward.", duration=2.0, noise="maybe")
    assert excinfo.value.name == "unknown_noise_mode"


def test_random_noise_reports_the_seed_it_used(app):
    payload = app.plan(prompt="A person walks forward.", duration=2.0, noise="random", seed=11)
    assert payload["noise"]["mode"] == "random"
    assert payload["noise"]["seed"] == 11
    assert payload["noise"]["source_files"] == []
    assert app.runner.calls[-1]["noise_fn"] is None      # MLX's RNG, not an injection


def test_golden_noise_for_a_duration_without_a_recorded_xt_is_refused(app, monkeypatch):
    monkeypatch.setattr(ardy_viewer, "GOLDEN_RUNS", {**ardy_viewer.GOLDEN_RUNS, 6.0: ardy_viewer.GOLDEN_RUNS[4.0]})
    monkeypatch.setattr(ardy_viewer, "golden_noise_files", lambda duration: [])
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.plan(prompt="A person walks forward.", duration=2.0, noise="golden")
    assert excinfo.value.name == "no_golden_noise_for_duration"


def test_golden_noise_shape_is_validated_against_the_runner(app, tmp_path, monkeypatch):
    """A dump for the wrong window size must fail where it is read.

    The app fixture runs the real Core layout, so its per-window x_T is
    ``(1, 10, 148)``; the 4.0 s run's *whole* window is ``(1, 20, 148)``.
    """
    wrong = tmp_path / "randn_0.npy"
    np.save(wrong, np.zeros((1, 20, 148), dtype=np.float32))
    monkeypatch.setattr(ardy_viewer, "golden_noise_files", lambda duration: [wrong])
    with pytest.raises(ValueError) as excinfo:
        app.plan(prompt="A person walks forward.", duration=2.0, noise="golden")
    assert "expected" in str(excinfo.value) and "(1, 10, 148)" in str(excinfo.value)


def test_golden_noise_uses_one_recorded_window_per_autoregressive_step(app, tmp_path, monkeypatch):
    files = []
    for index in range(2):
        path = tmp_path / f"randn_{index}.npy"
        np.save(path, np.full(app.runner.noise_window_shape, 0.01 * (index + 1), dtype=np.float32))
        files.append(path)
    monkeypatch.setattr(ardy_viewer, "golden_noise_files", lambda duration: files)
    payload = app.plan(prompt="A person walks forward.", duration=4.0, noise="golden")
    assert payload["noise"]["mode"] == "golden"
    assert payload["noise"]["windows"] == 2
    assert len(payload["noise"]["source_files"]) == 2
    assert app.runner.calls[-1]["noise_fn"] is not None


def test_golden_noise_picks_the_window_matching_the_step(app, tmp_path, monkeypatch):
    """Window 1 gets file 1: a stuck index would make the two windows identical."""
    files = []
    for index, level in enumerate((0.1, 0.9)):
        path = tmp_path / f"randn_{index}.npy"
        np.save(path, np.full(app.runner.noise_window_shape, level, dtype=np.float32))
        files.append(path)
    monkeypatch.setattr(ardy_viewer, "golden_noise_files", lambda duration: files)
    seen = []

    original = app.runner.generate

    def spy(**kwargs):
        noise_fn = kwargs["noise_fn"]
        seen.append(float(np.asarray(noise_fn(0, app.runner.noise_window_shape)).mean()))
        seen.append(float(np.asarray(noise_fn(1, app.runner.noise_window_shape)).mean()))
        return original(**kwargs)

    app.runner.generate = spy
    app.plan(prompt="A person walks forward.", duration=4.0, noise="golden")
    assert seen[0] == pytest.approx(0.1) and seen[1] == pytest.approx(0.9)


# --------------------------------------------------------------------------- #
# golden reference
# --------------------------------------------------------------------------- #
def test_reference_for_a_duration_we_do_not_have_fails_by_name(app):
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.reference(6.0)
    assert excinfo.value.name == "no_golden_reference_for_duration"
    assert excinfo.value.status == 404


def test_reference_prompt_mismatch_is_refused(app, tmp_path, monkeypatch):
    """Comparing two prompts would not be a comparison."""
    npz = tmp_path / "mine.npz"
    np.savez(
        npz,
        posed_joints=np.zeros((2, 27, 3), dtype=np.float32),
        root_positions=np.zeros((2, 3), dtype=np.float32),
        foot_contacts=np.zeros((2, 4), dtype=bool),
        fps=np.int64(20),
        text=np.array("A person runs backwards."),
    )
    monkeypatch.setattr(ardy_viewer, "golden_npz", lambda duration: npz)
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        app.reference(2.0, prompt="A person walks forward.")
    assert excinfo.value.name == "golden_reference_prompt_mismatch"


def test_reference_payload_shape(app, tmp_path, monkeypatch):
    npz = tmp_path / "golden.npz"
    joints = synthetic_frames(40)[:, :3].reshape(40, 1, 3)          # any (T,27,3) will do
    np.savez(
        npz,
        posed_joints=np.repeat(joints, 27, axis=1).astype(np.float32),
        root_positions=np.zeros((40, 3), dtype=np.float32),
        foot_contacts=np.ones((40, 4), dtype=bool),
        fps=np.int64(20),
        text=np.array("A person walks forward."),
    )
    monkeypatch.setattr(ardy_viewer, "golden_npz", lambda duration: npz)
    payload = app.reference(2.0, prompt="A person walks forward.")
    assert payload["kind"] == "golden"
    assert payload["frames"] == 40 and payload["joints"] == 27
    assert payload["fps"] == 20
    assert payload["npz_sha256"] == ardy_viewer.sha256_file(npz)
    assert payload["source"]["postprocess"] is False
    assert _decode(payload, "posed_joints_b64", (40, 27, 3)).shape == (40, 27, 3)


# --------------------------------------------------------------------------- #
# the shared runner loader
# --------------------------------------------------------------------------- #
def test_build_mlx_runner_names_a_missing_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError) as excinfo:
        pipeline.build_mlx_runner("ARDY-Core-RP-20FPS-Horizon40", object(),
                                  checkpoint_dir=tmp_path / "nope", repo_root=ROOT)
    assert "checkpoint directory not found" in str(excinfo.value)


def test_build_mlx_runner_names_a_missing_converted_weight(tmp_path):
    (tmp_path / "mlx").mkdir()
    with pytest.raises(FileNotFoundError) as excinfo:
        pipeline.build_mlx_runner("ARDY-Core-RP-20FPS-Horizon40", object(),
                                  checkpoint_dir=tmp_path, repo_root=ROOT)
    assert "ardy_convert.py" in str(excinfo.value)


def test_build_mlx_runner_rejects_an_unknown_model():
    with pytest.raises(KeyError):
        pipeline.build_mlx_runner("not-a-model", object(), repo_root=ROOT)


def test_runner_noise_window_shape_is_the_model_s_own():
    """The x_T shape is derived from the layout, never written down twice."""
    runner = FakeRunner(layout=CORE)
    assert runner.noise_window_shape == (1, 10, 148)
    assert FakeRunner(layout=TINY).noise_window_shape == (1, 1, 148)


def test_weight_store_total_elements_does_not_mark_tensors_used():
    """The HUD's parameter count must not make ``unused()`` lie."""
    import mlx.core as mx

    from ardy_mlx.model.nn import WeightStore

    store = WeightStore({"a.weight": mx.zeros((2, 3)), "a.bias": mx.zeros((2,))})
    assert store.total_elements() == 8
    assert store.unused() == ["a.bias", "a.weight"]


# --------------------------------------------------------------------------- #
# continuation: a new prompt is not a reset
# --------------------------------------------------------------------------- #
# The model-backed half of this (a real clip generated *from* the clip on screen)
# lives in ``tests/test_viewer_service.py`` and needs the checkpoint. What is tested
# here is everything around it: which state is carried, what the payload promises
# about the seam, when an explicit ``reset`` wins, and that a seam whose carried
# state did not survive is refused rather than served.


class _MotionRepStub:
    """Only what the viewer's bookkeeping asks of ``MotionRep``.

    Identity, because these tests look at *whether* the state was carried and in
    what shape -- not at metres. The real normalise/unnormalise pair is affine and
    per channel; replacing it with identity keeps the arithmetic under test (the
    re-centring, the anchor, the seam check) exactly as it is in production.
    """

    @staticmethod
    def normalize(frames):
        return frames

    @staticmethod
    def unnormalize(frames):
        return frames


class _BatchStub:
    """A stand-in ``ArdyBatch``: the capability check, nothing else."""

    denoiser = object()
    tokenizer = object()
    motion_rep = _MotionRepStub()
    gen_horizon_len = 40


class ContinuationRunner(FakeRunner):
    """A ``FakeRunner`` that reports a continuation path, without a model."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.batch = _BatchStub()


@pytest.fixture()
def continuation_app(embeddings_dir):
    """A real ``ViewerApp`` whose runner can be continued from (see the stub above).

    Two prompts are on disk, because "a new prompt continues the clip on screen" is
    only meaningful with a *second* prompt to ask for.
    """
    import json

    digest = pipeline.prompt_hash(WRONG_PROMPT)
    rng = np.random.default_rng(11)
    np.save(embeddings_dir / f"{digest}.npy", rng.standard_normal((1, 1, 4096)).astype(np.float32))
    (embeddings_dir / f"{digest}.json").write_text(
        json.dumps({"prompt": WRONG_PROMPT, "prompt_hash": digest}), encoding="utf-8"
    )
    application = ardy_viewer.ViewerApp(embeddings_dir=embeddings_dir)
    application.embeddings.refresh()
    application.runner = ContinuationRunner()
    application.error = None
    application.boot_seconds = 0.01
    return application


def _decode(payload, key, shape):
    return np.frombuffer(base64.b64decode(payload[key]), dtype="<f4").reshape(shape)


def _stub_continuation(monkeypatch, gap_mm=0.0):
    """Replace the model-backed continuation with the data contract it must keep.

    The stub returns exactly what the real one returns -- a clip that opens on the
    previous clip's own last frame, in the space the window loop hands back and
    forth, plus that loop's run info -- so the assertions below are about the
    viewer's promises (the seam, the state, the reset) rather than about a model.
    """
    import mlx.core as mx

    def fake(self, *, runner, embedding, previous, frames_wanted, num_steps, weights, seed, noise_fn):
        assert previous is not None, "a continuation without a previous clip is a bug"
        carried = np.asarray(previous.frames[:, -1:, :], dtype=np.float32)
        new = np.asarray(synthetic_frames(frames_wanted - 1, seed=int(seed or 0)), dtype=np.float32)
        new = new[None, ...].copy()
        if gap_mm:
            # The failure the seam check exists for: the new motion does not start
            # where the clip on screen ended.
            new[..., 2] += float(gap_mm) / 1000.0
        served = np.concatenate([carried, new], axis=1)
        return (
            mx.array(served),
            served,
            {"offsets": [0, frames_wanted], "history_frames": int(previous.frames.shape[1])},
        )

    monkeypatch.setattr(ardy_viewer.ViewerApp, "_generate_continued", fake)


def test_the_first_clip_has_nothing_to_continue_from(continuation_app):
    payload = continuation_app.plan(prompt=PROMPT, duration=2.0, noise="random")
    block = payload["continuation"]
    assert block["requested"] is True, "the page sends no reset, so a continue was asked for"
    assert block["applied"] is False
    assert block["reason"] == "no_previous_clip"
    assert block["seam"] is None, "no continuation, so no seam to report"
    # ... and the clip that was drawn is now what the next request starts from.
    assert continuation_app.continuation is not None
    assert continuation_app.continuation.frames.shape == (1, 40, 330)
    assert continuation_app.continuation.prompt_hash == PROMPT_HASH
    assert continuation_app.continuation.roots.shape == (3,)


def test_a_new_prompt_continues_the_clip_on_screen(continuation_app, monkeypatch):
    _stub_continuation(monkeypatch)
    first = continuation_app.plan(prompt=PROMPT, duration=2.0, noise="random")
    second = continuation_app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")

    block = second["continuation"]
    assert block["applied"] is True
    assert block["reason"] is None
    assert block["history_frames"] == 40
    assert block["previous"]["prompt_hash"] == PROMPT_HASH

    # The criterion, on the frames the page will draw: frame 0 of the new clip *is*
    # the last frame of the previous one, joint for joint.
    previous = _decode(first, "posed_joints_b64", (first["frames"], first["joints"], 3))
    current = _decode(second, "posed_joints_b64", (second["frames"], second["joints"], 3))
    assert float(np.abs(current[0] - previous[-1]).max()) == 0.0
    assert block["seam"]["carried_frame_joint_mm"] == 0.0
    assert block["seam"]["carried_frame_root_mm"] == 0.0
    # The model's own next step is one frame further along, and is reported, not
    # hidden: the reader can check it against the clip's own frame-to-frame motion.
    assert block["seam"]["model_first_joint_mm"] >= 0.0
    first_roots = _decode(first, "root_positions_b64", (first["frames"], 3))
    assert block["seam"]["previous_last_root"] == [
        float(v) for v in first_roots[-1]
    ], "the seam is quoted against the frame the page was showing"
    assert second["frames"] == 40, "the carried frame does not lengthen the clip"
    assert second["duration_s"] == pytest.approx(2.0)


def test_negative_control_reset_does_not_continue(continuation_app, monkeypatch):
    _stub_continuation(monkeypatch)
    first = continuation_app.plan(prompt=PROMPT, duration=2.0, noise="random")
    second = continuation_app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random", reset=True)

    block = second["continuation"]
    assert block["requested"] is False
    assert block["applied"] is False
    assert block["reason"] == "reset_requested"
    assert block["seam"] is None
    # The jump the negative control exists for: a reset does *not* open on the frame
    # that was on screen.
    previous = _decode(first, "posed_joints_b64", (first["frames"], first["joints"], 3))
    current = _decode(second, "posed_joints_b64", (second["frames"], second["joints"], 3))
    assert float(np.abs(current[0] - previous[-1]).max()) > 0.0
    # ... and the state moves on to the clip that was actually drawn.
    assert continuation_app.continuation.prompt_hash == pipeline.prompt_hash(WRONG_PROMPT)


def test_golden_noise_falls_back_by_name_inside_a_continuation(continuation_app, monkeypatch):
    _stub_continuation(monkeypatch)
    continuation_app.plan(prompt=PROMPT, duration=2.0, noise="random")
    payload = continuation_app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="golden")
    assert payload["continuation"]["applied"] is True
    # 2.0 s *does* have a recorded x_T, so this is the continuation's own rule and
    # not the duration rule: golden noise reproduces the reference only from the
    # reference's own beginning.
    assert payload["noise"]["mode"] == "random"
    assert payload["noise"]["fallback_from"] == "golden"
    assert payload["noise"]["fallback_reason"] == ardy_viewer.CONTINUATION_NOISE_FALLBACK
    assert payload["noise"]["requested"] == "golden"


def test_negative_control_golden_noise_still_works_on_a_fresh_clip(app):
    """The fallback above is about continuations, not a blanket downgrade."""
    application = app
    application.runner = FakeRunner()          # no batch: every plan is a fresh one
    payload = application.plan(prompt=PROMPT, duration=2.0, noise="golden")
    assert payload["noise"]["mode"] == "golden"
    assert "fallback_from" not in payload["noise"]
    assert payload["continuation"]["applied"] is False
    assert payload["continuation"]["reason"] == "no_previous_clip"


def test_the_capability_check_needs_the_batch_path(continuation_app, monkeypatch):
    assert continuation_app.continuation_supported() is True
    _stub_continuation(monkeypatch)
    continuation_app.plan(prompt=PROMPT, duration=2.0, noise="random")
    assert continuation_app.continuation is not None, "a clip is on screen"

    # The runner loses the continuation path (an older server, a test double). The
    # state is still there, so the *capability* is what the payload names -- not
    # "there was nothing to continue from".
    continuation_app.runner = FakeRunner()
    assert continuation_app.continuation_supported() is False
    payload = continuation_app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    assert payload["continuation"]["applied"] is False
    assert payload["continuation"]["reason"] == "runner_has_no_continuation_path"
    # ... and a first plan on a runner with no batch says the honest other thing.
    fresh = ardy_viewer.ViewerApp(embeddings_dir=continuation_app.embeddings.directory)
    fresh.embeddings.refresh()
    fresh.runner = FakeRunner()
    fresh.error = None
    assert fresh.plan(prompt=PROMPT, duration=2.0, noise="random")["continuation"]["reason"] == (
        "no_previous_clip"
    )


# --------------------------------------------------------------------------- #
# the seam invariant: a lost carry-over is refused, not served
# --------------------------------------------------------------------------- #
class _SamplerStub:
    """The loop's result for one continuation, with the root wherever the test says."""

    def __init__(self, frames):
        self._frames = frames
        self.offsets = [0, 40]

    def run(self):
        return self

    def translated(self):
        return self._frames


def _continuation_state(viewer, frames=None):
    frames = np.asarray(
        frames if frames is not None else synthetic_frames(40, seed=3)[None, ...],
        dtype=np.float32,
    )
    return viewer.ContinuationState(
        frames=frames,
        joints=np.zeros((27, 3), dtype=np.float32),
        roots=np.array([0.0, 0.9, 1.2], dtype=np.float32),
        contacts=np.zeros(4, dtype=np.uint8),
        prompt_hash=PROMPT_HASH,
        prompt_text=PROMPT,
        model="ARDY-Core-RP-20FPS-Horizon40",
    )


def _patch_continuation_loop(monkeypatch, first_frame_offset_m):
    """Stub B's loop so the *viewer's* carry-over arithmetic is what runs."""
    import types

    import mlx.core as mx
    from ardy_mlx.model import ardy as ardy_model
    from ardy_mlx.sampling import autoregressive

    frames = np.asarray(synthetic_frames(40, seed=5), dtype=np.float32)[None, ...].copy()
    frames[0, :, 0] = 0.0
    frames[0, :, 2] = 0.0
    frames[0, 1:, 2] = np.linspace(0.0, 0.5, 39, dtype=np.float32)
    frames[0, 0, 2] = float(first_frame_offset_m)   # where the new motion claims to start

    monkeypatch.setattr(ardy_model, "make_step_fn", lambda *a, **k: (lambda request: None))
    monkeypatch.setattr(
        autoregressive, "ArdyAutoregressiveConfig", lambda **kwargs: types.SimpleNamespace(**kwargs)
    )
    monkeypatch.setattr(
        autoregressive, "ArdySampler",
        lambda config, step_fn, initial_history_frames=None: _SamplerStub(mx.array(frames)),
    )
    return frames


def test_the_carried_state_really_reaches_the_new_clip(continuation_app, monkeypatch):
    """The positive half: with the root where it should be, the seam is exact."""
    _patch_continuation_loop(monkeypatch, 0.0)
    previous = _continuation_state(ardy_viewer)
    generated, served_norm, run_info = continuation_app._generate_continued(
        runner=continuation_app.runner,
        embedding=type("E", (), {"embeddings": None, "mask": None})(),
        previous=previous,
        frames_wanted=40,
        num_steps=10,
        weights=(2.0, 2.0),
        seed=0,
        noise_fn=None,
    )
    assert run_info == {"offsets": [0, 40], "history_frames": 40}
    assert np.asarray(served_norm).shape == (1, 40, 330)
    assert np.array_equal(np.asarray(served_norm)[:, 0, :], previous.frames[:, -1, :])
    import mlx.core as mx

    assert np.asarray(mx.array(generated)).shape == (1, 40, 330)


def test_negative_control_a_lost_carry_over_is_refused(continuation_app, monkeypatch):
    """The failure mode: the new motion does not start where the clip on screen ended.

    A metre in one frame is not a motion, it is a bug -- a wrong anchor, a mixed
    unit, a dropped translation -- and it must not reach the page as a clip.
    """
    _patch_continuation_loop(monkeypatch, 1.0)
    previous = _continuation_state(ardy_viewer)
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        continuation_app._generate_continued(
            runner=continuation_app.runner,
            embedding=type("E", (), {"embeddings": None, "mask": None})(),
            previous=previous,
            frames_wanted=40,
            num_steps=10,
            weights=(2.0, 2.0),
            seed=0,
            noise_fn=None,
        )
    assert excinfo.value.name == "continuation_seam_too_far"
    assert excinfo.value.status == 500
    assert "mm" in excinfo.value.message
    # The limit is a *seam* budget: a quarter of a metre in 50 ms (5 m/s), well
    # above the ~60 mm/frame these prompts produce.
    assert 0.05 <= ardy_viewer.CONTINUATION_SEAM_LIMIT_M <= 0.5


def test_the_history_window_is_capped_at_the_checkpoint_s_budget(continuation_app, monkeypatch):
    """A longer clip does not become a longer context: the crop budget wins."""
    assert ardy_viewer.continuation_history_frames(CORE) == CORE.default_history_frames == 160
    _patch_continuation_loop(monkeypatch, 0.0)
    long_state = _continuation_state(ardy_viewer, frames=synthetic_frames(240, seed=1)[None, ...])
    _, served_norm, run_info = continuation_app._generate_continued(
        runner=continuation_app.runner,
        embedding=type("E", (), {"embeddings": None, "mask": None})(),
        previous=long_state,
        frames_wanted=40,
        num_steps=10,
        weights=(2.0, 2.0),
        seed=0,
        noise_fn=None,
    )
    assert run_info["history_frames"] == 160
    assert np.asarray(served_norm).shape == (1, 40, 330)


# ---------------------------------------------------------------------------
# the stream block: what the server can honestly say about playback
# ---------------------------------------------------------------------------
#
# `docs/decisions/0009` §二 asks for the stream state to be observable, and the one
# thing that matters about it is what it *declines* to report: the runahead queue
# lives in the page, so a `queue_depth` here would be a number the server invented.


def test_health_says_it_does_not_own_the_playback_queue(app):
    stream = app.health()["stream"]
    assert stream["owner"] == "page"
    # Nothing in the block may claim to know how deep the page's buffer is.
    assert "queue_depth" not in stream
    assert "buffer_depth" not in stream
    assert "buffered_frames" not in stream
    # What it does report is what it owns: the continuation state it would serve from.
    assert stream["continuation_active"] is False
    assert stream["segments_generated"] == 0
    assert stream["frames_generated"] == 0
    assert stream["last_generate_ms"] is None
    assert stream["history_frames"] == ardy_viewer.continuation_history_frames(CORE)


def test_the_stream_block_counts_what_was_served(app, embeddings_dir):
    """The counters must follow the clip actually served, not the request."""
    app.plan(prompt=PROMPT, duration=2.0, noise="random")
    first = app.health()["stream"]
    assert first["segments_generated"] == 1
    assert first["frames_generated"] == 40
    assert isinstance(first["last_generate_ms"], float)
    assert first["last_windows"] == 1

    app.plan(prompt=PROMPT, duration=4.0, noise="random")
    second = app.health()["stream"]
    assert second["segments_generated"] == 2
    assert second["frames_generated"] == 40 + 80
    assert second["last_windows"] == 2
    # And the continuation state it would serve from now exists, which is the fact
    # `continue_ready` is about.
    assert second["continuation_active"] is (app.continuation is not None)
    assert second["continue_ready"] == (
        app.continuation is not None and app.continuation_supported()
    )


def test_negative_control_reset_does_not_erase_what_was_generated(continuation_app, monkeypatch):
    """A reset starts a new *continuation*, not a new server.

    `segments_generated` is a count of what this server produced, so a reset must not
    zero it -- otherwise the page could not tell "the server has never generated"
    from "the reader just pressed Reload".
    """
    _stub_continuation(monkeypatch)
    continuation_app.plan(prompt=PROMPT, duration=2.0, noise="random")
    after_first = continuation_app.health()["stream"]
    assert after_first["segments_generated"] == 1

    continuation_app.plan(prompt=PROMPT, duration=2.0, noise="random", reset=True)
    block = continuation_app.health()["stream"]
    assert block["segments_generated"] == 2, "a reset is not a server restart"
    # A reset does not *erase* the state; it replaces it with the clip it just made
    # from noise, which is what the next plan will continue from. The distinction the
    # block has to preserve is "reset was asked for" (a plan-level fact, asserted in
    # the continuation tests) versus "this server has produced nothing yet" (a
    # server-level fact, which is what these counters are).
    assert block["continuation_active"] is True
    assert block["continue_ready"] == continuation_app.continuation_supported()
