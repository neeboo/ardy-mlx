"""The local text encoder: the file interface, the cache key, and the refusals.

Two things are being checked here, and they are different questions:

1. **Is the cache key ``(prompt_hash, encoder_id)``, and does source precedence
   hold?** These run against a fake encoder and a real :class:`ViewerApp`, so they
   are exact, fast and always run.
2. **Does the viewer really encode an unseen prompt through the sibling
   repository?** That is the file interface: a subprocess into ``llm2vec-mlx``'s own
   ``.venv``, reading back the ``.npy`` + ``.json`` pair. Those tests **skip with a
   reason** when the checkout or its weights are absent, because the encoder's
   weights are not committed and a test that pretended to exercise them would be the
   worst kind of green.

The negative controls are the point of the file. A local encoder that is missing, or
that fails, must produce a *named* error, and must never be papered over with another
prompt's vector or with another encoder's vector for the same prompt — those are the
two substitutions that would produce a perfectly convincing clip under the wrong
conditioning signal.
"""

from __future__ import annotations

import base64
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
for _candidate in (str(ROOT), str(ROOT / "src")):
    if _candidate not in sys.path:
        sys.path.insert(0, _candidate)


def _load(name: str, relative: str):
    """Import a repository file by path, **idempotently**.

    ``sys.modules`` is consulted first and the module is registered before it is
    executed. Without that, a second file loading ``scripts/ardy_viewer.py`` would
    create a *second* module object: ``ardy_viewer.ViewerError`` in a test and the
    ``ViewerError`` the server raises would then be different classes, and
    ``pytest.raises(ardy_viewer.ViewerError)`` would fail against an exception that
    looks perfect. The duplicate is silent, which is what makes it dangerous.
    """
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec and spec.loader, f"cannot load {relative}"
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# The viewer module under its fixed name, and the shared fixtures, so a
# ``monkeypatch`` here reaches the same object the server under test uses.
ardy_viewer = _load("ardy_viewer", "scripts/ardy_viewer.py")
conftest = _load("viewer_conftest", "tests/conftest.py")

from ardy_mlx.sampling import pipeline  # noqa: E402
from ardy_mlx.sampling.local_encoder import (  # noqa: E402
    DEFAULT_LLM2VEC_REPO,
    REFERENCE_ENCODER_ID,
    EncodeResult,
    LocalEncoder,
    LocalEncoderError,
    encoder_id_for,
    quantization_name,
)

PROMPT = conftest.PROMPT
WRONG_PROMPT = conftest.WRONG_PROMPT
PROMPT_HASH = pipeline.prompt_hash(PROMPT)
QUANTIZATION_4BIT = conftest.QUANTIZATION_4BIT
QUANTIZATION_8BIT = conftest.QUANTIZATION_8BIT
ENCODER_ID_4BIT = conftest.ENCODER_ID_4BIT
ENCODER_ID_8BIT = conftest.ENCODER_ID_8BIT
FakeLocalEncoder = conftest.FakeLocalEncoder
deterministic_vector = conftest.deterministic_vector

#: The 4-bit encoder artifact's own manifest blocks, copied from the shipped
#: ``llm2vec-mlx/weights/manifest-4bit.json``. The point of the literal is that the
#: *plan* is what names the artifact: 4-bit at group 32 with the last two layers and
#: the token embedding held at 8 bits. Plain 4-bit at group 32 measures a minimum
#: cosine of 0.9828 against the bf16 reference and does not clear the frozen 0.985
#: floor, so "4-bit" on its own names a different, failing model.
MANIFEST_4BIT = {
    "bits": 4,
    "group_size": 32,
    "mode": "affine",
    "encoder": "mlx-llm2vec",
    "recipe": {"quantization_plan": {"embed_tokens": 8, "__layers_from__": 30}},
    "quantization": {
        "tensors_quantized": 225,
        "escalated_tensors": 15,
        "mixed_precision": {
            "embed_tokens.weight": 8,
            **{
                f"layers.{layer}.{module}.weight": 8
                for layer in (30, 31)
                for module in (
                    "self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj",
                    "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj",
                )
            },
        },
    },
}

MANIFEST_8BIT = {
    "bits": 8,
    "group_size": 64,
    "mode": "affine",
    "encoder": "mlx-llm2vec",
    "recipe": {"quantization_plan": {}},
    "quantization": {"tensors_quantized": 225, "escalated_tensors": 0, "mixed_precision": {}},
}


def _decode(base64_text: str) -> np.ndarray:
    return np.frombuffer(base64.b64decode(base64_text), dtype="<f4")


# --------------------------------------------------------------------------- #
# naming the quantization in full
# --------------------------------------------------------------------------- #
def test_the_4bit_plan_is_named_in_full_not_as_just_4bit():
    """The failing artifact and the passing one must not share a name."""
    name = quantization_name(MANIFEST_4BIT)
    assert name == "4-bit group 32, last-2 layers + embed_tokens at 8-bit"
    # The two halves that actually distinguish it:
    assert "group 32" in name
    assert "8-bit" in name
    assert name != "4-bit"


def test_a_uniform_width_is_named_as_having_no_exceptions():
    assert quantization_name(MANIFEST_8BIT) == "8-bit group 64, no exceptions"


def test_a_manifest_that_does_not_record_its_plan_says_so():
    """An unknown plan is reported as unknown, never as the nominal width alone."""
    name = quantization_name({"bits": 4, "group_size": 32, "quantization": {}})
    assert name == "4-bit group 32, no exceptions"
    incomplete = quantization_name({"bits": 4, "quantization": {}})
    assert incomplete is None


def test_encoder_id_folds_the_quantization_into_the_key():
    """Two precisions of one prompt are two cache keys, not one."""
    assert encoder_id_for("mlx-llm2vec", quantization_name(MANIFEST_4BIT)) == ENCODER_ID_4BIT
    assert encoder_id_for("mlx-llm2vec", quantization_name(MANIFEST_8BIT)) == ENCODER_ID_8BIT
    assert ENCODER_ID_4BIT != ENCODER_ID_8BIT
    # No quantization at all still yields a stable, usable id rather than a bare name
    # that would silently collide with every other unquantized artifact.
    assert encoder_id_for("mlx-llm2vec", None) == "mlx-llm2vec"


# --------------------------------------------------------------------------- #
# the adapter's probe: state, not a guess
# --------------------------------------------------------------------------- #
def test_a_missing_checkout_is_unavailable_and_names_the_path(tmp_path):
    encoder = LocalEncoder(repo=tmp_path / "not-a-checkout")
    status = encoder.probe()
    assert status.available is False
    assert status.state == "unavailable"
    assert "not-a-checkout" in status.reason
    assert status.checks["repo"]["exists"] is False


def test_a_checkout_without_the_manifest_is_unavailable_and_says_what_to_run(tmp_path):
    repo = tmp_path / "llm2vec-mlx"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "encode.py").write_text("# stub\n", encoding="utf-8")
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    encoder = LocalEncoder(repo=repo, bits=4)
    status = encoder.probe()
    assert status.available is False
    assert "manifest" in status.reason
    assert "build_weights.py" in status.reason
    assert status.checks["manifest"]["exists"] is False


def test_a_complete_checkout_is_ready_and_reports_the_full_quantization(tmp_path):
    repo = tmp_path / "llm2vec-mlx"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "encode.py").write_text("# stub\n", encoding="utf-8")
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "weights").mkdir(parents=True)
    (repo / "weights" / "manifest-4bit.json").write_text(
        json.dumps(MANIFEST_4BIT), encoding="utf-8"
    )
    encoder = LocalEncoder(repo=repo, bits=4)
    status = encoder.probe()
    assert status.available is True
    assert status.state == "ready"
    assert status.encoder_id == ENCODER_ID_4BIT
    assert status.quantization == QUANTIZATION_4BIT
    # The probe must not load weights, so it cannot report a fake memory figure.
    assert status.probe_ms is not None and status.probe_ms < 5.0


def test_the_probe_never_spawns_a_subprocess(tmp_path, monkeypatch):
    """A health check that ran the encoder would cost 5 GB to answer a GET."""
    import subprocess

    def explode(*args, **kwargs):  # pragma: no cover - must not be reached
        raise AssertionError("probe() spawned a subprocess")

    monkeypatch.setattr(subprocess, "run", explode)
    encoder = LocalEncoder(repo=tmp_path / "nope")
    encoder.probe()


# --------------------------------------------------------------------------- #
# (prompt_hash, encoder_id) cache semantics
# --------------------------------------------------------------------------- #
def _write_vector(directory: Path, prompt: str, *, encoder_id, source_fields, tokens=1, salt="x"):
    """Write one cached vector + sidecar the way the cache layout prescribes."""
    directory.mkdir(parents=True, exist_ok=True)
    digest = pipeline.prompt_hash(prompt)
    vector = deterministic_vector(prompt, salt=salt, tokens=tokens)
    path = directory / f"{digest}.npy"
    np.save(path, vector)
    sidecar = {"prompt": prompt, "prompt_hash": digest, **source_fields}
    path.with_suffix(".json").write_text(json.dumps(sidecar), encoding="utf-8")
    return path, vector


def test_one_prompt_can_hold_a_reference_and_a_local_vector_under_two_keys(tmp_path):
    cache = ardy_viewer.EmbeddingCache(tmp_path)
    _write_vector(
        tmp_path, PROMPT,
        encoder_id=REFERENCE_ENCODER_ID,
        source_fields={"encoder": REFERENCE_ENCODER_ID},
        salt="ref",
    )
    _write_vector(
        tmp_path / ENCODER_ID_4BIT, PROMPT,
        encoder_id=ENCODER_ID_4BIT,
        source_fields={
            "encoder": "mlx-llm2vec",
            "encoder_id": ENCODER_ID_4BIT,
            "quantization": QUANTIZATION_4BIT,
            "bits": 4,
            "group_size": 32,
        },
        salt="mlx",
    )
    entries = cache.refresh()
    # One prompt, listed once -- as the reference -- but two addressable keys.
    assert len(entries) == 1
    assert entries[0]["source"] == "reference"
    assert sorted(cache.by_key) == sorted(
        [(PROMPT_HASH, REFERENCE_ENCODER_ID), (PROMPT_HASH, ENCODER_ID_4BIT)]
    )
    assert cache.resolve_cached(PROMPT_HASH, REFERENCE_ENCODER_ID)["source"] == "reference"
    assert cache.resolve_cached(PROMPT_HASH, ENCODER_ID_4BIT)["source"] == "local"
    # The sidecar's own names travel with the entry, unparaphrased.
    local = cache.resolve_cached(PROMPT_HASH, ENCODER_ID_4BIT)
    assert local["quantization"] == QUANTIZATION_4BIT
    assert local["encoder_id"] == ENCODER_ID_4BIT
    assert local["bits"] == 4 and local["group_size"] == 32
    # The two files really are different bytes, so the two keys are not cosmetic.
    assert (
        cache.resolve_cached(PROMPT_HASH, REFERENCE_ENCODER_ID)["embedding_sha256"]
        != local["embedding_sha256"]
    )


def test_a_cached_vector_is_never_returned_for_a_different_encoder_key(tmp_path):
    """The key is exact: asking for 8-bit must not hand back the 4-bit vector."""
    cache = ardy_viewer.EmbeddingCache(tmp_path)
    _write_vector(
        tmp_path / ENCODER_ID_4BIT, PROMPT,
        encoder_id=ENCODER_ID_4BIT,
        source_fields={"encoder_id": ENCODER_ID_4BIT, "quantization": QUANTIZATION_4BIT},
        salt="mlx",
    )
    cache.refresh()
    assert cache.resolve_cached(PROMPT_HASH, ENCODER_ID_4BIT) is not None
    assert cache.resolve_cached(PROMPT_HASH, ENCODER_ID_8BIT) is None
    # And it says so by name rather than quietly returning the other precision.
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        cache.resolve(PROMPT, encoder_id=ENCODER_ID_8BIT)
    assert excinfo.value.name == "encoder_variant_not_cached"
    assert ENCODER_ID_4BIT in excinfo.value.message


def test_the_reference_vector_wins_when_a_prompt_has_both(encoder_app, embeddings_dir):
    """The precedence rule, stated as a test: reference first, always.

    A locally-encoded vector is a different condition; using it for a prompt that
    has a reference vector would silently break the comparison with the golden clip.
    """
    _write_vector(
        embeddings_dir / ENCODER_ID_4BIT, PROMPT,
        encoder_id=ENCODER_ID_4BIT,
        source_fields={"encoder_id": ENCODER_ID_4BIT, "quantization": QUANTIZATION_4BIT},
        salt="mlx",
    )
    encoder_app.embeddings.refresh()
    payload = encoder_app.plan(prompt=PROMPT, duration=2.0, noise="random")
    assert payload["embedding"]["source"] == "reference"
    assert payload["embedding"]["source_kind"] == "reference"
    assert payload["embedding"]["encoder_id"] == REFERENCE_ENCODER_ID
    assert payload["embedding"]["embedding_path"].endswith(f"{PROMPT_HASH}.npy")
    # Nothing was encoded: the reference vector was already there.
    assert encoder_app.local_encoder.calls == []
    assert payload["timing"]["encode_ms"] is None
    assert payload["timing"]["cache_hit"] is True


# --------------------------------------------------------------------------- #
# free text: the cache misses, the encoder answers
# --------------------------------------------------------------------------- #
def test_free_text_is_encoded_locally_and_lands_in_its_own_cache_slot(encoder_app, embeddings_dir):
    payload = encoder_app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    digest = pipeline.prompt_hash(WRONG_PROMPT)

    assert [call["prompt"] for call in encoder_app.local_encoder.calls] == [WRONG_PROMPT]
    assert payload["embedding"]["text"] == WRONG_PROMPT
    assert payload["embedding"]["prompt_hash"] == digest
    assert payload["embedding"]["source"] == "local"
    assert payload["embedding"]["source_kind"] == "local-mlx"
    assert payload["embedding"]["encoder_id"] == ENCODER_ID_4BIT
    assert payload["embedding"]["quantization"] == QUANTIZATION_4BIT
    assert "local MLX" in payload["embedding"]["provenance"]["label"]
    # The vector is on disk under the local sub-directory, keyed by both halves.
    cached = embeddings_dir / ENCODER_ID_4BIT / f"{digest}.npy"
    assert cached.is_file()
    sidecar = json.loads(cached.with_suffix(".json").read_text(encoding="utf-8"))
    assert sidecar["encoder_id"] == ENCODER_ID_4BIT
    assert sidecar["quantization"] == QUANTIZATION_4BIT
    assert sidecar["prompt"] == WRONG_PROMPT
    # ...and it is now a cache entry, so a second request does not re-encode.
    # The encoder reported what it cost, and the request's total covers it. The two
    # are measured independently (adapter vs. server), so the total may legitimately
    # come out a hair below the adapter's own measurement of a sub-millisecond fake.
    assert payload["timing"]["encode_ms"] is not None
    assert 0.0 <= payload["timing"]["encode_ms"] < 60_000.0
    assert payload["timing"]["total_ms"] > 0.0
    assert encoder_app.embeddings.resolve_cached(digest, ENCODER_ID_4BIT) is not None

    again = encoder_app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    assert len(encoder_app.local_encoder.calls) == 1
    assert again["timing"]["cache_hit"] is True
    assert again["timing"]["encode_ms"] is None


def test_by_text_and_by_hash_agree_for_a_locally_encoded_prompt(encoder_app):
    digest = pipeline.prompt_hash(WRONG_PROMPT)
    by_text = encoder_app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    by_hash = encoder_app.plan(prompt=digest, duration=2.0, noise="random")
    assert by_text["embedding"]["embedding_sha256"] == by_hash["embedding"]["embedding_sha256"]
    assert encoder_app.local_encoder.calls == [encoder_app.local_encoder.calls[0]]


def test_the_encoded_vector_is_what_conditions_the_clip(encoder_app):
    """Not just reported: the generated frames change with the encoded condition."""
    same = encoder_app.plan(prompt=PROMPT, duration=2.0, noise="random")
    other = encoder_app.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    a = _decode(same["posed_joints_b64"])
    b = _decode(other["posed_joints_b64"])
    assert a.shape == b.shape
    assert float(np.abs(a - b).max()) > 1e-6
    assert same["embedding"]["embedding_sha256"] != other["embedding"]["embedding_sha256"]


# --------------------------------------------------------------------------- #
# negative controls: named refusals, never a substitution
# --------------------------------------------------------------------------- #
def test_negative_a_missing_local_encoder_is_named_and_encodes_nothing(embeddings_dir, viewer):
    """The fault: the cache misses and no encoder is configured at all.

    The tempting wrong behaviour is to run the one prompt that *is* cached -- a
    convincing walking clip under the wrong caption. The viewer must instead refuse
    by name, and it may not claim to have encoded anything.

    ``no_cached_embedding_for_prompt`` is the right name here rather than
    ``local_encoder_unavailable``: with no encoder configured there is no encoder to
    be unavailable, so the honest statement is the one about the cache. The message
    says which of the two situations it is.
    """
    application = viewer.ViewerApp(embeddings_dir=embeddings_dir)  # no encoder
    application.embeddings.refresh()
    application.runner = conftest.FakeRunner()
    with pytest.raises(viewer.ViewerError) as excinfo:
        application.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    assert excinfo.value.name == "no_cached_embedding_for_prompt"
    assert "cache-only lookup" in (excinfo.value.detail or "")
    assert application.runner.calls == []
    assert application.embedding_cache == {}
    # And the miss did not invent a cache entry.
    assert not any(path.is_dir() for path in embeddings_dir.iterdir())
    assert application.embeddings.resolve_cached(pipeline.prompt_hash(WRONG_PROMPT)) is None


def test_negative_an_unavailable_encoder_reports_its_reason(embeddings_dir, viewer):
    """A *configured but unusable* encoder is a different, equally named failure."""
    fake = FakeLocalEncoder(available=False, reason="missing manifest for 4-bit")
    application = viewer.ViewerApp(embeddings_dir=embeddings_dir, local_encoder=fake)
    application.embeddings.refresh()
    application.runner = conftest.FakeRunner()
    with pytest.raises(viewer.ViewerError) as excinfo:
        application.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    assert excinfo.value.name == "local_encoder_unavailable"
    assert "missing manifest for 4-bit" in excinfo.value.message
    assert application.runner.calls == []
    # It never even attempted an encode.
    assert fake.calls == []


def test_negative_an_encoder_that_fails_is_named_with_the_original_stderr(embeddings_dir, viewer):
    """The fault: the encoder runs and refuses (an over-length prompt, say).

    The substitution this guards against is generating from the *cached* prompt while
    the page shows the requested one. Nothing may be sampled.
    """
    fake = FakeLocalEncoder(
        fail_with="local_encoder_failed",
        fail_message="encoder refused: prompt is 640 tokens, over max_length 512",
    )
    application = viewer.ViewerApp(embeddings_dir=embeddings_dir, local_encoder=fake)
    application.embeddings.refresh()
    application.runner = conftest.FakeRunner()
    with pytest.raises(viewer.ViewerError) as excinfo:
        application.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    assert excinfo.value.name == "local_encoder_failed"
    assert excinfo.value.status == 502
    assert "over max_length 512" in excinfo.value.message
    # The encoder's own words are carried through, not paraphrased.
    assert "fake stderr tail" in excinfo.value.detail
    assert application.runner.calls == []
    assert application.embedding_cache == {}


def test_a_soft_encoder_failure_is_remembered_so_the_second_request_is_cheap(
    embeddings_dir, viewer
):
    """A refusal that cannot change is not re-run per request, and stays identical."""
    fake = FakeLocalEncoder(fail_with="local_encoder_failed", fail_message="too long")
    application = viewer.ViewerApp(embeddings_dir=embeddings_dir, local_encoder=fake)
    application.embeddings.refresh()
    application.runner = conftest.FakeRunner()
    for _ in range(3):
        with pytest.raises(viewer.ViewerError) as excinfo:
            application.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
        assert excinfo.value.name == "local_encoder_failed"
        assert excinfo.value.message == "too long"
    assert len(fake.calls) == 1


def test_negative_mixing_two_encoders_vectors_is_refused(embeddings_dir, viewer):
    """The fault: a cached vector from a *different* encoder than the one configured.

    The tempting wrong behaviour is to reuse it because it is "the vector for this
    prompt". It is not: it is a different artifact's vector, and conditioning on it
    while the HUD names the configured encoder is exactly the silent mix this cache
    layout exists to prevent. The viewer must treat it as a miss and encode, or fail
    by name -- but it must never report it as the configured encoder's vector.
    """
    # A vector labelled with the *8-bit* encoder, while the app is configured for 4-bit.
    _write_vector(
        embeddings_dir / ENCODER_ID_8BIT, WRONG_PROMPT,
        encoder_id=ENCODER_ID_8BIT,
        source_fields={"encoder_id": ENCODER_ID_8BIT, "quantization": QUANTIZATION_8BIT},
        salt="mlx-8bit",
    )
    fake = FakeLocalEncoder(bits=4)
    application = viewer.ViewerApp(embeddings_dir=embeddings_dir, local_encoder=fake)
    application.embeddings.refresh()
    application.runner = conftest.FakeRunner()
    payload = application.plan(prompt=WRONG_PROMPT, duration=2.0, noise="random")
    # It encoded afresh rather than reusing the 8-bit vector...
    assert [call["prompt"] for call in fake.calls] == [WRONG_PROMPT]
    assert payload["embedding"]["encoder_id"] == ENCODER_ID_4BIT
    assert payload["embedding"]["quantization"] == QUANTIZATION_4BIT
    # ...and the 8-bit vector is what it claims to be: still addressable, not promoted.
    digest = pipeline.prompt_hash(WRONG_PROMPT)
    assert application.embeddings.resolve_cached(digest, ENCODER_ID_8BIT)["quantization"] == (
        QUANTIZATION_8BIT
    )
    assert len(application.embeddings.variants(digest)) == 2


def test_one_install_never_touches_another_encoders_vector(embeddings_dir):
    """Two encoders' vectors for one prompt are two files, in two directories.

    This is the negative control for the layout itself. With a single flat
    ``mlx/<prompt-hash>.npy`` the second precision would silently replace the first
    and the cache would hold one vector claiming to be two; here the install has to
    land beside it, and both keys survive.
    """
    digest = pipeline.prompt_hash(WRONG_PROMPT)
    eight_bit = embeddings_dir / ENCODER_ID_8BIT
    _write_vector(
        eight_bit, WRONG_PROMPT,
        encoder_id=ENCODER_ID_8BIT,
        source_fields={"encoder_id": ENCODER_ID_8BIT, "quantization": QUANTIZATION_8BIT},
        salt="mlx-8bit",
    )
    before = (eight_bit / f"{digest}.npy").read_bytes()

    staged = embeddings_dir.parent / "staged.npy"
    np.save(staged, deterministic_vector(WRONG_PROMPT, salt="mlx-4bit"))
    result = EncodeResult(
        prompt=WRONG_PROMPT,
        prompt_hash=digest,
        encoder_id=ENCODER_ID_4BIT,
        vector_path=staged,
        embedding_sha256=pipeline.sha256_file(staged),
        shape=[1, 1, 4096],
        encode_ms=1.0,
        quantization=QUANTIZATION_4BIT,
    )
    cache = ardy_viewer.EmbeddingCache(embeddings_dir)
    cache.refresh()
    entry = cache.install(result)

    # The new vector landed in its own directory, under both halves of its own key.
    # The cache state says where it went, and it is the same place the filesystem does.
    assert entry["source"] == "local"
    assert entry["encoder_id"] == ENCODER_ID_4BIT
    assert cache.local_dir_for(ENCODER_ID_4BIT) == embeddings_dir / ENCODER_ID_4BIT
    assert cache.local_dir_for(ENCODER_ID_4BIT) == Path(entry["path_abs"]).parent
    assert (embeddings_dir / ENCODER_ID_4BIT / f"{digest}.npy").is_file()
    # ...and the 8-bit vector is byte-for-byte untouched, still addressable as itself.
    assert (eight_bit / f"{digest}.npy").read_bytes() == before
    assert cache.resolve_cached(digest, ENCODER_ID_8BIT)["quantization"] == QUANTIZATION_8BIT
    assert cache.resolve_cached(digest, ENCODER_ID_4BIT)["quantization"] == QUANTIZATION_4BIT
    assert len(cache.variants(digest)) == 2
    # Each prompt is listed once, as the vector a request would actually use: the
    # fixture's reference prompt stays the reference, the newly-encoded one is local.
    assert sorted(item["source"] for item in cache.list()) == ["local", "reference"]


def test_negative_a_mislabelled_file_in_an_encoders_directory_is_refused(embeddings_dir):
    """A directory *is* an encoder's claim; a contradicting sidecar is a refusal."""
    digest = pipeline.prompt_hash(WRONG_PROMPT)
    target = embeddings_dir / ENCODER_ID_4BIT
    _write_vector(
        target, WRONG_PROMPT,
        encoder_id=ENCODER_ID_8BIT,
        source_fields={"encoder_id": ENCODER_ID_8BIT, "quantization": QUANTIZATION_8BIT},
        salt="wrong-label",
    )
    staged = embeddings_dir.parent / "staged2.npy"
    np.save(staged, deterministic_vector(WRONG_PROMPT, salt="mlx-4bit"))
    result = EncodeResult(
        prompt=WRONG_PROMPT,
        prompt_hash=digest,
        encoder_id=ENCODER_ID_4BIT,
        vector_path=staged,
        embedding_sha256=pipeline.sha256_file(staged),
        shape=[1, 1, 4096],
        encode_ms=1.0,
        quantization=QUANTIZATION_4BIT,
    )
    cache = ardy_viewer.EmbeddingCache(embeddings_dir)
    cache.refresh()
    with pytest.raises(ardy_viewer.ViewerError) as excinfo:
        cache.install(result)
    assert excinfo.value.name == "local_encoder_failed"
    assert ENCODER_ID_8BIT in excinfo.value.message


def test_negative_a_wrong_shaped_vector_is_refused(tmp_path):
    """A (4096,) vector would broadcast into something the model accepts."""
    encoder = LocalEncoder(repo=tmp_path / "repo", bits=4)
    encoder._status = encoder.probe()  # noqa: SLF001 - the adapter's own cache
    staged = tmp_path / "flat.npy"
    np.save(staged, np.zeros((4096,), dtype=np.float32))
    result = encoder._read_result(  # noqa: SLF001 - the validation under test
        prompt="x", prompt_hash="0" * 16, vector_path=staged, encode_ms=1.0
    )
    assert isinstance(result, LocalEncoderError)
    assert result.name == "local_encoder_failed"
    assert "(4096,)" in result.message


def test_negative_a_non_finite_vector_is_refused(tmp_path):
    encoder = LocalEncoder(repo=tmp_path / "repo", bits=4)
    encoder._status = encoder.probe()  # noqa: SLF001
    staged = tmp_path / "nan.npy"
    vector = np.zeros((1, 1, 4096), dtype=np.float32)
    vector[0, 0, 3] = np.nan
    np.save(staged, vector)
    result = encoder._read_result(  # noqa: SLF001
        prompt="x", prompt_hash="0" * 16, vector_path=staged, encode_ms=1.0
    )
    assert isinstance(result, LocalEncoderError)
    assert "non-finite" in result.message


def test_negative_a_vector_labelled_with_another_encoder_is_refused(tmp_path):
    """The subprocess must have used *this* artifact, or the vector is not ours."""
    repo = tmp_path / "llm2vec-mlx"
    (repo / "scripts").mkdir(parents=True)
    (repo / "scripts" / "encode.py").write_text("# stub\n", encoding="utf-8")
    (repo / ".venv" / "bin").mkdir(parents=True)
    (repo / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    (repo / "weights").mkdir(parents=True)
    (repo / "weights" / "manifest-4bit.json").write_text(
        json.dumps(MANIFEST_4BIT), encoding="utf-8"
    )
    encoder = LocalEncoder(repo=repo, bits=4)
    encoder._status = encoder.probe()  # noqa: SLF001
    staged = tmp_path / "vector.npy"
    np.save(staged, np.zeros((1, 1, 4096), dtype=np.float32))
    staged.with_suffix(".json").write_text(
        json.dumps(
            {
                "encoder_id": "mlx-llm2vec-8-bit-group-64-no-exceptions",
                "encoder": "mlx-llm2vec",
                "quantization": QUANTIZATION_8BIT,
            }
        ),
        encoding="utf-8",
    )
    result = encoder._read_result(  # noqa: SLF001
        prompt="x", prompt_hash="0" * 16, vector_path=staged, encode_ms=1.0
    )
    assert isinstance(result, LocalEncoderError)
    assert result.name == "local_encoder_failed"
    assert "8-bit" in result.message and ENCODER_ID_4BIT in result.message


# --------------------------------------------------------------------------- #
# the file interface, against the real sibling checkout
# --------------------------------------------------------------------------- #
REAL_REPO = DEFAULT_LLM2VEC_REPO
REAL_ENCODER_MANIFEST = REAL_REPO / "weights" / "manifest-4bit.json"
REAL_ENCODER_INTERPRETER = REAL_REPO / ".venv" / "bin" / "python"
REAL_ENCODER_SCRIPT = REAL_REPO / "scripts" / "encode.py"
REAL_ENCODER_AVAILABLE = (
    REAL_ENCODER_MANIFEST.is_file()
    and REAL_ENCODER_INTERPRETER.is_file()
    and REAL_ENCODER_SCRIPT.is_file()
)

#: The reference bf16 vector for the one prompt the authorised reference run
#: encoded. It is weight-derived and not committed, so the comparison skips with it.
REFERENCE_VECTOR = ROOT / "artifacts" / "golden-run" / "raw" / "text_feat.npy"


def test_the_default_repo_points_at_the_sibling_checkout():
    """The decoupling is a path, so the path has to be the documented one."""
    assert REAL_REPO.name == "llm2vec-mlx"
    assert "dev" in REAL_REPO.parts


@pytest.mark.skipif(
    not REAL_ENCODER_AVAILABLE,
    reason=(
        "the real llm2vec-mlx encoder is not on disk (its weights are weight-derived "
        "and never committed, per NOTICE); this is the only test that needs it"
    ),
)
def test_the_real_encoder_answers_through_the_file_interface(tmp_path):
    """One real subprocess: the other repository's python, its script, its two files.

    Skipped unless the sibling checkout and its 4-bit manifest are present. What it
    proves and nothing else can: that the paths, the CLI flags and the sidecar field
    names this adapter relies on are the ones ``llm2vec-mlx`` actually implements.
    """
    encoder = LocalEncoder(repo=REAL_REPO, bits=4, timeout_s=600)
    status = encoder.probe()
    assert status.available, status.reason
    assert status.quantization == QUANTIZATION_4BIT
    assert status.encoder_id == ENCODER_ID_4BIT

    prompt = "A person walks forward."
    result = encoder.encode(
        prompt, prompt_hash=pipeline.prompt_hash(prompt), out_dir=tmp_path
    )
    assert not isinstance(result, LocalEncoderError), result
    assert result.shape == [1, 1, 4096]
    assert result.quantization == QUANTIZATION_4BIT
    assert result.encoder_id == ENCODER_ID_4BIT
    # The full quantization name is in the sidecar the encoder wrote -- not only in
    # this adapter's summary of it.
    assert result.sidecar["bits"] == 4
    assert result.sidecar["group_size"] == 32
    # The recorded git sha is a real revision of the sibling repository.
    assert len(str(result.sidecar.get("git_sha", ""))) >= 7
    vector = np.load(result.vector_path, allow_pickle=False)
    assert vector.dtype == np.float32 and np.isfinite(vector).all()


@pytest.mark.skipif(
    not REAL_ENCODER_AVAILABLE or not REFERENCE_VECTOR.is_file(),
    reason=(
        "needs both the real encoder and the reference bf16 vector "
        "(artifacts/golden-run/raw/text_feat.npy is weight-derived, never committed)"
    ),
)
def test_the_real_encoder_is_close_to_the_reference_vector(tmp_path):
    """The whole point of using a local vector: it conditions nearly like the real one.

    The threshold here is deliberately *looser* than the encoder repository's own
    acceptance floor (0.985 for the 4-bit artifact). This test is not the acceptance
    measurement -- that lives with the encoder and is recorded in its decision
    record. What it catches is this repository wiring the wrong artifact: a vector
    from a random-weights or 1-bit build would land far below 0.9.
    """
    encoder = LocalEncoder(repo=REAL_REPO, bits=4, timeout_s=600)
    prompt = "A person walks forward."
    result = encoder.encode(
        prompt, prompt_hash=pipeline.prompt_hash(prompt), out_dir=tmp_path
    )
    assert not isinstance(result, LocalEncoderError), result
    mine = np.load(result.vector_path, allow_pickle=False).reshape(-1).astype(np.float64)
    reference = np.load(REFERENCE_VECTOR, allow_pickle=False).reshape(-1).astype(np.float64)
    cosine = float(mine @ reference / (np.linalg.norm(mine) * np.linalg.norm(reference)))
    assert cosine > 0.98, cosine
    # And it is not *identical*: the reference vector was produced by a different
    # encoder on a different machine, and claiming bit-parity would be a lie.
    assert cosine < 1.0
