"""Tests for the read-only safetensors header inspector (scripts/ardy_inspect.py).

The fixture builds a real (tiny) safetensors container, so the parser is tested
against the actual byte layout rather than against a mocked header.
"""

from __future__ import annotations

import importlib.util
import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load_inspector():
    path = ROOT / "scripts" / "ardy_inspect.py"
    spec = importlib.util.spec_from_file_location("ardy_inspect", path)
    assert spec and spec.loader, f"cannot load {path}"
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


inspector = _load_inspector()


def write_safetensors(path: Path, tensors: dict) -> None:
    """Write a minimal but spec-shaped safetensors file."""
    header, offset = {}, 0
    for name, array in tensors.items():
        payload = array.tobytes()
        header[name] = {
            "dtype": "F32",
            "shape": list(array.shape),
            "data_offsets": [offset, offset + len(payload)],
        }
        offset += len(payload)
    blob = json.dumps(header).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for array in tensors.values():
            fh.write(array.tobytes())


@pytest.fixture
def sample(tmp_path):
    tensors = {
        "pose_net.encoder.input_proj.weight": np.zeros((512, 1300), dtype=np.float32),
        "pose_net.encoder.output_proj.weight": np.zeros((128, 512), dtype=np.float32),
        "pose_net.decoder.output_proj.weight": np.zeros((1316, 512), dtype=np.float32),
    }
    path = tmp_path / "tokenizer.safetensors"
    write_safetensors(path, tensors)
    return path, tensors


def test_read_header_roundtrip(sample):
    path, tensors = sample
    header, header_len, file_size = inspector.read_header(path)
    assert set(inspector.tensors_of(header)) == set(tensors)
    assert header_len > 0
    assert file_size == path.stat().st_size
    assert header["pose_net.decoder.output_proj.weight"]["shape"] == [1316, 512]


def test_validate_accepts_a_well_formed_file(sample):
    path, _ = sample
    header, header_len, file_size = inspector.read_header(path)
    assert inspector.validate(header, header_len, file_size, path) == []


def test_validate_rejects_shape_offset_mismatch(sample):
    """改坏就红: lie about a shape and validation must complain."""
    path, _ = sample
    header, header_len, file_size = inspector.read_header(path)
    header["pose_net.encoder.output_proj.weight"]["shape"] = [128, 999]
    problems = inspector.validate(header, header_len, file_size, path)
    assert any("imply" in p for p in problems), problems


def test_validate_rejects_unknown_dtype(sample):
    path, _ = sample
    header, header_len, file_size = inspector.read_header(path)
    header["pose_net.encoder.output_proj.weight"]["dtype"] = "F4"
    assert any("unknown dtype" in p for p in inspector.validate(header, header_len, file_size, path))


def test_describe_reports_counts(sample, capsys):
    path, _ = sample
    assert inspector.describe(path, show_names=False, depth=2) == 0
    out = capsys.readouterr().out
    assert "tensors       3" in out
    assert "parameters    " in out
    assert "validation    OK" in out


def test_describe_names_lists_every_tensor(sample, capsys):
    path, tensors = sample
    inspector.describe(path, show_names=True, depth=2)
    out = capsys.readouterr().out
    for name in tensors:
        assert name in out


def test_param_count_and_prefix_tree(sample):
    _, tensors = sample
    assert inspector.param_count({"shape": [1316, 512]}) == 1316 * 512
    assert inspector.param_count({"shape": []}) == 1
    tree = dict(inspector.prefix_tree(list(tensors), depth=2))
    assert tree["pose_net.encoder"] == 2
    assert tree["pose_net.decoder"] == 1


def test_diff_identical_and_changed(sample, tmp_path):
    path, tensors = sample
    clone = tmp_path / "clone.safetensors"
    write_safetensors(clone, tensors)
    assert inspector.diff(path, clone) == 0

    changed = dict(tensors)
    changed["pose_net.encoder.output_proj.weight"] = np.zeros((64, 512), dtype=np.float32)
    changed["pose_net.extra.bias"] = np.zeros((8,), dtype=np.float32)
    other = tmp_path / "other.safetensors"
    write_safetensors(other, changed)
    assert inspector.diff(path, other) == 1


def test_rejects_a_file_that_is_too_small(tmp_path):
    tiny = tmp_path / "tiny.safetensors"
    tiny.write_bytes(b"\x00\x01\x02")
    with pytest.raises(inspector.SafetensorsError):
        inspector.read_header(tiny)


def test_rejects_a_header_length_past_eof(tmp_path):
    bad = tmp_path / "bad.safetensors"
    bad.write_bytes(struct.pack("<Q", 10_000) + b"{}")
    with pytest.raises(inspector.SafetensorsError, match="does not fit"):
        inspector.read_header(bad)


def test_rejects_non_json_header(tmp_path):
    bad = tmp_path / "notjson.safetensors"
    blob = b"not json at all"
    bad.write_bytes(struct.pack("<Q", len(blob)) + blob)
    with pytest.raises(inspector.SafetensorsError, match="valid UTF-8 JSON"):
        inspector.read_header(bad)


def test_cli_missing_file_returns_2(tmp_path, capsys):
    assert inspector.main([str(tmp_path / "nope.safetensors")]) == 2
    assert "no such file" in capsys.readouterr().err


def test_cli_json_dumps_header(sample, capsys):
    path, _ = sample
    assert inspector.main([str(path), "--json"]) == 0
    parsed = json.loads(capsys.readouterr().out)
    assert "pose_net.decoder.output_proj.weight" in parsed
