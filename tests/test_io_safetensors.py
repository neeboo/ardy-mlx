"""Tests for the safetensors container I/O used by the inspector and the converter.

The fixture builds real containers on disk, so the reader/writer pair is tested
against the actual byte layout rather than a mocked header.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ardy_mlx.io import safetensors as st  # noqa: E402


def write_minimal(path: Path, tensors: dict) -> None:
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
        "pose_net.encoder.input_proj.weight": np.arange(12, dtype=np.float32).reshape(3, 4),
        "pose_net.encoder.output_proj.weight": (np.arange(8, dtype=np.float32) / 7).reshape(2, 4),
        "denoiser.backbone.root_model.output_linear.bias": np.array([1.5, -2.25], dtype=np.float32),
    }
    path = tmp_path / "sample.safetensors"
    write_minimal(path, tensors)
    return path, tensors


def test_header_reads_names_shapes_offsets(sample):
    path, tensors = sample
    header, header_len, file_size = st.read_header(path)
    assert set(st.tensors_of(header)) == set(tensors)
    assert file_size == path.stat().st_size
    assert header_len > 0
    assert st.param_count(header["pose_net.encoder.input_proj.weight"]) == 12


def test_validate_accepts_and_rejects(sample):
    path, _ = sample
    header, header_len, file_size = st.read_header(path)
    assert st.validate(header, header_len, file_size, path) == []

    header["pose_net.encoder.output_proj.weight"]["shape"] = [2, 999]
    assert any("imply" in p for p in st.validate(header, header_len, file_size, path))

    header["pose_net.encoder.output_proj.weight"]["dtype"] = "F7"
    assert any("unknown dtype" in p for p in st.validate(header, header_len, file_size, path))


def test_read_payload_is_the_exact_raw_bytes(sample):
    path, tensors = sample
    for name, array in tensors.items():
        assert st.read_payload(path, name) == array.tobytes()


def test_iter_tensors_yields_every_payload(sample):
    path, tensors = sample
    seen = {name: payload for name, _entry, payload in st.iter_tensors(path)}
    assert set(seen) == set(tensors)
    for name, array in tensors.items():
        assert seen[name] == array.tobytes()


def test_writer_roundtrips_payload_bytes(tmp_path, sample):
    """The converter's byte-exactness claim depends on this being true."""
    _path, tensors = sample
    out = tmp_path / "out.safetensors"
    st.write_safetensors(out, tensors, metadata={"format": "pt"})
    header, _hl, _fs = st.read_header(out)
    assert header["__metadata__"] == {"format": "pt"}
    for name, array in tensors.items():
        assert st.read_payload(out, name) == array.tobytes()
        assert list(st.tensors_of(header)[name]["shape"]) == list(array.shape)


def test_writer_detects_a_real_change(tmp_path, sample):
    """改坏就红: flip one byte and the byte comparison must notice."""
    _path, tensors = sample
    out = tmp_path / "out.safetensors"
    st.write_safetensors(out, tensors)
    mutated = {k: v.copy() for k, v in tensors.items()}
    target = "denoiser.backbone.root_model.output_linear.bias"
    mutated[target][0] += 1.0
    other = tmp_path / "other.safetensors"
    st.write_safetensors(other, mutated)
    assert st.read_payload(out, target) != st.read_payload(other, target)
    assert st.read_payload(out, "pose_net.encoder.input_proj.weight") == st.read_payload(
        other, "pose_net.encoder.input_proj.weight"
    )


def test_prefix_tree_groups_by_first_components(sample):
    _path, tensors = sample
    tree = dict(st.prefix_tree(list(tensors), depth=2))
    assert tree["pose_net.encoder"] == 2
    assert tree["denoiser.backbone"] == 1


def test_missing_tensor_raises(sample):
    path, _ = sample
    with pytest.raises(st.SafetensorsError):
        st.read_payload(path, "does.not.exist")
