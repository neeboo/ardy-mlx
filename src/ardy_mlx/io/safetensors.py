"""Safetensors container I/O for ARDY checkpoints (no torch dependency).

This is the single parser used by ``scripts/ardy_inspect.py`` (read-only probe)
and ``scripts/ardy_convert.py`` (weight conversion + byte-exact verification), so
the two can never disagree about a header.

The container format is the frozen 8-byte-header safetensors spec:

    <u64 little-endian header length> <UTF-8 JSON header> <data section>

where the JSON header maps every tensor name to
``{"dtype": ..., "shape": [...], "data_offsets": [start, end]}`` with offsets
relative to the start of the data section, plus an optional ``"__metadata__"``.

Nothing here imports MLX or torch: payload access is raw bytes via ``mmap``, so
the module is safe to point at the pristine reference checkpoints.
"""

from __future__ import annotations

import json
import mmap
import struct
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Tuple

__all__ = [
    "DTYPE_BYTES",
    "SafetensorsError",
    "read_header",
    "tensors_of",
    "param_count",
    "validate",
    "prefix_tree",
    "read_payload",
    "iter_tensors",
    "tensor_bytes_of",
    "write_safetensors",
]

# safetensors dtype -> bytes per element (spec, as of the 0.4 header format).
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
}

# numpy dtype string <-> safetensors dtype, for the writer and for callers that
# need to reinterpret payload bytes.
NUMPY_TO_SAFETENSORS = {
    "<f4": "F32", "<f2": "F16", "<f8": "F64",
    "<i1": "I8", "|i1": "I8", "|u1": "U8", "|b1": "BOOL",
    "<i2": "I16", "<u2": "U16", "<i4": "I32", "<u4": "U32",
    "<i8": "I64", "<u8": "U64",
}


class SafetensorsError(RuntimeError):
    """The file is not a well-formed safetensors container."""


def read_header(path: str | Path) -> Tuple[Dict, int, int]:
    """Return ``(header, header_len, file_size)`` for a safetensors file.

    The returned header maps tensor name -> ``{"dtype", "shape", "data_offsets"}``
    plus an optional ``"__metadata__"`` entry.
    """
    p = Path(path)
    file_size = p.stat().st_size
    if file_size < 8:
        raise SafetensorsError(f"{p}: too small to be safetensors ({file_size} bytes)")

    with p.open("rb") as fh:
        raw_len = fh.read(8)
        (header_len,) = struct.unpack("<Q", raw_len)
        if header_len <= 0 or 8 + header_len > file_size:
            raise SafetensorsError(
                f"{p}: header length {header_len} does not fit in {file_size} bytes"
            )
        blob = fh.read(header_len)

    try:
        header = json.loads(blob.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafetensorsError(f"{p}: header is not valid UTF-8 JSON: {exc}") from exc
    if not isinstance(header, dict):
        raise SafetensorsError(f"{p}: header is not a JSON object")
    return header, header_len, file_size


def tensors_of(header: Dict) -> Dict[str, Dict]:
    """Header entries excluding ``__metadata__``."""
    return {k: v for k, v in header.items() if k != "__metadata__"}


def param_count(entry: Dict) -> int:
    n = 1
    for dim in entry.get("shape", []):
        n *= int(dim)
    return n


def validate(header: Dict, header_len: int, file_size: int, path: str | Path) -> List[str]:
    """Return a list of problems; empty means the data section looks consistent."""
    problems: List[str] = []
    data_start = 8 + header_len
    data_bytes = file_size - data_start
    pairs = sorted(entry["data_offsets"] for entry in tensors_of(header).values())

    seen_end = 0
    for start, end in pairs:
        if start != seen_end:
            problems.append(f"data_offsets not contiguous at {seen_end} (next starts {start})")
        if end < start:
            problems.append(f"data_offsets inverted: [{start},{end}]")
        seen_end = max(seen_end, end)
    if pairs and seen_end != data_bytes:
        problems.append(f"data section ends at {seen_end} but file has {data_bytes} bytes of payload")

    for name, entry in tensors_of(header).items():
        dtype = entry.get("dtype")
        if dtype not in DTYPE_BYTES:
            problems.append(f"{name}: unknown dtype {dtype!r}")
            continue
        expected = param_count(entry) * DTYPE_BYTES[dtype]
        got = entry["data_offsets"][1] - entry["data_offsets"][0]
        if expected != got:
            problems.append(f"{name}: shape/dtype imply {expected} bytes, offsets span {got}")
    return problems


def prefix_tree(names: List[str], depth: int = 2) -> List[Tuple[str, int]]:
    """Count names by their first ``depth`` dot-separated components."""
    counts: Dict[str, int] = {}
    for name in names:
        key = ".".join(name.split(".")[:depth])
        counts[key] = counts.get(key, 0) + 1
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))


def read_payload(path: str | Path, name: str, header: Dict | None = None) -> bytes:
    """Exact bytes of one tensor's payload (source file bytes, no reinterpretation)."""
    if header is None:
        header = read_header(path)[0]
    _, header_len, _ = read_header(path)
    tensors = tensors_of(header)
    if name not in tensors:
        raise SafetensorsError(f"{path}: no tensor named {name!r}")
    start, end = tensors[name]["data_offsets"]
    with open(path, "rb") as fh:
        fh.seek(8 + header_len + start)
        blob = fh.read(end - start)
    if len(blob) != end - start:
        raise SafetensorsError(f"{path}: short read for {name!r}")
    return blob


def tensor_bytes_of(path: str | Path) -> Dict[str, bytes]:
    """``{name: payload bytes}`` for a whole file, read through one ``mmap``.

    Used for byte-exact conversion checks; memory cost is the file size because
    ``bytes`` copies out of the map.
    """
    header, header_len, file_size = read_header(path)
    tensors = tensors_of(header)
    out: Dict[str, bytes] = {}
    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            base = 8 + header_len
            for name, entry in tensors.items():
                start, end = entry["data_offsets"]
                out[name] = mm[base + start: base + end]
    return out


def iter_tensors(path: str | Path) -> Iterator[Tuple[str, Dict, bytes]]:
    """Yield ``(name, header_entry, payload_bytes)`` in header order."""
    header, header_len, _ = read_header(path)
    with open(path, "rb") as fh:
        with mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ) as mm:
            base = 8 + header_len
            for name, entry in tensors_of(header).items():
                start, end = entry["data_offsets"]
                yield name, entry, mm[base + start: base + end]


def _dtype_of(array) -> str:
    key = array.dtype.str if hasattr(array.dtype, "str") else str(array.dtype)
    if key not in NUMPY_TO_SAFETENSORS:
        raise SafetensorsError(f"unsupported numpy dtype for safetensors: {array.dtype!r}")
    return NUMPY_TO_SAFETENSORS[key]


def write_safetensors(
    path: str | Path,
    tensors: Mapping[str, object],
    metadata: Mapping[str, str] | None = None,
) -> None:
    """Write a safetensors file from numpy-compatible arrays.

    The data section is written in the given mapping's iteration order with no
    padding, so a payload round-tripped from a source file is byte-identical.
    """
    import numpy as np

    header: Dict[str, object] = {}
    offset = 0
    payloads: List[bytes] = []
    for name, array in tensors.items():
        arr = np.ascontiguousarray(array)
        blob = arr.tobytes()
        header[name] = {
            "dtype": _dtype_of(arr),
            "shape": list(arr.shape),
            "data_offsets": [offset, offset + len(blob)],
        }
        offset += len(blob)
        payloads.append(blob)
    if metadata:
        header["__metadata__"] = dict(metadata)

    blob = json.dumps(header).encode("utf-8")
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as fh:
        fh.write(struct.pack("<Q", len(blob)))
        fh.write(blob)
        for payload in payloads:
            fh.write(payload)
