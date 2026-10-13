#!/usr/bin/env python3
"""Read-only inspector for ARDY ``safetensors`` checkpoints.

Parses the safetensors header (8-byte little-endian header length, then JSON)
and reports the tensor inventory: names, dtypes, shapes, parameter counts and
byte totals. It never reads tensor payloads and never imports torch/mlx, so it
is safe to run against the pristine reference checkpoints.

Used by A to verify weight name/shape parity tensor by tensor during conversion
(README roadmap step 2). This is a *probe*, not a converter: it writes nothing
and produces no motion.

Examples
--------
    ./.venv/bin/python scripts/ardy_inspect.py denoiser.safetensors
    ./.venv/bin/python scripts/ardy_inspect.py tokenizer.safetensors --names
    ./.venv/bin/python scripts/ardy_inspect.py a.safetensors --diff b.safetensors
"""

from __future__ import annotations

import argparse
import json
import struct
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# safetensors dtype -> bytes per element (spec, as of the 0.4 header format).
DTYPE_BYTES = {
    "BOOL": 1, "U8": 1, "I8": 1, "F8": 1,
    "I16": 2, "U16": 2, "F16": 2, "BF16": 2,
    "I32": 4, "U32": 4, "F32": 4,
    "I64": 8, "U64": 8, "F64": 8,
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


def describe(path: str | Path, *, show_names: bool, depth: int) -> int:
    header, header_len, file_size = read_header(path)
    tensors = tensors_of(header)

    print(f"file          {path}")
    print(f"size          {file_size:,} bytes")
    print(f"header        {header_len:,} bytes"
          + (f"  metadata={header['__metadata__']}" if "__metadata__" in header else ""))
    print(f"tensors       {len(tensors)}")

    if not tensors:
        return 0

    by_dtype: Dict[str, int] = {}
    total_params = 0
    total_bytes = 0
    for entry in tensors.values():
        by_dtype[entry["dtype"]] = by_dtype.get(entry["dtype"], 0) + 1
        total_params += param_count(entry)
        total_bytes += entry["data_offsets"][1] - entry["data_offsets"][0]

    print(f"parameters    {total_params:,}")
    print(f"payload       {total_bytes:,} bytes")
    print(f"dtypes        " + ", ".join(f"{k}x{v}" for k, v in sorted(by_dtype.items())))

    print(f"--- top-level prefixes (first {depth} components) ---")
    for key, count in prefix_tree(list(tensors), depth)[:40]:
        print(f"{count:6d}  {key}")

    if show_names:
        print("--- tensors ---")
        for name in sorted(tensors):
            entry = tensors[name]
            shape = "x".join(str(d) for d in entry["shape"])
            print(f"{name}  {entry['dtype']}  [{shape}]")

    problems = validate(header, header_len, file_size, path)
    if problems:
        print(f"!!! {len(problems)} problem(s) found:")
        for problem in problems[:20]:
            print(f"    {problem}")
        return 1
    print("validation    OK (contiguous data_offsets, shapes match dtypes)")
    return 0


def diff(a: str | Path, b: str | Path) -> int:
    """Compare two tensor inventories by name, dtype and shape."""
    ta = tensors_of(read_header(a)[0])
    tb = tensors_of(read_header(b)[0])
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))
    changed = sorted(
        name for name in set(ta) & set(tb)
        if (ta[name]["dtype"], tuple(ta[name]["shape"])) != (tb[name]["dtype"], tuple(tb[name]["shape"]))
    )

    print(f"A {a}: {len(ta)} tensors")
    print(f"B {b}: {len(tb)} tensors")
    print(f"only in A     {len(only_a)}")
    for name in only_a[:20]:
        print(f"    - {name}")
    print(f"only in B     {len(only_b)}")
    for name in only_b[:20]:
        print(f"    + {name}")
    print(f"shape/dtype   {len(changed)}")
    for name in changed[:20]:
        print(f"    ~ {name}: {ta[name]['dtype']}{ta[name]['shape']} -> {tb[name]['dtype']}{tb[name]['shape']}")

    exact = not (only_a or only_b or changed)
    print("IDENTICAL inventory" if exact else "DIFFERENT inventory")
    return 0 if exact else 1


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Inspect ARDY safetensors checkpoints (read-only).")
    parser.add_argument("path", help="safetensors file to inspect")
    parser.add_argument("--names", action="store_true", help="also print every tensor name/dtype/shape")
    parser.add_argument("--depth", type=int, default=2, help="prefix depth for the grouping summary (default: 2)")
    parser.add_argument("--diff", metavar="OTHER", default=None, help="compare inventory against another safetensors file")
    parser.add_argument("--json", action="store_true", help="print the raw JSON header instead of a summary")
    args = parser.parse_args(argv)

    target = Path(args.path)
    if not target.is_file():
        print(f"no such file: {target}", file=sys.stderr)
        return 2

    try:
        if args.json:
            header, _, _ = read_header(target)
            json.dump(header, sys.stdout, indent=2, sort_keys=True)
            print()
            return 0
        if args.diff:
            return diff(target, args.diff)
        return describe(target, show_names=args.names, depth=args.depth)
    except SafetensorsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
